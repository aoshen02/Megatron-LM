# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vLLM worker integration for MLite weight refits."""

from __future__ import annotations

import gc
import logging

import torch
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    BucketedWeightReceiver,
)
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

logger = logging.getLogger(__name__)


# Drafter tensors the refit stores as sent (no quantization or repacking).
_DRAFT_PROBE_SUFFIXES = ("eh_proj.weight", "enorm.weight", "hnorm.weight", "final_layernorm.weight")


def _needs_native_reload(vllm_config) -> bool:
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    quant_config = getattr(vllm_config, "quant_config", None)
    return isinstance(quant_config, ModelOptMixedPrecisionConfig)


def _incomplete_layers(model, loaded: set[str], skip_extra=frozenset()) -> list[str]:
    """Layers still missing a checkpoint tensor once every bucket has arrived.

    Finalize would otherwise process them from uninitialized storage or keep
    serving the previous weights.
    """
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
    from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS
    from vllm.model_executor.model_loader.reload.utils import (
        get_layer_params_buffers,
        get_tensor_load_numel,
    )

    # Attention's q/prob scale placeholders are not checkpoint tensors.
    skip = SKIP_LOAD_TENSORS | {"q_scale", "prob_scale"} | skip_extra
    incomplete = []
    for prefix, layer in model.named_modules():
        info = LAYERWISE_INFO.get(layer)
        if info is None or not info.can_load():
            continue
        # Parameters only, unlike vLLM's load_numel_total: the layers pending
        # here are attention layers, whose buffers are runtime copies of the
        # k/v scales. A padded layer would be reported, failing closed.
        expected = sum(
            get_tensor_load_numel(tensor)
            for name, tensor in get_layer_params_buffers(layer)[0].items()
            if name not in skip
        )
        if info.load_numel != expected:
            incomplete.append(prefix)
    # Loaded in place, outside layerwise accounting (e.g. the router's
    # e_score_correction_bias); the other SKIP_LOAD_TENSORS are runtime buffers.
    incomplete += [
        name
        for name, _ in model.named_parameters()
        if name.rpartition(".")[2] in SKIP_LOAD_TENSORS and name not in loaded
    ]
    return incomplete


class MLiteVLLMColocateWorkerExtension(vLLMColocateWorkerExtension):
    """Refit ModelOpt mixed-precision checkpoints through vLLM's native
    layerwise lifecycle: restore the checkpoint layout, load every bucket,
    process each completed layer once and copy the result into the original
    kernel storage. The actor sends its deployment tensors in checkpoint
    format; every other config keeps verl's path.

    ``mtp.*`` tensors (a trained MTP block) refit the speculative drafter's own
    layers, through the same lifecycle; its embedding and LM head are the
    target's modules and follow the target refit.
    """

    @torch.no_grad()
    def update_weights_from_ipc(
        self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False
    ):
        vllm_config = self.model_runner.vllm_config
        if peft_config is not None or not _needs_native_reload(vllm_config):
            return super().update_weights_from_ipc(
                peft_config=peft_config, base_sync_done=base_sync_done, use_shm=use_shm
            )
        if getattr(self, "_refit_failed", False):
            raise RuntimeError("An earlier MLite refit failed; refusing further updates")

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_processing,
            initialize_layerwise_reload,
        )

        if self.device is None:
            self.device = torch.device(f"{get_device_name()}:{self.local_rank}")
        model = self.model_runner.model
        drafter = self._get_drafter_model()
        # Only the drafter's own layers: its embedding and LM head are shared
        # with the target and reload with it.
        draft_layers = getattr(getattr(drafter, "model", None), "layers", None)
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(), device=self.device, use_shm=use_shm
        )

        received, loaded, draft_loaded = set(), set(), set()
        probes = {}
        draft_started = False
        error = None

        def load(weights, is_last):
            nonlocal error, draft_started
            if error is not None:
                return
            # Raising here would skip the ack and leave the sender blocked.
            try:
                names = [name for name, _ in weights]
                if len(set(names)) != len(names) or not received.isdisjoint(names):
                    raise RuntimeError("MLite refit received a tensor twice")
                received.update(names)
                # vLLM buffers loader arguments until a layer is complete,
                # while the receiver reuses its bucket once this returns.
                target = [(n, t.clone()) for n, t in weights if not n.startswith("mtp.")]
                draft = [(n, t.clone()) for n, t in weights if n.startswith("mtp.")]
                if target:
                    loaded.update(model.load_weights(target))
                if draft:
                    if draft_layers is None:
                        raise RuntimeError("MLite refit received mtp.* tensors without an MTP drafter")
                    if not draft_started:
                        initialize_layerwise_reload(draft_layers)
                        draft_started = True
                    # Unprocessed drafter tensors, compared after the refit.
                    for name, tensor in draft:
                        if name.endswith(_DRAFT_PROBE_SUFFIXES):
                            probes[name.replace("mtp.layers.", "model.layers.", 1)] = tensor
                    draft_loaded.update(drafter.load_weights(draft))
            except Exception as e:
                error = e

        self._refit_failed = True
        with set_current_vllm_config(vllm_config):
            initialize_layerwise_reload(model)
            receiver.receive_weights(on_bucket_received=load)
            if error is not None:
                raise error
            if incomplete := _incomplete_layers(model, loaded):
                raise RuntimeError(f"MLite refit left layers incomplete: {incomplete}")
            if draft_started:
                local = {name.removeprefix("model.layers.") for name in draft_loaded}
                # The checkpoint and the actor carry no KV-cache scales for the
                # drafter's attention; it keeps its own.
                if incomplete := _incomplete_layers(draft_layers, local, {"k_scale", "v_scale"}):
                    raise RuntimeError(f"MLite refit left drafter layers incomplete: {incomplete}")
            finalize_layerwise_processing(model, vllm_config.model_config)
            if draft_started:
                finalize_layerwise_processing(
                    draft_layers, vllm_config.speculative_config.draft_model_config
                )
        self._refit_failed = False
        if draft_started:
            params = dict(drafter.named_parameters())
            exact = sum(
                torch.equal(params[name], value.to(params[name].dtype)) for name, value in probes.items()
            )
            logger.warning(
                "MLite refit: drafter loaded %d mtp.* tensors; %d/%d unprocessed tensors equal the sent ones",
                len(draft_loaded), exact, len(probes),
            )
            if exact != len(probes):
                raise RuntimeError("MLite refit: drafter tensors differ from the sent mtp.* tensors")
        # Release the staged checkpoint-format tensors before the KV cache wakes.
        gc.collect()
        get_torch_device().empty_cache()


__all__ = ["MLiteVLLMColocateWorkerExtension"]
