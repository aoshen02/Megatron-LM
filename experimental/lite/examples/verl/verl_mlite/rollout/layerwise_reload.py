# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Online checkpoint-format weight sync through vLLM's layerwise reload."""

from __future__ import annotations

import gc
import re
from collections.abc import Iterable, Mapping

import torch
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    BucketedWeightReceiver,
)
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl.workers.rollout.vllm_rollout.weight_update_utils import (
    drop_tied_alias_updates,
)

# Checkpoint-format tensors that vLLM creates but a Lightning NVFP4 checkpoint
# never provides, keyed by (layer kind, tensor name); see _layer_kind. Layers
# keeping them unloaded are processed by finalize; every other tensor must
# arrive. Citations are to the vLLM tree the image is built from.
ALLOWED_UNLOADED: Mapping[tuple[str, str], str] = {
    # ModelOptNvFp4FusedMoE.create_weights always registers per-expert
    # activation scales (modelopt.py:957-971); W4A16 experts never quantize
    # activations and the checkpoint/actor export has no expert input_scale.
    ("W4A16NvFp4RoutedExperts", "w13_input_scale"): "modelopt.py:957-965",
    ("W4A16NvFp4RoutedExperts", "w2_input_scale"): "modelopt.py:967-971",
    # W4A16 NVFP4 linears register a NaN input_scale only to drop it after
    # loading (_DropInputScale, modelopt.py:2519-2541, chosen at :2736).
    ("W4A16NvFp4Linear", "input_scale"): "modelopt.py:2526-2541",
    # BaseKVCacheMethod.create_weights adds q/prob scale placeholders
    # (kv_cache.py:64-71); the checkpoint stores only k_scale/v_scale.
    ("Attention", "q_scale"): "kv_cache.py:67,71",
    ("Attention", "prob_scale"): "kv_cache.py:67,71",
    # Runtime copies set by set_default_quant_scales (attention.py:133-141)
    # and recomputed from k/v scales in process_weights_after_loading
    # (kv_cache.py:76-194); never checkpoint tensors.
    ("Attention", "_k_scale"): "attention.py:136",
    ("Attention", "_v_scale"): "attention.py:137",
    ("Attention", "_q_scale"): "attention.py:138",
    ("Attention", "_prob_scale"): "attention.py:139",
}


def _layer_kind(layer: torch.nn.Module) -> str:
    """Name W4A16 NVFP4 variants so their exemptions never cover FP8 layers."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod,
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Static,
    )

    method = getattr(layer, "quant_method", None)
    if isinstance(method, ModelOptNvFp4FusedMoE) and method.use_a16:
        return "W4A16NvFp4RoutedExperts"
    if (
        isinstance(method, ModelOptLinearMethod)
        and method.spec.weight == kNvfp4Static
        and method.spec.activation is None
    ):
        return "W4A16NvFp4Linear"
    return type(layer).__name__


# Checkpoint names a model legitimately consumes without loading anything on
# this rank: MTP weights dropped by the main model, and non-local experts.
# RoutedExperts.load_weights also silently skips expert ids it has no mapping
# for, so the id must be within the model's routed experts.
_DROPPED_PREFIXES = ("mtp.",)
_EXPERT_NAME = re.compile(
    r"\.experts\.(?P<expert>\d+)\.(?:up_proj|down_proj)\."
    r"(?:weight|weight_scale|weight_scale_2|input_scale)$"
)


def require_layerwise_reload_support(vllm_config) -> None:
    """Reject quantization configs whose online reload path is unverified."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    quant_config = getattr(vllm_config, "quant_config", None)
    if not isinstance(quant_config, ModelOptMixedPrecisionConfig):
        raise NotImplementedError(
            "LayerwiseReloadWorkerExtension only supports ModelOpt "
            f"MIXED_PRECISION checkpoints, got {type(quant_config).__name__}"
        )


class LayerwiseReloadSession:
    """Load one full checkpoint-format update and audit it before finalize."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.names: set[str] = set()
        self.loaded: set[str] = set()

    def load(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load one received bucket, one tensor at a time.

        The reload retains loader arguments until every tensor of a layer has
        arrived while the receiver reuses its buffer, so tensors are cloned.
        Loading per tensor exposes names the model accepts but never loads.
        """
        for name, tensor in drop_tied_alias_updates(self.model, list(weights)):
            if name in self.names:
                raise ValueError(f"Duplicate tensor in one weight update: {name}")
            self.names.add(name)
            loaded = set(self.model.load_weights([(name, tensor.clone())]) or ())
            if not loaded and not self._may_load_nothing(name):
                raise ValueError(f"Model loaded nothing for tensor {name!r}")
            self.loaded |= loaded

    def _may_load_nothing(self, name: str) -> bool:
        if name.startswith(_DROPPED_PREFIXES):
            mapper = getattr(self.model, "hf_to_vllm_mapper", None)
            return mapper is not None and not mapper.apply_list([name])
        match = _EXPERT_NAME.search(name)
        config = getattr(self.model, "config", None)
        num_experts = getattr(config, "n_routed_experts", None)
        return (
            match is not None
            and num_experts is not None
            and int(match["expert"]) < num_experts
        )

    def deficits(
        self, allowed: Mapping[tuple[str, str], str] = ALLOWED_UNLOADED
    ) -> list[dict]:
        """List layers whose finalize would complete from stale or
        uninitialized memory. Must run before ``finalize_layerwise_reload``."""
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
        from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS
        from vllm.model_executor.model_loader.reload.utils import (
            get_tensor_load_numel,
        )

        report = []
        for prefix, layer in self.model.named_modules():
            info = LAYERWISE_INFO.get(layer)
            if info is None or not info.can_load() or info.kernel_tensors is None:
                continue
            params, buffers = info.restore_metadata
            required = {
                n: t
                for n, t in (params | buffers).items()
                if n not in SKIP_LOAD_TENSORS
            }
            loaded = {n for n, _ in info.loaded_weights}
            kind = _layer_kind(layer)
            skipped = {n for n in required if (kind, n) in allowed} - loaded
            missing = sorted(required.keys() - loaded - skipped)
            expected = info.load_numel_total - sum(
                get_tensor_load_numel(required[n]) for n in skipped
            )
            if missing or info.load_numel != expected:
                report.append(
                    {
                        "layer": prefix,
                        "type": kind,
                        "missing": missing,
                        "unloaded_allowed": sorted(skipped),
                        "load_numel": info.load_numel,
                        "expected_numel": expected,
                        "total_numel": info.load_numel_total,
                    }
                )
        for name, _ in self.model.named_parameters():
            # Loaded in place, outside layerwise accounting (SKIP_LOAD_TENSORS).
            if name.endswith(".e_score_correction_bias") and name not in self.loaded:
                report.append({"layer": name, "type": "live", "missing": [name]})
        return report


class LayerwiseReloadWorkerExtension(vLLMColocateWorkerExtension):
    """Colocated weight sync for checkpoint-format ModelOpt NVFP4/FP8 updates.

    verl's base extension treats ``ModelOptMixedPrecisionConfig`` as
    unquantized: it loads checkpoint tensors into already processed kernel
    layouts and reruns non-idempotent ``process_weights_after_loading``. This
    extension restores the checkpoint layout, loads each bucket, lets vLLM
    process every completed layer once and copies the result into the original
    kernel storage, so captured CUDA graphs keep reading valid addresses.

    Every update must carry the complete checkpoint: finalize would otherwise
    keep stale weights for absent layers and process partial layers from
    uninitialized memory. Incomplete, duplicate or unknown payloads fail closed
    and the worker refuses all later updates.
    """

    def __new__(cls, **kwargs):
        require_layerwise_reload_support(kwargs.get("vllm_config"))
        return super().__new__(cls, **kwargs)

    @torch.no_grad()
    def update_weights_from_ipc(
        self,
        peft_config: dict = None,
        base_sync_done=False,
        use_shm: bool = False,
        strict: bool = True,
    ):
        """Receive and apply one full update.

        ``strict=False`` is a diagnostic: it records every deficit, including
        allowlisted ones, in ``last_reload_audit`` and finalizes anyway.
        """
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        if peft_config is not None:
            raise NotImplementedError("Layerwise reload does not support LoRA sync")
        if self._use_mtp_drafter_weight_sync():
            raise NotImplementedError(
                "Layerwise reload does not support MTP drafter sync"
            )
        if getattr(self, "_layerwise_reload_failed", False):
            raise RuntimeError(
                "A previous layerwise reload failed; rollout weights are unknown"
            )
        if self.device is None:
            self.device = torch.device(f"{get_device_name()}:{self.local_rank}")

        model = self.model_runner.model
        vllm_config = self.model_runner.vllm_config
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(), device=self.device, use_shm=use_shm
        )
        session = LayerwiseReloadSession(model)
        callback_error: BaseException | None = None

        def on_bucket_received(weights, is_last: bool) -> None:
            nonlocal callback_error
            if callback_error is not None:
                return
            try:
                session.load(weights)
            except BaseException as exc:
                # Keep acknowledging buckets so the sender is not left waiting.
                callback_error = exc

        # Cleared only after a complete payload passed the audit and finalize
        # returned; until then layers may hold meta or stale tensors.
        self._layerwise_reload_failed = True
        self.last_reload_audit = None
        with set_current_vllm_config(vllm_config):
            initialize_layerwise_reload(model)
            receiver.receive_weights(on_bucket_received=on_bucket_received)
            if callback_error is not None:
                self.last_reload_audit = {"error": repr(callback_error)}
                raise callback_error
            deficits = session.deficits(ALLOWED_UNLOADED if strict else {})
            self.last_reload_audit = {
                "tensors": len(session.names),
                "deficits": deficits,
            }
            if strict and deficits:
                raise RuntimeError(
                    "Incomplete layerwise reload payload; refusing to finalize:\n"
                    + "\n".join(map(str, deficits))
                )
            finalize_layerwise_reload(model, vllm_config.model_config)
        self._layerwise_reload_failed = False

        # Retained bucket clones are dead now; return them before KV cache wakes.
        gc.collect()
        get_torch_device().empty_cache()
        return len(session.names)


__all__ = [
    "ALLOWED_UNLOADED",
    "LayerwiseReloadSession",
    "LayerwiseReloadWorkerExtension",
    "require_layerwise_reload_support",
]
