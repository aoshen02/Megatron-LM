# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Online checkpoint-format weight sync through vLLM's layerwise reload."""

from __future__ import annotations

import gc
from collections.abc import Iterable

import torch
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    BucketedWeightReceiver,
)
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl.workers.rollout.vllm_rollout.weight_update_utils import (
    drop_tied_alias_updates,
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


def load_checkpoint_bucket(
    model: torch.nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> int:
    """Load one received bucket into a model inside a layerwise reload.

    The reload retains loader arguments until every tensor of a layer has
    arrived, while the receiver reuses its transfer buffer for the next bucket.
    """
    weights = [
        (name, tensor.clone())
        for name, tensor in drop_tied_alias_updates(model, list(weights))
    ]
    model.load_weights(weights)
    return len(weights)


class LayerwiseReloadWorkerExtension(vLLMColocateWorkerExtension):
    """Colocated weight sync for checkpoint-format ModelOpt NVFP4/FP8 updates.

    verl's base extension treats ``ModelOptMixedPrecisionConfig`` as
    unquantized: it loads checkpoint tensors into already processed kernel
    layouts and reruns non-idempotent ``process_weights_after_loading``. This
    extension restores the checkpoint layout, loads each bucket, lets vLLM
    process every completed layer once and copies the result into the original
    kernel storage, so captured CUDA graphs keep reading valid addresses.
    """

    def __new__(cls, **kwargs):
        require_layerwise_reload_support(kwargs.get("vllm_config"))
        return super().__new__(cls, **kwargs)

    @torch.no_grad()
    def update_weights_from_ipc(
        self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False
    ):
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
        loaded = 0
        callback_error: BaseException | None = None

        def on_bucket_received(weights, is_last: bool) -> None:
            nonlocal loaded, callback_error
            if callback_error is not None:
                return
            try:
                loaded += load_checkpoint_bucket(model, weights)
            except BaseException as exc:
                # Keep acknowledging buckets so the sender is not left waiting.
                callback_error = exc

        # Until finalize returns, layers may still hold meta tensors.
        self._layerwise_reload_failed = True
        with set_current_vllm_config(vllm_config):
            initialize_layerwise_reload(model)
            receiver.receive_weights(on_bucket_received=on_bucket_received)
            if callback_error is not None:
                raise callback_error
            finalize_layerwise_reload(model, vllm_config.model_config)
        self._layerwise_reload_failed = False

        # Retained bucket clones are dead now; return them before KV cache wakes.
        gc.collect()
        get_torch_device().empty_cache()
        return loaded


__all__ = [
    "LayerwiseReloadWorkerExtension",
    "load_checkpoint_bucket",
    "require_layerwise_reload_support",
]
