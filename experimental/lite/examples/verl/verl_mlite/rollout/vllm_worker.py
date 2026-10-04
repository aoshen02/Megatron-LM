# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vLLM worker integration for MLite weight refits."""

from __future__ import annotations

import gc

import torch
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    BucketedWeightReceiver,
)
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


def _needs_native_reload(vllm_config) -> bool:
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    quant_config = getattr(vllm_config, "quant_config", None)
    return isinstance(quant_config, ModelOptMixedPrecisionConfig)


class MLiteVLLMColocateWorkerExtension(vLLMColocateWorkerExtension):
    """Refit ModelOpt checkpoints through vLLM's native layerwise lifecycle.

    As the DeepSeek-V4 refit: restore the checkpoint layout, load every
    bucket, process each completed layer once and copy the result into the
    original kernel storage, which captured CUDA graphs keep reading. The
    actor sends its deployment tensors in checkpoint format.

    DeepSeek-V4 installs this lifecycle under verl's quantized-refit hooks,
    which verl only enters for FP8 configs; a ModelOpt mixed-precision
    checkpoint is driven here. Every other config keeps verl's path.
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
        if self._use_mtp_drafter_weight_sync():
            raise NotImplementedError("MLite layerwise refit does not sync MTP drafters")

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_processing,
            initialize_layerwise_reload,
        )

        if self.device is None:
            self.device = torch.device(f"{get_device_name()}:{self.local_rank}")
        model = self.model_runner.model
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(), device=self.device, use_shm=use_shm
        )

        def load(weights, is_last):
            # vLLM buffers loader arguments until a layer is complete, while
            # the receiver reuses its bucket once this callback returns.
            model.load_weights([(name, tensor.clone()) for name, tensor in weights])

        with set_current_vllm_config(vllm_config):
            initialize_layerwise_reload(model)
            receiver.receive_weights(on_bucket_received=load)
            finalize_layerwise_processing(model, vllm_config.model_config)
        # Release the staged checkpoint-format tensors before the KV cache wakes.
        gc.collect()
        get_torch_device().empty_cache()


__all__ = ["MLiteVLLMColocateWorkerExtension"]
