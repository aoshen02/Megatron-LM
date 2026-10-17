# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NeMo-RL vLLM worker extension for refitting ModelOpt mixed-precision rollouts.

The mlite policy exports deployment tensors in checkpoint format (NVFP4/FP8
weights, FP32 scales, static k/v scales). Only vLLM's full-model layerwise
reload can repack them into the kernel layout. Select it with
``generation.vllm_cfg.internal_worker_extension_cls``.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

import torch
from nemo_rl.models.generation.vllm.vllm_backend import (
    VllmInternalWorkerExtension,
    _refresh_hpc_modules_after_layerwise_reload,
)


def _graph_bound_kernel_state(model: torch.nn.Module) -> dict[int, tuple]:
    """Objects whose CUDA tensors captured graphs may reference, per module.

    Records each module's quant method, its MoE kernel and the unregistered CUDA
    tensors of its linear kernel (e.g. humming ``locks``).
    """
    state = {}
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        kernel = getattr(quant_method, "kernel", None)
        scratch = {
            name: value
            for name, value in (vars(kernel).items() if kernel is not None else ())
            if isinstance(value, torch.Tensor) and value.is_cuda
        }
        state[id(module)] = (
            quant_method,
            getattr(quant_method, "moe_kernel", None),
            kernel,
            scratch,
        )
    return state


def _restore_graph_bound_kernel_state(model: torch.nn.Module, state: dict[int, tuple]) -> list[str]:
    """Undo object and scratch re-creation done by ``process_weights_after_loading``.

    Captured CUDA graphs keep the addresses of tensors owned by the kernel
    objects built at load time (lazily created MoE index buffers, linear-kernel
    workspaces). A layerwise refit re-runs weight processing, which may build
    new kernel objects or workspaces and free the old ones; replaying the graphs
    then reads freed memory. The kernels read weights and scales through the
    layer, whose storage the reload restores in place, so the original objects
    stay valid.
    """
    restored = []
    for name, module in model.named_modules():
        saved = state.get(id(module))
        if saved is None:
            continue
        quant_method, moe_kernel, kernel, scratch = saved
        if module.quant_method is not quant_method:
            module.quant_method = quant_method
            restored.append(f"{name}.quant_method")
            continue
        if moe_kernel is not None and quant_method.moe_kernel is not moe_kernel:
            quant_method.moe_kernel = moe_kernel
            restored.append(f"{name}.quant_method.moe_kernel")
        if kernel is not None:
            for attr, value in list(vars(kernel).items()):
                old = scratch.get(attr)
                if (
                    old is None
                    or old is value
                    or not isinstance(value, torch.Tensor)
                    or old.shape != value.shape
                    or old.dtype != value.dtype
                ):
                    continue
                old.copy_(value)
                setattr(kernel, attr, old)
                restored.append(f"{name}.quant_method.kernel.{attr}")
    return restored


def _incomplete_layerwise_layers(
    model: torch.nn.Module, skip_extra: frozenset[str] | set[str] = frozenset()
) -> list[str]:
    """Layers whose buffered layerwise reload is missing checkpoint tensors."""
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
    from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS
    from vllm.model_executor.model_loader.reload.utils import (
        get_layer_params_buffers,
        get_tensor_load_numel,
    )

    # Attention's q/prob scale placeholders are not checkpoint tensors.
    skip = SKIP_LOAD_TENSORS | {"q_scale", "prob_scale"} | set(skip_extra)
    incomplete = []
    for prefix, layer in model.named_modules():
        info = LAYERWISE_INFO.get(layer)
        if info is None or not info.can_load():
            continue
        expected = sum(
            get_tensor_load_numel(tensor)
            for name, tensor in get_layer_params_buffers(layer)[0].items()
            if name not in skip
        )
        if info.load_numel != expected:
            incomplete.append(prefix)
    return incomplete


class MLiteVllmWorkerExtension(VllmInternalWorkerExtension):
    # Drafter layers reloaded layerwise during the current refit.
    _mtp_reload: Optional[dict[str, Any]] = None

    def _uses_modelopt_mixed_precision_refit(self) -> bool:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig

        return isinstance(
            getattr(self.model_runner.vllm_config, "quant_config", None),
            ModelOptMixedPrecisionConfig,
        )

    def _uses_modelopt_refit(self, transport: Any) -> bool:
        return transport in ("ipc", "collective") and self._uses_modelopt_mixed_precision_refit()

    def _uses_native_layerwise_refit(self, transport: Any) -> bool:
        return self._uses_modelopt_refit(transport) or super()._uses_native_layerwise_refit(
            transport
        )

    def _load_hf_weights(self, policy_weights: list[tuple[str, torch.Tensor]]) -> None:
        if self._mtp_reload is not None:
            # The drafter owns mtp.*; the target model has no such layers.
            policy_weights = [w for w in policy_weights if not w[0].startswith("mtp.")]
        super()._load_hf_weights(policy_weights)

    def _maybe_refit_mtp_drafter(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        reload = self._mtp_reload
        if reload is None:
            return super()._maybe_refit_mtp_drafter(weights)
        # Only the drafter's own layers: its embedding and LM head are the
        # target's and reload with it. Loader arguments stay buffered until a
        # layer completes, past the transport buffer's reuse.
        mtp = [(n, t.clone()) for n, t in weights if n.startswith("mtp.")]
        if not mtp:
            return
        if not reload["started"]:
            from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

            with torch.device(self.device):
                initialize_layerwise_reload(reload["layers"])
            reload["started"] = True
        reload["loaded"].update(self._get_drafter_model().load_weights(weights=mtp))

    def _finalize_mtp_drafter_reload(self) -> None:
        reload = self._mtp_reload
        if reload is None or not reload["started"]:
            return
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        layers = reload["layers"]
        # Neither the checkpoint nor the trainer carries KV-cache scales for the
        # drafter's attention; it keeps its own.
        if incomplete := _incomplete_layerwise_layers(layers, skip_extra={"k_scale", "v_scale"}):
            raise RuntimeError(f"Refit left drafter layers incomplete: {incomplete}")
        finalize_layerwise_reload(
            layers, self.model_runner.vllm_config.speculative_config.draft_model_config
        )
        restored = _restore_graph_bound_kernel_state(layers, reload["kernel_state"])
        print(
            f"[nemorl_mlite] drafter refit: loaded {len(reload['loaded'])} mtp.* tensors, "
            f"restored {len(restored)} graph-bound kernel objects",
            flush=True,
        )

    @contextmanager
    def _weight_update_lifecycle(self, transport: Any) -> Iterator[Any]:
        if not self._uses_modelopt_refit(transport):
            with super()._weight_update_lifecycle(transport) as finalize:
                yield finalize
            return

        previous_failure = self._nrl_layerwise_reload_failure
        if previous_failure is not None:
            raise RuntimeError(
                "The vLLM worker is unusable after a failed native layerwise refit"
            ) from previous_failure
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        model = self.model_runner.model
        kernel_state = _graph_bound_kernel_state(model)
        draft_model = self._get_drafter_model() if self._mtp_drafter_refit_enabled() else None
        draft_layers = getattr(getattr(draft_model, "model", None), "layers", None)

        def finalize() -> None:
            with torch.device(self.device):
                # Finalizing would process a partly loaded layer from
                # uninitialized storage, or keep serving old weights.
                if incomplete := _incomplete_layerwise_layers(model):
                    raise RuntimeError(f"Refit left layers incomplete: {incomplete[:10]}")
                finalize_layerwise_reload(model, self.model_config)
                restored = _restore_graph_bound_kernel_state(model, kernel_state)
                kinds = sorted({re.sub(r"\d+", "N", r) for r in restored})
                print(
                    f"[nemorl_mlite] refit restored {len(restored)} graph-bound kernel "
                    f"objects: {kinds}",
                    flush=True,
                )
                _refresh_hpc_modules_after_layerwise_reload(model)
                self._finalize_mtp_drafter_reload()
            torch.cuda.synchronize()

        try:
            if draft_layers is not None:
                self._mtp_reload = {
                    "layers": draft_layers,
                    "kernel_state": _graph_bound_kernel_state(draft_layers),
                    "started": False,
                    "loaded": set(),
                }
            with set_current_vllm_config(self.model_runner.vllm_config):
                with torch.device(self.device):
                    initialize_layerwise_reload(model)
                self._nrl_layerwise_reload_active = True
                yield finalize
        except Exception as error:
            self._nrl_layerwise_reload_failure = error
            raise
        finally:
            self._nrl_layerwise_reload_active = False
            self._mtp_reload = None
