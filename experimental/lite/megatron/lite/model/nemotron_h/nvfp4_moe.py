"""Routed W4A16 forward (FlashInfer CuTe-DSL, the rollout's ``moe_backend``)
with the BF16 master-weight routed VJP."""

import torch

from .nvfp4_experts import Nvfp4ExpertWeights
from .nvfp4_moe_vjp import RoutedExpertsVJP


class Nvfp4RoutedDeployment(torch.nn.Module):
    """Own a frozen W4A16 deployment alongside BF16 expert masters.

    Inputs are fixed expert IDs and continuous routing weights; this adapter does
    not route, normalize scores, apply routed_scaling_factor, or add shared output.
    No Graph/compile/concurrent-host dispatch support is claimed. Training runs
    the rollout's EP4 combine so the forward can keep its visible FC1 output
    for ``nvfp4_moe_vjp.routed_vjp``.
    """

    def __init__(self, weights, model_config, *, ep_group=None):
        super().__init__()
        if not isinstance(weights, Nvfp4ExpertWeights):
            raise TypeError("Expected Nvfp4ExpertWeights")
        if (
            model_config.n_routed_experts != weights.num_experts
            or weights._geometry["up_proj"]
            != (model_config.moe_intermediate_size, model_config.hidden_size)
            or model_config.mlp_hidden_act != "relu2"
            or model_config.mlp_bias
        ):
            raise ValueError("Expected matching bias-free ReLU2 expert geometry")
        if not 1 <= model_config.num_experts_per_tok <= weights.num_experts:
            raise ValueError("Invalid top-k expert count")
        self.ep_group = ep_group
        if (weights.num_local != weights.num_experts) != (ep_group is not None):
            raise ValueError("EP experts need the EP group, and only they")
        if (
            weights.num_experts, model_config.hidden_size,
            model_config.moe_intermediate_size, model_config.num_experts_per_tok,
        ) != (128, 2688, 1856, 6):
            raise ValueError("EP4 reduction requires Lightning expert geometry")
        if weights.up_proj.device.type != "cuda":
            raise ValueError("Routed deployment requires CUDA checkpoint storage")
        self.weights = weights
        self.config = model_config
        self._ready = False
        self._install()

    def _apply(self, fn, recurse=True):
        raise RuntimeError(
            "Construct deployment on its final device; do not convert its recipe"
        )

    def _validate_checkpoint(self):
        self.weights._validate_storage()
        if (
            self.weights._dirty
            or self.weights._versions() != self.weights._synced_versions
        ):
            raise RuntimeError(
                "Refresh deployment after changing expert masters/storage"
            )

    @torch.no_grad()
    def _install(self):
        from .kernels import CuteDslRoutedExperts

        self._ready = False
        self._validate_checkpoint()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Deployment construction cannot run during Graph capture"
            )
        w = self.weights
        stacks = tuple(
            tuple(getattr(w, f"_{projection}_{s}") for s in ("packed", "scale", "global"))
            for projection in ("up_proj", "down_proj")
        )
        self._experts = CuteDslRoutedExperts(
            *stacks, num_experts=self.config.n_routed_experts, offset=w.offset
        )
        self._deployed_versions = w._versions()
        self._ready = True

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        """Call after every optimizer/runtime update, including .data writes."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Deployment refresh cannot run during Graph capture")
        self._ready = False
        self.weights.refresh_quantized(recompute_scales=recompute_scales)
        self._install()

    def forward(self, x, ids, routing_weights):
        active_grad = torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (x, routing_weights, self.weights.up_proj, self.weights.down_proj)
        )
        if self.ep_group is not None:
            from .ep import ep_routed_experts

            self._check_inputs(x, ids, routing_weights)
            return ep_routed_experts(self, x, ids, routing_weights, grad=active_grad)
        if not active_grad:
            return self._visible(x, ids, routing_weights)
        return RoutedExpertsVJP.apply(
            x, self.weights.up_proj, self.weights.down_proj, routing_weights, ids, self
        )

    def _visible(self, x, ids, routing_weights, *, return_fc1=False):
        self._check_inputs(x, ids, routing_weights)
        if x.shape[0] == 0:
            if return_fc1:
                raise ValueError("Routed training requires at least one token")
            return torch.empty_like(x)
        from .nvfp4_ep4 import ep4_routed_experts

        return ep4_routed_experts(
            self._experts, x, routing_weights, ids, return_fc1=return_fc1
        )

    def _check_inputs(self, x, ids, routing_weights):
        self._validate_checkpoint()
        if not self._ready or self.weights._versions() != self._deployed_versions:
            raise RuntimeError("Refresh deployment before routed forward")
        c, device = self.config, self.weights.up_proj.device
        if (
            x.ndim != 2
            or x.shape[1] != c.hidden_size
            or x.dtype != torch.bfloat16
            or ids.shape != (x.shape[0], c.num_experts_per_tok)
            or routing_weights.shape != ids.shape
            or ids.dtype != torch.int32
            or routing_weights.dtype != torch.float32
            or any(
                t.device != device or not t.is_contiguous()
                for t in (x, ids, routing_weights)
            )
        ):
            raise ValueError(
                "Expected contiguous BF16 X, int32 IDs and FP32 route weights"
            )
        if (
            not torch.isfinite(x).all()
            or not torch.isfinite(routing_weights).all()
            or (ids < 0).any()
            or (ids >= c.n_routed_experts).any()
        ):
            raise ValueError("Nonfinite input or out-of-range expert IDs")
