"""Nemotron-H MoE: grouped sigmoid router, shared expert and routed deployment."""

import torch
from torch import nn

from megatron.lite.model.nemotron_h.vllm.primitive.dense import (
    projection,
    visible_linear,
)
from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import (
    Nvfp4ExpertWeights,
    Nvfp4RoutedDeployment,
)


class _FixedRouteVJP(torch.autograd.Function):
    """Visible grouped top-k; FP32 sigmoid-gather-renorm replay on fixed ids."""

    @staticmethod
    def forward(ctx, logits, visible, renormalize):
        weights, ids = visible(logits)
        ctx.save_for_backward(logits, ids.clone())
        ctx.renormalize = renormalize
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    def backward(ctx, grad_weights, _grad_ids):
        logits, ids = ctx.saved_tensors
        with torch.enable_grad():
            replay = logits.detach().float().requires_grad_(True)
            selected = replay.sigmoid().gather(-1, ids.long())
            if ctx.renormalize:
                selected = selected / selected.sum(-1, keepdim=True)
            (grad_logits,) = torch.autograd.grad(selected, replay, grad_weights.float())
        return grad_logits.to(logits.dtype), None, None


class _CombineVJP(torch.autograd.Function):
    """Visible compiled ``shared + routed * scale``; closed-form VJP."""

    @staticmethod
    def forward(ctx, visible, shared, routed, scale):
        ctx.scale = scale
        return visible(shared, routed)

    @staticmethod
    def backward(ctx, grad):
        return None, grad, grad * ctx.scale, None


class Router(nn.Module):
    def __init__(self, config, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, device=device, dtype=dtype
            )
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.n_routed_experts, device=device, dtype=torch.float32),
        )

    def forward(self, x):
        from vllm.model_executor.determinism import batch_invariant
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            grouped_topk,
        )

        # torch.mm(out_dtype=fp32) is M-invariant only once the process has
        # run init_batch_invariance (build_model does, before any forward).
        if not batch_invariant._batch_invariant_MODE:
            raise RuntimeError("Router GEMM before init_batch_invariance()")
        config = self.config
        logits = visible_linear(
            lambda x: torch.mm(x, self.weight.T, out_dtype=torch.float32),
            x,
            self.weight,
        )

        def visible(logits):
            return grouped_topk(
                x,
                logits,
                config.num_experts_per_tok,
                config.norm_topk_prob,
                config.n_group,
                config.topk_group,
                "sigmoid",
                1.0,
                self.e_score_correction_bias.float(),
            )

        if not torch.is_grad_enabled():
            weights, ids = visible(logits)
        else:
            weights, ids = _FixedRouteVJP.apply(logits, visible, config.norm_topk_prob)
        return ids, weights


class SharedExperts(nn.Module):
    def __init__(
        self,
        config,
        *,
        device=None,
        dtype=torch.bfloat16,
        projection_factory,
        hf_prefix,
    ):
        super().__init__()
        self.up_proj = projection_factory(
            f"{hf_prefix}.up_proj",
            config.hidden_size,
            config.moe_shared_expert_intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = projection_factory(
            f"{hf_prefix}.down_proj",
            config.moe_shared_expert_intermediate_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = torch.nn.functional.relu(projection(x, self.up_proj)).square()
        return projection(x, self.down_proj)


class MoE(nn.Module):
    """Routing, the routed experts and the shared expert."""

    def __init__(
        self,
        config,
        ps,
        *,
        device=None,
        dtype=torch.bfloat16,
        projection_factory,
        hf_prefix,
    ):
        super().__init__()
        if config.n_shared_experts != 1:
            raise ValueError("Nemotron MoE requires the single shared expert contract")
        self.gate = Router(config, device=device, dtype=dtype)
        weights = Nvfp4ExpertWeights(
            projection_factory.root,
            f"{hf_prefix}.experts",
            projection_factory.recipes,
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            tp_size=ps.tp_size,
            ep_size=ps.ep_size,
            ep_rank=ps.ep_rank,
            device=device,
        )
        self.experts = Nvfp4RoutedDeployment(
            weights, config, ep_group=ps.ep_group if ps.ep_size > 1 else None
        )
        self.shared_experts = SharedExperts(
            config,
            device=device,
            dtype=dtype,
            projection_factory=projection_factory,
            hf_prefix=f"{hf_prefix}.shared_experts",
        )
        self.routed_scaling_factor = scale = config.routed_scaling_factor

        def combine(shared, routed):
            return shared + routed * scale

        self._visible_combine = torch.compile(combine, fullgraph=True, dynamic=True)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        ids, weights = self.gate(x)
        routed = self.experts(x, ids, weights)
        shared = self.shared_experts(x)
        if not torch.is_grad_enabled():
            return self._visible_combine(shared, routed).view(shape)
        return _CombineVJP.apply(
            self._visible_combine, shared, routed, self.routed_scaling_factor
        ).view(shape)
