"""EP routed experts over normal DeepEP, as in the DeepSeek-V4 aligned actor.

Each source token is sent once to every EP rank that owns one of its routes,
as one row per (token, rank) pair. DeepEP therefore never sums: every combine
row has a single contributor and the transport is lossless in both
directions. The expert rank computes its local Humming routes and its BF16
partial (``moe_fused_mul_sum`` over its slots), the partial returns to the
source rank, and the source rank reduces the partials in the serving order
(``nvfp4_ep4.reduce_ep4_parts``). The backward sends the output gradient over
the same rows, runs ``routed_vjp`` on the local experts, and sums the returned
input/route gradients on the source rank in rank order.
"""

import torch
import torch.distributed as dist

from .nvfp4_ep4 import reduce_ep4_parts
from .nvfp4_moe_vjp import routed_vjp

_buffer = None
# Normal-mode intranode combine needs the row width (16-byte units) to be a
# multiple of the 32 warp lanes: 256 BF16. Lightning's 2688 is not; combine
# rows are zero-padded and sliced back, which is exact.
_COMBINE_ALIGN = 256


def _deepep_buffer(group, hidden_bytes):
    """Process-wide normal-mode DeepEP buffer (DS4 ``_get_deepep_buffer``)."""
    import deep_ep

    global _buffer
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        # Deterministic debug fill races DeepEP's own writes.
        torch.utils.deterministic.fill_uninitialized_memory = False
    size = dist.get_world_size(group)
    nvl_bytes = max(
        config.get_nvl_buffer_size_hint(hidden_bytes, size)
        for config in (
            deep_ep.Buffer.get_dispatch_config(size),
            deep_ep.Buffer.get_combine_config(size),
        )
    )
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < nvl_bytes:
        deep_ep.Buffer.set_num_sms(20)
        _buffer = deep_ep.Buffer(
            group=group, num_nvl_bytes=nvl_bytes, num_rdma_bytes=0,
            explicitly_destroy=True,
        )
    return _buffer


class _Plan:
    """Source-side (token, rank) rows of one routed call."""

    def __init__(self, ids, ep_size, num_local):
        owners = ids.long() // num_local
        present = torch.stack([(owners == r).any(1) for r in range(ep_size)], 1)
        self.token, self.rank = present.nonzero(as_tuple=True)
        self.owned = owners.index_select(0, self.token) == self.rank[:, None]
        self.ids = torch.where(self.owned, ids.index_select(0, self.token).long(), -1)
        self.tokens, self.ep_size = ids.shape[0], ep_size

    def per_rank(self):
        for r in range(self.ep_size):
            rows = (self.rank == r).nonzero(as_tuple=True)[0]
            yield r, rows, self.token.index_select(0, rows)


def _dispatch(buffer, plan, x, weights, num_experts):
    rows_x = x.index_select(0, plan.token)
    rows_w = torch.where(plan.owned, weights.index_select(0, plan.token), 0.0)
    layout = buffer.get_dispatch_layout(plan.ids, num_experts=num_experts)
    recv_x, recv_idx, recv_w, _, handle, _ = buffer.dispatch(
        rows_x,
        topk_idx=plan.ids,
        topk_weights=rows_w.float(),
        num_tokens_per_rank=layout[0],
        num_tokens_per_rdma_rank=layout[1],
        is_token_in_rank=layout[3],
        num_tokens_per_expert=layout[2],
    )
    return recv_x, recv_idx, recv_w, handle


def _combine(buffer, x, handle, topk_weights=None):
    width = x.shape[1]
    pad = -width % _COMBINE_ALIGN
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    out, weights, _ = buffer.combine(x.contiguous(), handle, topk_weights=topk_weights)
    return out[:, :width], weights


def _expert_ids(experts, recv_idx):
    """Global ids for the local kernel; other ranks' routes get a non-local id."""
    absent = (experts.offset + experts.num_experts) % experts.global_num_experts
    return torch.where(recv_idx >= 0, recv_idx + experts.offset, absent).to(torch.int32)


def _forward(experts, group, x, ids, weights, recipe):
    plan = _Plan(ids, dist.get_world_size(group), experts.num_experts)
    width = x.shape[1] + (-x.shape[1] % _COMBINE_ALIGN)
    buffer = _deepep_buffer(group, width * x.element_size())
    recv_x, recv_idx, recv_w, handle = _dispatch(
        buffer, plan, x, weights, experts.global_num_experts
    )
    tokens = torch.tensor([x.shape[0]], device=x.device)
    dist.all_reduce(tokens, group=group)
    global_ids = _expert_ids(experts, recv_idx)
    fc1, down = experts.routes(recv_x, global_ids, global_tokens=int(tokens.item()))
    partial = experts.rank_partial(down, recv_w, global_ids, experts.expert_map)
    returned, _ = _combine(buffer, partial, handle)
    parts = x.new_zeros(plan.ep_size, x.shape[0], x.shape[1])
    for r, rows, tokens in plan.per_rank():
        parts[r].index_copy_(0, tokens, returned.index_select(0, rows))
    out = reduce_ep4_parts(list(parts.unbind(0)), ids, recipe)
    return out, (plan, buffer, handle, recv_x, recv_idx, recv_w, fc1, down)


class EPRoutedExpertsVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down, routes, ids, owner):
        out, state = _forward(
            owner._experts, owner.ep_group, x, ids, routes, owner.routed_forward_reduction
        )
        ctx.owner, ctx.versions = owner, owner.weights._versions()
        ctx.state, ctx.x_shape = state, x.shape
        ctx.save_for_backward(up, down)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        if ctx.owner.weights._versions() != ctx.versions:
            raise RuntimeError("Expert masters changed before backward")
        up, down = ctx.saved_tensors
        plan, buffer, handle, recv_x, recv_idx, recv_w, fc1, visible = ctx.state
        ctx.state = None
        rows_dy = dy.to(torch.bfloat16).index_select(0, plan.token).contiguous()
        recv_dy, *_ = buffer.dispatch(rows_dy, handle=handle)
        dx_rows, d_up, d_down, dw_rows = routed_vjp(
            recv_x, fc1, visible.view(-1, visible.shape[-1]), up, down, recv_w,
            recv_idx, recv_dy,
        )
        returned_dx, returned_dw = _combine(
            buffer, dx_rows, handle, topk_weights=dw_rows.float()
        )
        dx = dy.new_zeros(ctx.x_shape, dtype=torch.float32)
        d_routes = dy.new_zeros((plan.tokens, plan.ids.shape[1]), dtype=torch.float32)
        returned_dw = torch.where(plan.owned, returned_dw, 0.0)
        for _, rows, tokens in plan.per_rank():
            dx.index_add_(0, tokens, returned_dx.index_select(0, rows).float())
            d_routes.index_add_(0, tokens, returned_dw.index_select(0, rows))
        return dx.to(torch.bfloat16), d_up, d_down, d_routes, None, None


def ep_routed_experts(owner, x, ids, routes, *, grad):
    if grad:
        return EPRoutedExpertsVJP.apply(
            x, owner.weights.up_proj, owner.weights.down_proj, routes, ids, owner
        )
    with torch.no_grad():
        return _forward(
            owner._experts, owner.ep_group, x, ids, routes, owner.routed_forward_reduction
        )[0]
