"""EP routed experts over normal DeepEP, as in the DeepSeek-V4 aligned actor.

Each source token is sent once to every EP rank that owns one of its routes,
as one row per (token, rank) pair. DeepEP therefore never sums: every combine
row has a single contributor and the transport is lossless in both
directions. The expert rank computes its local routes and its BF16 partial
as a serving rank does (the CuTe-DSL launch over its 32 experts and that
launch's own ``moe_unpermute``), the partial returns to the
source rank, and the source rank reduces the partials in the serving order
(``reduce_ep4_parts``). The backward sends the output gradient over
the same rows and runs ``routed_vjp`` on the local experts; the input
gradient of every route returns unsummed, and the source rank adds a token's
routes in slot order with BF16 rounding after each add (DS4's deterministic
scatter backward), as the single-rank VJP does.
"""

import contextlib
import socket

import torch
import torch.distributed as dist

from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import (
    routed_vjp,
    sum_route_grads,
)


_buffer = None
# Normal-mode intranode combine needs the row width (16-byte units) to be a
# multiple of the 32 warp lanes: 256 BF16. Lightning's 2688 is not; combine
# rows are zero-padded and sliced back, which is exact.
_COMBINE_ALIGN = 256
# The DeepEP build (DEEPEP_NUM_MAX_NVL_PEERS) and the buffer below are NVLink
# only: no RDMA buffer is allocated.
_MAX_NVL_PEERS = 4


def reduce_ep4_parts(parts, ids):
    """Combine the four BF16 rank partials as the FlashInfer one-sided combine."""
    if (
        len(parts) != 4
        or ids.ndim != 2
        or ids.shape[1] != 6
        or any(
            p.dtype != torch.bfloat16 or p.ndim != 2 or p.shape != parts[0].shape
            for p in parts
        )
        or parts[0].shape[0] != ids.shape[0]
    ):
        raise ValueError("Require four BF16 rank partials and six routes per row")
    owners = ids // 32
    stacked = torch.stack(parts)
    rows = torch.arange(ids.shape[0], device=ids.device)
    slots = []
    for slot in range(6):
        rank = owners[:, slot]
        duplicate = (owners[:, :slot] == rank[:, None]).any(dim=1)
        value = stacked[rank, rows]
        slots.append(torch.where(duplicate[:, None], 0, value).float())
    # FlashInfer 0.7.0 one-sided combine (csrc/nv_internal/tensorrt_llm/kernels/
    # communicationKernels/moeAlltoAllKernels.cu): dispatch sends each token once
    # per target rank, from the first top-k slot naming it (:495); combine loads
    # that rank's partial for the first slot and FP32 zero for duplicates
    # (:971); TOP_K=6 sums ((s0+s1)+(s2+s3))+(s4+s5) in FP32 (:1104-1115).
    total = ((slots[0] + slots[1]) + (slots[2] + slots[3])) + (slots[4] + slots[5])
    return total.to(torch.bfloat16)


@contextlib.contextmanager
def _deepep_memory():
    """Turn off PyTorch's deterministic fill of new allocations for DeepEP calls.

    The fill writes NaN into buffers DeepEP's kernels are concurrently filling;
    restored on exit, so other code keeps the caller's setting.
    """
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = fill


def _check_intranode(group):
    size = dist.get_world_size(group)
    hosts = [None] * size
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if size > _MAX_NVL_PEERS or len(set(hosts)) != 1:
        raise RuntimeError(
            f"DeepEP EP group must be one NVLink node of at most {_MAX_NVL_PEERS} "
            f"ranks (no RDMA buffer); got {size} ranks on {sorted(set(hosts))}"
        )


def _deepep_buffer(group, hidden_bytes):
    """Process-wide normal-mode DeepEP buffer (DS4 ``_get_deepep_buffer``)."""
    import deep_ep

    global _buffer
    size = dist.get_world_size(group)
    nvl_bytes = max(
        config.get_nvl_buffer_size_hint(hidden_bytes, size)
        for config in (
            deep_ep.Buffer.get_dispatch_config(size),
            deep_ep.Buffer.get_combine_config(size),
        )
    )
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < nvl_bytes:
        if _buffer is None or _buffer.group != group:
            _check_intranode(group)
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


def _forward(experts, group, x, ids, weights, *, save=True):
    plan = _Plan(ids, dist.get_world_size(group), experts.num_experts)
    # The backward returns one row of topk per-route input gradients.
    width = ids.shape[1] * x.shape[1]
    width += -width % _COMBINE_ALIGN
    with _deepep_memory():
        buffer = _deepep_buffer(group, width * x.element_size())
        recv_x, recv_idx, recv_w, handle = _dispatch(
            buffer, plan, x, weights, experts.global_num_experts
        )
        # The serving rank's launch over its own experts; its top-k combine
        # (moe_unpermute) forms the partial.
        result = experts.rank_partial(
            recv_x, recv_w, _expert_ids(experts, recv_idx), save=save
        )
        partial, fc1, visible, activated = result if save else (result, None, None, None)
        returned, _ = _combine(buffer, partial, handle)
    parts = x.new_zeros(plan.ep_size, x.shape[0], x.shape[1])
    for r, rows, token in plan.per_rank():
        parts[r].index_copy_(0, token, returned.index_select(0, rows))
    out = reduce_ep4_parts(list(parts.unbind(0)), ids)
    if not save:
        return out, None
    # The visible FC1 and expert outputs cover this rank's routes only; a
    # received row carries all topk slots, most of them other ranks'.
    owned = (recv_idx.reshape(-1) >= 0).nonzero().squeeze(1)
    state = (plan, buffer, handle, recv_x, recv_idx, recv_w, owned, fc1, visible, activated)
    return out, state


# One Function, not DS4's scatter / grouped / gather Functions: a received row
# is a (token, rank) pair holding several of the token's routes, and the source
# rank must add every route's input gradient in slot order with BF16 rounding
# (as the single-rank VJP). An autograd boundary at the dispatch would sum a
# row's routes on the expert rank first.
class EPRoutedExpertsVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down, routes, ids, owner):
        out, state = _forward(owner._experts, owner.ep_group, x, ids, routes)
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
        # Kept (not cleared) so a retained graph can run backward again.
        plan, buffer, handle, recv_x, recv_idx, recv_w, owned, fc1, visible, activated = (
            ctx.state
        )
        m, k = ctx.x_shape
        topk = plan.ids.shape[1]
        slots = recv_x.shape[0] * topk
        fc1 = fc1.new_zeros(slots, fc1.shape[1]).index_copy_(0, owned, fc1)
        visible = visible.new_zeros(slots, k).index_copy_(0, owned, visible)
        activated = activated.new_zeros(slots, activated.shape[1]).index_copy_(
            0, owned, activated
        )
        rows_dy = dy.to(torch.bfloat16).index_select(0, plan.token).contiguous()
        with _deepep_memory():
            recv_dy, *_ = buffer.dispatch(rows_dy, handle=handle)
            dx_routes, d_up, d_down, dw_rows = routed_vjp(
                recv_x, fc1, visible, up, down, recv_w, recv_idx, recv_dy, activated,
                per_route=True,
            )
            returned_dx, returned_dw = _combine(
                buffer, dx_routes.view(-1, topk * k), handle, topk_weights=dw_rows.float()
            )
        # One owner per (token, slot): place each route's gradient, then add a
        # token's routes in slot order as DS4 does.
        routes_dx = dy.new_zeros((m, topk, k), dtype=torch.bfloat16)
        d_routes = dy.new_zeros((m, topk), dtype=torch.float32)
        returned_dx = returned_dx.view(-1, topk, k)
        for _, rows, token in plan.per_rank():
            owned = plan.owned.index_select(0, rows)
            current = routes_dx.index_select(0, token)
            routes_dx.index_copy_(
                0, token,
                torch.where(owned[..., None], returned_dx.index_select(0, rows), current),
            )
            weights = torch.where(owned, returned_dw.index_select(0, rows), 0.0)
            d_routes.index_add_(0, token, weights)
        return sum_route_grads(routes_dx), d_up, d_down, d_routes, None, None


def ep_routed_experts(owner, x, ids, routes, *, grad):
    if grad:
        return EPRoutedExpertsVJP.apply(
            x, owner.weights.up_proj, owner.weights.down_proj, routes, ids, owner
        )
    with torch.no_grad():
        return _forward(owner._experts, owner.ep_group, x, ids, routes, save=False)[0]
