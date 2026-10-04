"""Routed-expert VJP on the BF16 masters (TE ``high_precision`` semantics).

Mirrors the DeepSeek-V4 aligned actor's grouped backward
(NVIDIA/Megatron-LM#7050): the visible FC1 and expert outputs are saved in
forward, the route gradient comes from the visible expert output and the
dgrad/wgrad GEMMs run as Transformer Engine grouped BF16 GEMMs.
"""

import torch


def _te_grouped_gemm(lhs, rhs, out, *, layout, m_splits, single_output=False):
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    outputs = [out] if isinstance(out, torch.Tensor) else list(out)
    general_grouped_gemm(
        list(lhs),
        list(rhs),
        outputs,
        [None] * len(lhs),
        torch.bfloat16,
        single_output=single_output,
        layout=layout,
        m_splits=list(m_splits),
        grad=True,
        use_split_accumulator=True,
    )


def routed_vjp(x, fc1, visible, up, down, routes, ids, dy, activated=None, *, per_route=False):
    """ReLU2 routed experts: ``y = sum_s routes[:, s] * down(relu(up(x))**2)``.

    Per-input contract (as DeepSeek-V4's grouped MoE): the route-weight
    gradient is ``<dy, visible expert output>``, exact because the weights
    only scale the visible outputs; the input and expert-weight gradients are
    Transformer Engine ``high_precision`` BF16 GEMMs on the masters from the
    visible FC1 output.

    Args:
        x: BF16 tokens, ``[M, K]``.
        fc1: Visible FC1 output per route (token-major, slot-minor), ``[M*topk, I]``.
        visible: Visible expert output per route (same order), ``[M*topk, K]``.
        up, down: BF16 masters, ``[E, I, K]`` and ``[E, K, I]``.
        routes: FP32 routing weights, ``[M, topk]``.
        ids: Expert ids, ``[M, topk]``; ``-1`` marks a route held elsewhere
            (zero gradients for it).
        dy: BF16 output gradient, ``[M, K]``.
        activated: Visible activation GEMM2 consumed (same order),
            ``[M*topk, I]``, for the down-weight gradient; None recomputes it
            as ``bf16(relu(fc1)**2)``.
        per_route: Return the input gradient of every route, ``[M, topk, K]``
            (zero for absent routes), instead of their sum.

    Returns:
        ``(dx, d_up, d_down, d_routes)``.
    """
    m, k = x.shape
    topk, experts = ids.shape[1], up.shape[0]
    if m == 0:
        dx = x.new_zeros(0, topk, k) if per_route else torch.zeros_like(x)
        return dx, torch.zeros_like(up), torch.zeros_like(down), routes.new_zeros(0, topk)
    flat = ids.reshape(-1).long()
    held = int((flat >= 0).sum())
    # Absent routes sort first (as -1) and are dropped.
    order = torch.argsort(flat, stable=True)[flat.numel() - held :]
    counts = torch.bincount(flat[flat >= 0], minlength=experts).tolist()
    token = order // topk
    u = fc1.index_select(0, order)
    if activated is None:
        h = u.float().relu().square().to(torch.bfloat16)
    else:
        h = activated.index_select(0, order)
    x_rows = x.index_select(0, token)
    dy_rows = dy.index_select(0, token)
    weight = routes.reshape(-1).index_select(0, order)
    dv = (dy_rows.float() * weight[:, None]).to(torch.bfloat16)

    def split(rows):
        return torch.split(rows, counts)

    d_weight = (dy_rows.float() * visible.index_select(0, order).float()).sum(-1)
    dh = torch.empty_like(h)
    _te_grouped_gemm(down.unbind(0), split(dv), dh, layout="NN", m_splits=counts,
                     single_output=True)
    d_down = torch.zeros_like(down)
    _te_grouped_gemm(split(h), split(dv), d_down.unbind(0), layout="NT",
                     m_splits=counts)
    du = (dh.float() * 2 * u.float().relu()).to(torch.bfloat16)
    dx_rows = torch.empty_like(x_rows)
    _te_grouped_gemm(up.unbind(0), split(du), dx_rows, layout="NN", m_splits=counts,
                     single_output=True)
    d_up = torch.zeros_like(up)
    _te_grouped_gemm(split(x_rows), split(du), d_up.unbind(0), layout="NT",
                     m_splits=counts)
    routes_dx = dx_rows.new_zeros(m * topk, k).index_copy_(0, order, dx_rows).view(m, topk, k)
    d_routes = d_weight.new_zeros(m * topk).index_copy_(0, order, d_weight).view(m, topk)
    dx = routes_dx if per_route else sum_route_grads(routes_dx)
    return dx.to(x.dtype), d_up, d_down, d_routes


def sum_route_grads(per_route):
    """[M, topk, K] -> [M, K] as DS4's deterministic scatter backward: slot
    order, rounded to BF16 after each add."""
    total = per_route[:, 0]
    for slot in range(1, per_route.shape[1]):
        total = (total.float() + per_route[:, slot].float()).to(torch.bfloat16)
    return total


class RoutedExpertsVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down, routes, ids, owner):
        out, fc1, visible, activated = owner._visible(x, ids, routes, return_fc1=True)
        ctx.owner, ctx.versions = owner, owner.weights._versions()
        ctx.save_for_backward(x, fc1, visible, up, down, routes, ids, activated)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        if ctx.owner.weights._versions() != ctx.versions:
            raise RuntimeError("Expert masters changed before backward")
        x, fc1, visible, up, down, routes, ids, activated = ctx.saved_tensors
        dx, d_up, d_down, d_routes = routed_vjp(
            x, fc1, visible, up, down, routes, ids, dy, activated
        )
        return dx, d_up, d_down, d_routes, None, None
