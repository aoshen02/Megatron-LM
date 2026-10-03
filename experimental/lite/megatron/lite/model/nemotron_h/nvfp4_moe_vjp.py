"""Routed-expert VJP on the BF16 masters (TE ``high_precision`` semantics).

Mirrors the DeepSeek-V4 aligned actor's grouped backward
(NVIDIA/Megatron-LM#7050): the visible FC1 output is saved in forward and the
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


def routed_vjp(x, fc1, up, down, routes, ids, dy):
    """ReLU2 routed experts: ``y = sum_s routes[:, s] * down(relu(up(x))**2)``.

    Args:
        x: BF16 tokens, ``[M, K]``.
        fc1: Visible FC1 output per route (token-major, slot-minor), ``[M*topk, I]``.
        up, down: BF16 masters, ``[E, I, K]`` and ``[E, K, I]``.
        routes: FP32 routing weights, ``[M, topk]``.
        ids: Expert ids, ``[M, topk]``.
        dy: BF16 output gradient, ``[M, K]``.

    Returns:
        ``(dx, d_up, d_down, d_routes)``.
    """
    m, k = x.shape
    topk, experts = ids.shape[1], up.shape[0]
    flat = ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=experts).tolist()
    token = order // topk
    u = fc1.index_select(0, order)
    h = u.float().relu().square().to(torch.bfloat16)
    x_rows = x.index_select(0, token)
    dy_rows = dy.index_select(0, token)
    weight = routes.reshape(-1).index_select(0, order)
    dv = (dy_rows.float() * weight[:, None]).to(torch.bfloat16)

    def split(rows):
        return torch.split(rows, counts)

    v = torch.empty_like(dy_rows)
    _te_grouped_gemm(down.unbind(0), split(h), v, layout="TN", m_splits=counts,
                     single_output=True)
    d_weight = (dy_rows.float() * v.float()).sum(-1)
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
    per_route = torch.empty_like(dx_rows).index_copy_(0, order, dx_rows).view(m, topk, k)
    dx = per_route[:, 0].float()
    for slot in range(1, topk):
        dx = dx + per_route[:, slot].float()
    d_routes = torch.empty_like(d_weight).index_copy_(0, order, d_weight).view(m, topk)
    return dx.to(x.dtype), d_up, d_down, d_routes


class RoutedExpertsVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down, routes, ids, owner):
        out, fc1 = owner._visible(x, ids, routes, return_fc1=True)
        ctx.owner, ctx.versions = owner, owner.weights._versions()
        ctx.save_for_backward(x, fc1, up, down, routes, ids)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        if ctx.owner.weights._versions() != ctx.versions:
            raise RuntimeError("Expert masters changed before backward")
        x, fc1, up, down, routes, ids = ctx.saved_tensors
        dx, d_up, d_down, d_routes = routed_vjp(x, fc1, up, down, routes, ids, dy)
        return dx, d_up, d_down, d_routes, None, None
