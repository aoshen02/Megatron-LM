"""Diagnostic compact V2 execution; fixed-scale STE and BF16 edges unchanged.

Ported from the frozen compact_moe_vjp_splitk diagnostic, using F32-TMA
without split K. Eager single-owner execution only; not a quality approval.
"""

import torch

from .nvfp4_compact_gemm import GroupedTF32 as GroupedTF32
from .nvfp4_compact_gemm import row_gemm, weight_gemm


def route_plan(ids, experts):
    """GPU stable grouping, <=R+4E storage, no host route-count extraction."""
    flat = ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    sorted_ids = flat[order]
    counts = torch.bincount(flat, minlength=experts)
    padded = ((counts + 3) // 4 * 4).clamp_min(4)
    starts = padded.cumsum(0) - padded
    original_starts = counts.cumsum(0) - counts
    positions = starts[sorted_ids] + torch.arange(flat.numel(), device=ids.device)
    positions -= original_starts[sorted_ids]
    return order, positions, padded, starts, counts


@torch.no_grad()
def compact_vjp(adapter, x, up, down, routes, ids, dy):
    if (
        x.dtype != torch.bfloat16
        or dy.dtype != x.dtype
        or dy.shape != x.shape
        or up.dtype != torch.float32
        or down.dtype != torch.float32
        or routes.dtype != torch.float32
        or ids.shape != (x.shape[0], 6)
        or routes.shape != ids.shape
        or up.shape[2] != x.shape[1]
        or down.shape != (up.shape[0], x.shape[1], up.shape[1])
    ):
        raise ValueError("Expected BF16 edges, FP32 QDQ/routes and top6")
    m, k = x.shape
    e = up.shape[0]
    order, pos, count, start, actual_count = route_plan(ids, e)
    capacity = ids.numel() + 4 * e
    token = torch.div(order, 6, rounding_mode="floor")
    xp = x.new_zeros((capacity, k), dtype=torch.float32)
    xp.index_copy_(0, pos, x.float()[token])
    u = row_gemm(adapter, xp, up, count, start).bfloat16()
    h = u.float().relu().square().bfloat16()
    v = row_gemm(adapter, h.float(), down, count, start).bfloat16()
    gr = dy.float()[token]
    dr_sorted = (gr * v[pos].float()).sum(-1)
    dr = routes.new_empty(ids.numel())
    dr.index_copy_(0, order, dr_sorted)
    del v, dr_sorted
    gv = xp.new_zeros((capacity, k))
    gv.index_copy_(0, pos, (gr * routes.flatten()[order, None]).bfloat16().float())
    dd = weight_gemm(adapter, gv, h.float(), count, start)
    gh = (
        row_gemm(adapter, gv, down, count, start, transpose_weight=True)
        .bfloat16()
        .float()
    )
    gu = (gh * 2 * u.float().relu()).bfloat16().float()
    del gh, gv, h, u
    du = weight_gemm(adapter, gu, xp, count, start)
    dx_sorted = row_gemm(adapter, gu, up, count, start, transpose_weight=True)[pos]
    dx_routes = xp.new_empty((ids.numel(), k))
    dx_routes.index_copy_(0, order, dx_sorted)
    dx = torch.zeros_like(x, dtype=torch.float32)
    dx_routes = dx_routes.view(m, 6, k)
    for slot in reversed(range(6)):
        dx = dx + dx_routes[:, slot]
    return dx.to(x.dtype), du, dd, dr.view(m, 6)
