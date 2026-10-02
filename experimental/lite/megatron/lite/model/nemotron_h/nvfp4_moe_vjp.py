"""Grouped routed-expert surrogate VJP, contract v3.

Fixed-scale STE onto the FP32 masters with BF16 intermediate edges. The expert
GEMMs of the compact backend read FP32 storage as TF32 operands (the dequantized
weights are truncated toward zero; BF16 activations are exact) and accumulate in
FP32. The padded backend evaluates the same graph with FP32 GEMMs and serves as
an FP32 diagnostic reference. This module establishes no SFT/RL quality approval.
"""

import hashlib
from pathlib import Path

import torch

SURROGATE_CONTRACT = "moe-fixedscale-grouped-tf32rz-bf16edges-v3"
PADDED_BACKEND = "padded-v2"
COMPACT_BACKEND = "compact-f32-tma-nosplit"
COMPACT_KERNEL_SHA = "b30d3192f88a5dc324278f0b3f22b9cb44bcea6c8eb2c2cfa380c07d75368e74"


def compact_kernel_path():
    """Return the kernel shipped with this training package."""
    return Path(__file__).with_name("grouped_gemm_tma_f32.py").resolve()


def validate_backend(backend, kernel_source, surrogate_contract):
    """Validate diagnostic selection without importing a CUDA runtime."""
    if backend not in (PADDED_BACKEND, COMPACT_BACKEND):
        raise ValueError("Unknown routed VJP backend")
    if backend == PADDED_BACKEND:
        if kernel_source is not None:
            raise ValueError("padded-v2 does not accept a compact kernel artifact")
        return
    if surrogate_contract != SURROGATE_CONTRACT:
        raise ValueError("Compact backend requires the explicit v3 surrogate contract")
    path = compact_kernel_path()
    if kernel_source is not None and Path(kernel_source).resolve() != path:
        raise ValueError("Compact backend requires the installed package kernel")
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != COMPACT_KERNEL_SHA
    ):
        raise ValueError("Missing or unreviewed compact kernel artifact")


def validate_matmul_policy(device):
    if torch.is_autocast_enabled(device.type):
        raise RuntimeError("Routed surrogate requires caller-disabled autocast")
    if device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "Routed surrogate requires caller-disabled TF32 matmul; only the compact "
            "kernel opts its own operands into TF32"
        )


def grouped_vjp(x, up, down, routes, ids, dy):
    """Fixed top6 padded FP32 VJP; no per-token Python dispatch loops."""
    validate_matmul_policy(x.device)
    m, k = x.shape
    topk = ids.shape[1]
    if topk != 6:
        raise ValueError("Routed surrogate requires top6")
    experts = up.shape[0]
    count = m * topk
    flat_ids = ids.reshape(-1).long()
    route_index = torch.arange(count, device=x.device)
    flat_pos = flat_ids * count + route_index
    token_index = torch.arange(m, device=x.device).repeat_interleave(topk)

    def dispatch(rows):
        padded = rows.new_zeros(experts * count, rows.shape[-1])
        return padded.index_copy(0, flat_pos, rows).view(experts, count, -1)

    xp = dispatch(x.float().index_select(0, token_index))
    u = torch.bmm(xp, up.transpose(1, 2)).to(torch.bfloat16)
    h = u.float().relu().square().to(torch.bfloat16)
    v = torch.bmm(h.float(), down.transpose(1, 2)).to(torch.bfloat16)
    gr = dy.float().index_select(0, token_index)
    selected_v = v[flat_ids, route_index].float()
    dr = (gr * selected_v).sum(-1).view(m, topk)
    gv = dispatch((gr * routes.reshape(-1, 1)).to(torch.bfloat16).float())
    dd = torch.bmm(gv.transpose(1, 2), h.float())
    gh = torch.bmm(gv, down).to(torch.bfloat16).float()
    gu = (gh * 2 * u.float().relu()).to(torch.bfloat16).float()
    du = torch.bmm(gu.transpose(1, 2), xp)
    dx_routes = torch.bmm(gu, up)[flat_ids, route_index].view(m, topk, k)
    dx = torch.zeros_like(x, dtype=torch.float32)
    for slot in reversed(range(topk)):
        dx = dx + dx_routes[:, slot]
    return dx.to(x.dtype), du, dd, dr


class RoutedSurrogateVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up_master, down_master, routes, ids, owner):
        validate_matmul_policy(x.device)
        ctx.owner = owner
        ctx.generation = owner._surrogate_generation
        ctx.versions = owner.weights._versions()
        ctx.recompute = getattr(owner, "recompute_surrogate", False)
        ctx.backend = (
            getattr(owner, "routed_vjp_backend", PADDED_BACKEND),
            getattr(owner, "routed_vjp_kernel_source", None),
        )
        ctx.save_for_backward(
            x, routes, ids, up_master, down_master,
            None if ctx.recompute else owner._surrogate_up,
            None if ctx.recompute else owner._surrogate_down,
        )
        return owner._visible(x, ids, routes)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        owner = ctx.owner
        owner._validate_checkpoint()
        if (
            not owner._ready
            or owner._surrogate_generation != ctx.generation
            or owner.weights._versions() != ctx.versions
            or getattr(owner, "recompute_surrogate", False) != ctx.recompute
            or ctx.backend != (
                getattr(owner, "routed_vjp_backend", PADDED_BACKEND),
                getattr(owner, "routed_vjp_kernel_source", None),
            )
        ):
            raise RuntimeError("Routed deployment changed before backward")
        x, routes, ids, up_master, down_master, up, down = ctx.saved_tensors
        if (
            owner.weights.up_proj is not up_master
            or owner.weights.down_proj is not down_master
        ):
            raise RuntimeError("Expert masters changed before backward")
        if ctx.recompute:
            up, down = owner._recompute_surrogate_weights()
        dx, du, dd, dr = owner._routed_vjp(x, up, down, routes, ids, dy)
        return dx, du, dd, dr, None, None
