# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native PyTorch Mamba2 SSD used as the recomputed training VJP.

``chunk_scan`` follows ``mamba2_chunk_scan`` from Transformers 5.16.1
``models/nemotron_h/modeling_nemotron_h.py`` (Apache-2.0) operation by
operation. The only change is that the five broadcast contractions
(``G``, ``Y_diag``, ``states``, the inter-chunk state recurrence and
``C @ states``) go through ``chunk_product_sum``, which evaluates one chunk at a
time so the broadcast intermediates of a 16384-token sequence fit in memory
(the dense recurrence alone is quadratic in the chunk count: 32.5 GiB at 16384
tokens). Each chunk reduces the full contraction axis exactly as the dense
expression does.
"""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from transformers.models.nemotron_h.modeling_nemotron_h import (
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
)


class _ChunkProductSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, reduce_dim, chunk_dim):
        if left.dtype != torch.float32 or right.dtype != torch.float32:
            raise ValueError("Native SSD contraction operands must be FP32")
        if left.ndim != right.ndim:
            raise ValueError("Require explicit native broadcast dimensions")
        reduce_dim %= left.ndim
        chunk_dim %= left.ndim
        if (
            reduce_dim == chunk_dim
            or left.shape[chunk_dim] < 1
            or right.shape[chunk_dim] not in (1, left.shape[chunk_dim])
        ):
            raise ValueError("Only independent chunk axes (right may broadcast) may split")
        ctx.reduce_dim, ctx.chunk_dim = reduce_dim, chunk_dim
        ctx.save_for_backward(left, right)
        output_dim = chunk_dim - (reduce_dim < chunk_dim)
        parts = []
        shared = right.shape[chunk_dim] == 1 < left.shape[chunk_dim]
        for index in range(left.shape[chunk_dim]):
            a = left.narrow(chunk_dim, index, 1)
            b = right if shared else right.narrow(chunk_dim, index, 1)
            parts.append((a * b).sum(dim=reduce_dim))
        return torch.cat(parts, dim=output_dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        left, right = ctx.saved_tensors
        dim, chunk = ctx.reduce_dim, ctx.chunk_dim
        output_dim = chunk - (dim < chunk)
        dleft = torch.empty_like(left) if ctx.needs_input_grad[0] else None
        shared = right.shape[chunk] == 1 < left.shape[chunk]
        dright = None
        if ctx.needs_input_grad[1]:
            dright = torch.zeros_like(right) if shared else torch.empty_like(right)
        for index in range(left.shape[chunk]):
            a = left.narrow(chunk, index, 1)
            b = right if shared else right.narrow(chunk, index, 1)
            upstream = grad.narrow(output_dim, index, 1).unsqueeze(dim)
            upstream = upstream.expand(torch.broadcast_shapes(a.shape, b.shape))
            if dleft is not None:
                dleft.narrow(chunk, index, 1).copy_((upstream * b).sum_to_size(a.shape))
            if dright is not None and shared:
                dright.add_((upstream * a).sum_to_size(b.shape))
            elif dright is not None:
                dright.narrow(chunk, index, 1).copy_((upstream * a).sum_to_size(b.shape))
        return dleft, dright, None, None


def chunk_product_sum(left, right, reduce_dim, chunk_dim=1):
    """``(left * right).sum(reduce_dim)``, evaluated one SSD chunk at a time."""
    return _ChunkProductSum.apply(left, right, reduce_dim, chunk_dim)


def chunk_scan(
    hidden_states,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
):
    batch_size, sequence_length, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])

    hidden_states = hidden_states.float()
    B = B.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)
    C = C.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    D_residual = None
    if D is not None:
        D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    hidden_states = hidden_states * dt[..., None].float()
    A = A.to(hidden_states.dtype) * dt.float()

    hidden_states, A, B, C = [
        reshape_into_chunks(tensor, pad_size, chunk_size) for tensor in (hidden_states, A, B, C)
    ]

    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(segment_sum(A))
    G = chunk_product_sum(C[:, :, :, None, :, :], B[:, :, None, :, :, :], -1)
    M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
    Y_diag = chunk_product_sum(M[..., None], hidden_states[:, :, None], 3)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = chunk_product_sum(B_decay[..., None, :], hidden_states[..., None], 2)

    previous_states = (
        initial_states[:, None].to(dtype=states.dtype, device=states.device)
        if initial_states is not None
        else torch.zeros_like(states[:, :1])
    )
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    new_states = chunk_product_sum(decay_chunk[..., None, None], states[:, :, None, ...], 1, 2)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = (
        chunk_product_sum(C[..., None, :], states[:, :, None, ...], -1)
        * state_decay_out.permute(0, 2, 3, 1)[..., None]
    )

    output = Y_diag + Y_off
    output = output.reshape(batch_size, -1, num_heads, head_dim)

    if D_residual is not None:
        output = output + D_residual

    if pad_size > 0:
        output = output[:, :sequence_length]

    if return_final_states:
        return output, final_state

    return output
