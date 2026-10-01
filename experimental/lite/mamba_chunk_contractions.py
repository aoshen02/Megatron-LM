"""Isolated SSD broadcast contractions with chunk-local first-order VJPs."""

import torch
from torch.autograd.function import once_differentiable


class _ChunkProductSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, reduce_dim, chunk_dim):
        if left.dtype != torch.float32 or right.dtype != torch.float32:
            raise ValueError("Native SSD contraction operands must be FP32")
        if left.ndim != right.ndim:
            raise ValueError("Require explicit native broadcast dimensions")
        reduce_dim %= left.ndim
        chunk_dim %= left.ndim
        if (reduce_dim == chunk_dim or left.shape[chunk_dim] < 1
                or left.shape[chunk_dim] != right.shape[chunk_dim]):
            raise ValueError("Only independent, non-broadcast chunk axes may split")
        ctx.reduce_dim, ctx.chunk_dim = reduce_dim, chunk_dim
        ctx.save_for_backward(left, right)
        output_dim = chunk_dim - (reduce_dim < chunk_dim)
        parts = []
        for index in range(left.shape[chunk_dim]):
            a, b = (value.narrow(chunk_dim, index, 1) for value in (left, right))
            parts.append((a * b).sum(dim=reduce_dim))
        return torch.cat(parts, dim=output_dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        left, right = ctx.saved_tensors
        dim, chunk = ctx.reduce_dim, ctx.chunk_dim
        output_dim = chunk - (dim < chunk)
        dleft = torch.empty_like(left) if ctx.needs_input_grad[0] else None
        dright = torch.empty_like(right) if ctx.needs_input_grad[1] else None
        for index in range(left.shape[chunk]):
            a, b = (value.narrow(chunk, index, 1) for value in (left, right))
            upstream = grad.narrow(output_dim, index, 1).unsqueeze(dim)
            upstream = upstream.expand(torch.broadcast_shapes(a.shape, b.shape))
            if dleft is not None:
                dleft.narrow(chunk, index, 1).copy_((upstream * b).sum_to_size(a.shape))
            if dright is not None:
                dright.narrow(chunk, index, 1).copy_((upstream * a).sum_to_size(b.shape))
        return dleft, dright, None, None


def chunk_product_sum(left, right, reduce_dim, chunk_dim=1):
    """Preserve the full reduction axis; split only independent SSD chunks."""
    return _ChunkProductSum.apply(left, right, reduce_dim, chunk_dim)
