"""Contiguous context parallelism for the vLLM-aligned Nemotron-H layers.

Rank r holds packed tokens [r * T / cp, (r + 1) * T / cp). Attention gathers
the keys and values; the Mamba core runs head-parallel on the full sequence.
"""

import torch
import torch.distributed as dist

from megatron.lite.primitive.parallel.cp import (
    all_to_all_hidden_shards,
    build_headwise_section_perm,
    get_parameter_local_cp_headwise,
)


class _GatherSequence(torch.autograd.Function):
    """All-gather the token shards; backward sums every rank's gradient for
    this shard in rank order in FP32."""

    @staticmethod
    def forward(ctx, x, group, size):
        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(size)]
        dist.all_gather(parts, x, group=group)
        ctx.group, ctx.size = group, size
        return torch.cat(parts)

    @staticmethod
    def backward(ctx, grad):
        send = grad.contiguous().view(ctx.size, -1)
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=ctx.group)
        total = received[0].float()
        for part in received[1:]:
            total += part.float()
        return total.to(grad.dtype).view(-1, *grad.shape[1:]), None, None


def gather_sequence(x, meta):
    """[T / cp, ...] token shard -> [T, ...] packed sequence."""
    return _GatherSequence.apply(x, meta.cp_group, meta.cp_size)


def heads_to_full_sequence(x, sections, meta):
    """[T / cp, sum(sections)] -> [T, sum(sections) / cp]: every section's
    rank-``cp_rank`` block of heads over the whole packed sequence."""
    perm = build_headwise_section_perm(sections, meta.cp_size, x.device)
    x = x.index_select(-1, perm)
    width = x.shape[-1] // meta.cp_size
    received = all_to_all_hidden_shards(
        [x[:, k * width : (k + 1) * width] for k in range(meta.cp_size)],
        meta.cp_group,
    )
    return torch.cat(received)


def full_sequence_to_heads(x, meta):
    """Inverse for one section: [T, width / cp] -> [T / cp, width]."""
    received = all_to_all_hidden_shards(list(x.chunk(meta.cp_size)), meta.cp_group)
    return torch.cat(received, dim=-1)


def headwise(parameter, meta, sections=None):
    return get_parameter_local_cp_headwise(
        parameter, 0, meta.cp_size, meta.cp_rank, split_sections=sections
    )
