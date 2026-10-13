"""FlashAttention varlen VJP for the serving-visible KV-cache attention."""

import torch


class _AttentionVJP(torch.autograd.Function):
    """FlashAttention VJP on the dequantized Q/K/V of the visible forward.

    The visible output multiplies an FP8-rounded P by V, so it is not the
    softmax(QK^T)V that FlashAttention's backward assumes when it forms
    D = rowsum(dO * O). Backward recomputes the BF16 output and LSE on the
    dequantized Q/K/V and passes those instead. A BF16 KV cache passes its
    Q/K/V unchanged.
    """

    @staticmethod
    def forward(ctx, q, k, v, module, boundaries, key_boundaries):
        output, _, q_ref, k_ref, v_ref = module._visible(
            q, k, v, boundaries, key_boundaries
        )
        ctx.save_for_backward(q_ref, k_ref, v_ref)
        ctx.bounds = boundaries, key_boundaries
        ctx.scale = module.scale
        return output

    @staticmethod
    def backward(ctx, grad):
        from vllm.vllm_flash_attn.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
        )

        q, k, v = ctx.saved_tensors
        # Causal masks align the queries with the last keys (CP shards).
        cu_q, cu_k = (
            torch.tensor(b, dtype=torch.int32, device=q.device) for b in ctx.bounds
        )
        longest_q, longest_k = (
            max(end - start for start, end in zip(b, b[1:])) for b in ctx.bounds
        )
        varlen = dict(
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=longest_q,
            max_seqlen_k=longest_k,
        )
        output, lse, *_ = _flash_attn_fwd(
            q, k, v, softmax_scale=ctx.scale, causal=True, return_lse=True,
            **varlen,
        )
        # Fixed-order dQ accumulation: full_determinism needs run-to-run
        # identical gradients.
        gradients = _flash_attn_bwd(
            q, k, v, output, grad.contiguous(), lse, ctx.scale, True,
            deterministic=True, **varlen,
        )
        return *gradients[:3], None, None, None
