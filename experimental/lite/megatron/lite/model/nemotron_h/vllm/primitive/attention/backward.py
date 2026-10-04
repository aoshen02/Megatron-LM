"""FlashAttention varlen VJP for the serving-visible FP8 KV attention."""

import torch


class _Fp8AttentionVJP(torch.autograd.Function):
    """FlashAttention VJP on the dequantized Q/K/V of the visible forward.

    The visible output multiplies an FP8-rounded P by V, so it is not the
    softmax(QK^T)V that FlashAttention's backward assumes when it forms
    D = rowsum(dO * O). Backward recomputes the BF16 output and LSE on the
    dequantized Q/K/V and passes those instead.
    """

    @staticmethod
    def forward(ctx, q, k, v, module, boundaries):
        output, _, q_ref, k_ref, v_ref = module._visible(q, k, v, boundaries)
        ctx.save_for_backward(q_ref, k_ref, v_ref)
        ctx.boundaries, ctx.scale = boundaries, module.scale
        return output

    @staticmethod
    def backward(ctx, grad):
        from vllm.vllm_flash_attn.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
        )

        q, k, v = ctx.saved_tensors
        longest = max(b - a for a, b in zip(ctx.boundaries, ctx.boundaries[1:]))
        cu = torch.tensor(ctx.boundaries, dtype=torch.int32, device=q.device)
        varlen = dict(
            cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=longest,
            max_seqlen_k=longest,
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
        return *gradients[:3], None, None
