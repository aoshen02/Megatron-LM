"""FlashAttention varlen VJP for the serving-visible FP8 KV attention."""

import torch


class _Fp8AttentionVJP(torch.autograd.Function):
    """DeepSeek-V4 ``_VisibleSparseAttentionFunction`` pattern: save the
    visible output and LSE, no forward recompute in backward."""

    @staticmethod
    def forward(ctx, q, k, v, module, boundaries):
        output, lse, q_ref, k_ref, v_ref = module._visible(q, k, v, boundaries)
        ctx.save_for_backward(q_ref, k_ref, v_ref, output, lse)
        ctx.boundaries, ctx.scale = boundaries, module.scale
        return output

    @staticmethod
    def backward(ctx, grad):
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_bwd

        q, k, v, output, lse = ctx.saved_tensors
        longest = max(b - a for a, b in zip(ctx.boundaries, ctx.boundaries[1:]))
        cu = torch.tensor(ctx.boundaries, dtype=torch.int32, device=q.device)
        # Fixed-order dQ accumulation: full_determinism needs run-to-run
        # identical gradients.
        gradients = _flash_attn_bwd(
            q, k, v, output, grad.contiguous(), lse, ctx.scale, True,
            cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=longest,
            max_seqlen_k=longest, deterministic=True,
        )
        return *gradients[:3], None, None
