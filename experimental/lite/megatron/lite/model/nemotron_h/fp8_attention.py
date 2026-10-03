"""Packed serving-visible FP8 Q/KV attention with a FlashAttention VJP."""

from types import SimpleNamespace

import torch

from .kernels import scaled_fp8_quant


class _Fp8AttentionVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_scale, v_scale, module, boundaries):
        output, q_ref, k_ref, v_ref = module._visible(
            q, k, v, boundaries, return_query=True
        )
        ctx.save_for_backward(q_ref, k_ref, v_ref, q, k, v, k_scale, v_scale)
        ctx.boundaries, ctx.scale = boundaries, module.scale
        return output

    @staticmethod
    def backward(ctx, grad):
        from vllm.vllm_flash_attn.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
        )

        q, k, v, *_ = ctx.saved_tensors
        longest = max(b - a for a, b in zip(ctx.boundaries, ctx.boundaries[1:]))
        cu = torch.tensor(ctx.boundaries, dtype=torch.int32, device=q.device)
        varlen = dict(
            cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=longest, max_seqlen_k=longest
        )
        # The visible output's FP8 P@V is off the BF16 Q/K/V; recompute out/LSE.
        output, lse, *_ = _flash_attn_fwd(
            q, k, v, softmax_scale=ctx.scale, causal=True, return_lse=True, **varlen
        )
        # Fixed-order dQ accumulation: full_determinism needs run-to-run
        # identical gradients.
        gradients = _flash_attn_bwd(
            q, k, v, output, grad, lse, ctx.scale, True, deterministic=True, **varlen
        )
        return *gradients, None, None, None, None


class Fp8KVAttention(torch.nn.Module):
    """Reuse serving query/cache quantization, with fixed-scale identity STE.

    The VJP is the FlashAttention varlen forward/backward recomputed on the BF16
    dequantized Q/K/V from the visible forward. No claim of identical internal
    softmax arithmetic or training quality is implied. This initial adapter
    supports TP1, causal packed sequences starting at zero, without prefix
    sharing or sliding windows.
    """

    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        k_scale,
        v_scale,
        *,
        block_size=16,
        backend="triton",
    ):
        super().__init__()
        for name, value in (("k_scale", k_scale), ("v_scale", v_scale)):
            if (
                value.dtype != torch.float32
                or value.numel() != 1
                or not torch.isfinite(value).all()
                or not (value > 0).all()
            ):
                raise ValueError("Expected fixed positive FP32 KV scales")
            if value.requires_grad:
                raise ValueError("Trainable KV scales are not supported")
            self.register_buffer(name, value.detach().clone())
        if block_size not in (16, 64):
            raise ValueError("Unvalidated FP8 KV block size")
        self.scale = head_dim**-0.5
        self.block_size = block_size
        self.heads = (num_heads, num_kv_heads, head_dim)
        if backend == "triton":
            from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

            self.impl = TritonAttentionImpl(
                num_heads, head_dim, self.scale, num_kv_heads, None, None, "fp8_e4m3"
            )
        elif backend == "fa4":
            if self.heads != (32, 2, 128):
                raise ValueError("Aligned FA4 requires Nemotron Q32/KV2/D128")
            self.block_size = 6768
            self.impl = None
        else:
            raise ValueError(f"Unsupported FP8 attention backend: {backend}")

    def _visible(self, q, k, v, boundaries, *, return_query=False):
        from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

        lengths = [b - a for a, b in zip(boundaries, boundaries[1:])]
        counts = [(n + self.block_size - 1) // self.block_size for n in lengths]
        table_rows, slots, block = [], [], 0
        for length, count in zip(lengths, counts):
            table_rows.append(
                list(range(block, block + count)) + [0] * (max(counts) - count)
            )
            slots.extend(
                range(block * self.block_size, block * self.block_size + length)
            )
            block += count
        slots = torch.tensor(slots, dtype=torch.int64, device=q.device)
        table = torch.tensor(table_rows, dtype=torch.int32, device=q.device)
        cache = torch.zeros(
            block,
            k.shape[1],
            self.block_size,
            k.shape[2] * 2,
            dtype=torch.uint8,
            device=q.device,
        )
        # Serving loads scale parameters under the BF16 model default dtype,
        # then copies their rounded values into FP32 runtime buffers. Keep the
        # original checkpoint buffers unchanged for export.
        k_scale = self.k_scale.to(torch.bfloat16).float()
        v_scale = self.v_scale.to(torch.bfloat16).float()
        layer = SimpleNamespace(
            _k_scale=k_scale,
            _v_scale=v_scale,
            _q_scale=torch.ones_like(self.k_scale),
        )
        quantized_q, _ = scaled_fp8_quant(q.flatten(1), layer._q_scale)
        quantized_q = quantized_q.view(q.shape)
        self.impl.do_kv_cache_update(layer, k, v, cache, slots)
        meta = TritonAttentionMetadata(
            num_actual_tokens=q.shape[0],
            max_query_len=max(lengths),
            query_start_loc=torch.tensor(
                boundaries, dtype=torch.int32, device=q.device
            ),
            max_seq_len=max(lengths),
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device=q.device),
            block_table=table,
            slot_mapping=slots,
            seq_threshold_3D=0,
            num_par_softmax_segments=1,
            softmax_segm_output=None,
            softmax_segm_max=None,
            softmax_segm_expsum=None,
            causal=True,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )
        output = torch.empty_like(q)
        self.impl.forward(layer, quantized_q, k, v, cache, meta, output)
        # Gather bytes before reinterpreting FP8: indexing FP8 is not universal.
        values = cache.transpose(1, 2)[
            slots // self.block_size, slots % self.block_size
        ]
        key, value = values.view(torch.float8_e4m3fn).split(k.shape[2], dim=-1)
        result = (
            output,
            (key.float() * k_scale).to(k.dtype),
            (value.float() * v_scale).to(v.dtype),
        )
        if return_query:
            q_ref = (quantized_q.float() * layer._q_scale).to(q.dtype)
            return output, q_ref, *result[1:]
        return result

    def forward(self, q, k, v, meta):
        import vllm.envs as envs

        if not envs.VLLM_BATCH_INVARIANT:
            raise RuntimeError("Aligned FP8 attention requires batch invariance")
        meta.validate_tokens(q.shape[0])
        hq, hkv, dim = self.heads
        if (
            q.shape[1:] != (hq, dim)
            or k.shape != v.shape
            or k.shape != (q.shape[0], hkv, dim)
        ):
            raise ValueError("Attention geometry does not match the module")
        if any(
            x.dtype != torch.bfloat16
            or not x.is_cuda
            or x.device != self.k_scale.device
            for x in (q, k, v)
        ):
            raise ValueError("Expected CUDA BF16 QKV on the scale device")
        return _Fp8AttentionVJP.apply(
            q, k, v, self.k_scale, self.v_scale, self, meta.boundaries
        )


class Fa4Fp8KVAttention(Fp8KVAttention):
    """Serving-visible FA4 forward with the fixed-scale FlashAttention VJP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, backend="fa4", **kwargs)

    def _visible(self, q, k, v, boundaries, *, return_query=False):
        from vllm.model_executor.models.nemotron_h_fa4 import (
            MAX_SEQ_LEN,
            NUM_SPLITS,
            SEQLEN_K_PER_SPLIT,
        )
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

        if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != q.shape[0]:
            raise ValueError("Expected complete zero-origin packed boundaries")
        lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
        if any(length <= 0 or length > MAX_SEQ_LEN for length in lengths):
            raise ValueError(f"Expected nonempty sequences no longer than {MAX_SEQ_LEN}")
        pages = -(-MAX_SEQ_LEN // 6768)
        table = torch.arange(
            len(lengths) * pages, device=q.device, dtype=torch.int32
        ).reshape(-1, pages)
        slots = torch.cat(
            [
                torch.arange(length, device=q.device, dtype=torch.int64)
                + index * pages * 6768
                for index, length in enumerate(lengths)
            ]
        )
        lengths = torch.tensor(lengths, device=q.device, dtype=torch.int32)
        cu = torch.tensor(boundaries, device=q.device, dtype=torch.int32)
        cache = torch.zeros(
            table.numel(), 2, 6768, 256, dtype=torch.uint8, device=q.device
        )
        ks = self.k_scale.to(torch.bfloat16).float()
        vs = self.v_scale.to(torch.bfloat16).float()
        qs = torch.ones_like(ks)
        quantized_q, _ = scaled_fp8_quant(q.flatten(1), qs)
        quantized_q = quantized_q.view(q.shape)
        key_cache, value_cache = cache.transpose(1, 2).split(128, dim=-1)
        reshape_and_cache_flash(k, v, key_cache, value_cache, slots, "fp8_e4m3", ks, vs)
        key_cache = key_cache.view(torch.float8_e4m3fn)
        value_cache = value_cache.view(torch.float8_e4m3fn)
        output = torch.empty_like(q)
        scale_shape = (lengths.numel(), 2)
        _flash_attn_fwd(
            quantized_q,
            key_cache,
            value_cache,
            cu_seqlens_q=cu,
            seqused_k=lengths,
            max_seqlen_q=MAX_SEQ_LEN,
            max_seqlen_k=MAX_SEQ_LEN,
            page_table=table,
            softmax_scale=self.scale,
            causal=True,
            q_descale=qs.expand(scale_shape),
            k_descale=ks.expand(scale_shape),
            v_descale=vs.expand(scale_shape),
            tile_mn=(128, 128),
            pack_gqa=True,
            num_splits=NUM_SPLITS,
            seqlen_k_per_split=SEQLEN_K_PER_SPLIT,
            disable_scheduler_metadata=True,
            out=output,
        )
        saved = cache.transpose(1, 2)[slots // 6768, slots % 6768]
        saved_k, saved_v = saved.view(torch.float8_e4m3fn).split(128, -1)
        k_ref = (saved_k.float() * ks).to(k.dtype).detach().clone()
        v_ref = (saved_v.float() * vs).to(v.dtype).detach().clone()
        if return_query:
            q_ref = (quantized_q.float() * qs).to(q.dtype)
            return output, q_ref.detach().clone(), k_ref, v_ref
        return output, k_ref, v_ref
