"""Nemotron-H attention over a static FP8 or a BF16 KV cache."""

import torch
from torch import nn

from megatron.lite.model.nemotron_h.vllm.primitive.attention.backward import _AttentionVJP
from megatron.lite.model.nemotron_h.vllm.primitive.dense import projection
from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import SSMMeta


# Query tokens per FA4 call; bounds the split-KV scratch at long context.
QUERY_CHUNK_TOKENS = 4096


class Fa4KVAttention(torch.nn.Module):
    """Serving FA4 (``FlashAttnFixedSplitBackend``) with 6768-token pages and
    the fixed split-KV schedule. Causal packed sequences starting at zero.

    ``fp8`` (the release): static FP8 KV cache, FP8 query and KV quantization
    with the fixed scales. Otherwise (the BF16 activation mode,
    ``kv_cache_dtype=bfloat16``): BF16 query and KV; the checkpoint's scales,
    if given, are only held so the weight export keeps sending them, because
    vLLM's ModelOpt KV-cache method still expects them on every refit.
    """

    def __init__(self, num_heads, num_kv_heads, head_dim, k_scale=None, v_scale=None, *, fp8=None):
        super().__init__()
        if (k_scale is None) != (v_scale is None):
            raise ValueError("Expected both FP8 KV scales or neither")
        self.fp8 = k_scale is not None if fp8 is None else fp8
        if self.fp8 and k_scale is None:
            raise ValueError("The FP8 KV cache needs the checkpoint's KV scales")
        if k_scale is not None:
            self.register_buffer("k_scale", k_scale.detach().clone())
            self.register_buffer("v_scale", v_scale.detach().clone())
        self.scale = head_dim**-0.5
        self.heads = (num_heads, num_kv_heads, head_dim)

    def _visible(self, q, k, v, boundaries):
        """Return the serving output, ``None`` (no LSE) and the Q/K/V it attends over."""
        from vllm.v1.attention.backends.flash_attn_fixed_split import (
            MAX_SEQ_LEN,
            NUM_SPLITS,
            SEQLEN_K_PER_SPLIT,
        )
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
        from vllm import _custom_ops as ops
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

        lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
        if any(length > MAX_SEQ_LEN for length in lengths):
            raise ValueError(f"Expected sequences no longer than {MAX_SEQ_LEN}")
        # Each sequence owns the 6768-token pages it fills; the rest of its
        # MAX_SEQ_LEN-wide table row repeats its last page, never read.
        width = -(-MAX_SEQ_LEN // 6768)
        owned = [-(-length // 6768) for length in lengths]
        first = [sum(owned[:index]) for index in range(len(owned))]
        table = torch.tensor(
            [
                [start + min(page, count - 1) for page in range(width)]
                for start, count in zip(first, owned)
            ],
            device=q.device,
            dtype=torch.int32,
        )
        slots = torch.cat(
            [
                torch.arange(length, device=q.device, dtype=torch.int64)
                + start * 6768
                for start, length in zip(first, lengths)
            ]
        )
        dtype = torch.uint8 if self.fp8 else torch.bfloat16
        cache = torch.zeros(sum(owned), 2, 6768, 256, dtype=dtype, device=q.device)
        key_cache, value_cache = cache.transpose(1, 2).split(128, dim=-1)
        if self.fp8:
            # Serving rounds the KV scales to BF16 at load.
            ks = self.k_scale.to(torch.bfloat16).float()
            vs = self.v_scale.to(torch.bfloat16).float()
            qs = torch.ones_like(ks)
            query, _ = ops.scaled_fp8_quant(q.flatten(1), qs)
            query = query.view(q.shape)
            reshape_and_cache_flash(k, v, key_cache, value_cache, slots, "fp8_e4m3", ks, vs)
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            scale_shape = (1, 2)
            descales = dict(
                q_descale=qs.expand(scale_shape),
                k_descale=ks.expand(scale_shape),
                v_descale=vs.expand(scale_shape),
            )
        else:
            ones = torch.ones(1, dtype=torch.float32, device=q.device)
            reshape_and_cache_flash(k, v, key_cache, value_cache, slots, "auto", ones, ones)
            query, descales = q, {}
        # Serving's chunked prefill: each query chunk reads the cached keys up
        # to its last token. The fixed split-KV schedule makes every token's
        # output independent of the chunking, and the split scratch
        # (NUM_SPLITS x chunk x heads x dim, FP32) stays bounded.
        output = torch.empty_like(q)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            for lo in range(start, end, QUERY_CHUNK_TOKENS):
                hi = min(lo + QUERY_CHUNK_TOKENS, end)
                cu_q = torch.tensor([0, hi - lo], device=q.device, dtype=torch.int32)
                used_k = torch.tensor([hi - start], device=q.device, dtype=torch.int32)
                _flash_attn_fwd(
                    query[lo:hi],
                    key_cache,
                    value_cache,
                    cu_seqlens_q=cu_q,
                    seqused_k=used_k,
                    max_seqlen_q=MAX_SEQ_LEN,
                    max_seqlen_k=MAX_SEQ_LEN,
                    page_table=table[index : index + 1],
                    softmax_scale=self.scale,
                    causal=True,
                    **descales,
                    tile_mn=(128, 128),
                    pack_gqa=True,
                    num_splits=NUM_SPLITS,
                    seqlen_k_per_split=SEQLEN_K_PER_SPLIT,
                    disable_scheduler_metadata=True,
                    out=output[lo:hi],
                )
        if not self.fp8:
            return output, None, q, k, v
        saved = cache[slots // 6768, :, slots % 6768]
        saved_k, saved_v = saved.view(torch.float8_e4m3fn).split(128, -1)
        k_ref = (saved_k.float() * ks).to(k.dtype)
        v_ref = (saved_v.float() * vs).to(v.dtype)
        q_ref = (query.float() * qs).to(q.dtype)
        return output, None, q_ref, k_ref, v_ref

    def forward(self, q, k, v, meta):
        meta.validate_tokens(q.shape[0])
        hq, hkv, dim = self.heads
        if (
            q.shape[1:] != (hq, dim)
            or k.shape != v.shape
            or k.shape != (q.shape[0], hkv, dim)
        ):
            raise ValueError("Attention geometry does not match the module")
        device = self.k_scale.device if self.fp8 else q.device
        if any(
            x.dtype != torch.bfloat16 or not x.is_cuda or x.device != device
            for x in (q, k, v)
        ):
            raise ValueError("Expected CUDA BF16 QKV on the scale device")
        if not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, k, v)):
            return self._visible(q, k, v, meta.boundaries)[0]
        return _AttentionVJP.apply(q, k, v, self, meta.boundaries)


class Attention(nn.Module):
    """Nemotron's non-rotary attention; query width is independent of hidden size.

    ``fp8_kv_scales`` is the checkpoint's (k_scale, v_scale). The projection
    factory's activation precision picks the static FP8 KV cache (release) or
    a BF16 one (the BF16 activation mode), which keeps the scales for export.
    """

    def __init__(
        self,
        config,
        ps,
        *,
        device=None,
        dtype=torch.bfloat16,
        fp8_kv_scales,
        projection_factory,
        hf_prefix,
    ):
        super().__init__()
        if config.attention_bias:
            raise ValueError("Expected bias-free Nemotron attention")
        self.head_dim = config.head_dim
        scales = [] if fp8_kv_scales is None else fp8_kv_scales
        self.kv_attention = Fa4KVAttention(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            *(scale.to(device=device) for scale in scales),
            fp8=getattr(projection_factory, "fp8_activations", True),
        )
        for name, heads in (
            ("q_proj", config.num_attention_heads),
            ("k_proj", config.num_key_value_heads),
            ("v_proj", config.num_key_value_heads),
        ):
            setattr(
                self,
                name,
                projection_factory(
                    f"{hf_prefix}.{name}",
                    config.hidden_size,
                    heads * config.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )
        self.o_proj = projection_factory(
            f"{hf_prefix}.o_proj",
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, meta: SSMMeta):
        q, k, v = (
            projection(x, proj).view(x.shape[0], -1, self.head_dim)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        output = self.kv_attention(q, k, v, meta)
        return projection(output.flatten(1), self.o_proj)
