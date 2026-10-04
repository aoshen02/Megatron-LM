"""Nemotron-H attention over the checkpoint's static FP8 KV cache."""

import torch
from torch import nn

from megatron.lite.model.nemotron_h.vllm.primitive.attention.backward import (
    _Fp8AttentionVJP,
)
from megatron.lite.model.nemotron_h.vllm.primitive.dense import (
    projection,
    projection_layer,
    scaled_fp8_quant,
)
from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import SSMMeta


class Fa4Fp8KVAttention(torch.nn.Module):
    """Serving FA4 over the static FP8 KV cache, with a FlashAttention VJP.

    Replays ``vllm.model_executor.models.nemotron_h_fa4``: FP8 query and
    KV-cache quantization with fixed scales, serving's 6768-token pages and
    the fixed split-KV schedule. The VJP is the FlashAttention varlen
    backward on the dequantized Q/K/V with the visible output and LSE (an
    identity straight-through estimator for the quantization). Supports TP1
    causal packed sequences starting at zero, without prefix sharing or
    sliding windows.
    """

    def __init__(self, num_heads, num_kv_heads, head_dim, k_scale, v_scale):
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
        self.scale = head_dim**-0.5
        self.heads = (num_heads, num_kv_heads, head_dim)
        if self.heads != (32, 2, 128):
            raise ValueError("Aligned FA4 requires Nemotron Q32/KV2/D128")

    def _visible(self, q, k, v, boundaries):
        """Return the serving output, its LSE and the dequantized Q/K/V."""
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
        # Serving's page geometry (6768-token pages, a table row wide enough
        # for MAX_SEQ_LEN); each sequence owns only the pages it fills, and
        # the rest of its row repeats its last page, which is never read.
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
        lengths = torch.tensor(lengths, device=q.device, dtype=torch.int32)
        cu = torch.tensor(boundaries, device=q.device, dtype=torch.int32)
        cache = torch.zeros(
            sum(owned), 2, 6768, 256, dtype=torch.uint8, device=q.device
        )
        # Serving loads scale parameters under the BF16 model default dtype,
        # then copies their rounded values into FP32 runtime buffers. Keep the
        # original checkpoint buffers unchanged for export.
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
        _, lse, *_ = _flash_attn_fwd(
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
            return_lse=True,
        )
        saved = cache[slots // 6768, :, slots % 6768]
        saved_k, saved_v = saved.view(torch.float8_e4m3fn).split(128, -1)
        k_ref = (saved_k.float() * ks).to(k.dtype)
        v_ref = (saved_v.float() * vs).to(v.dtype)
        q_ref = (quantized_q.float() * qs).to(q.dtype)
        return output, lse, q_ref, k_ref, v_ref

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
        if not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, k, v)):
            return self._visible(q, k, v, meta.boundaries)[0]
        return _Fp8AttentionVJP.apply(q, k, v, self, meta.boundaries)


class Attention(nn.Module):
    """Nemotron's non-rotary attention; query width is independent of hidden size."""

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
        if ps.tp_size != 1:
            raise ValueError("Nemotron attention currently requires TP1")
        if config.attention_bias:
            raise ValueError("Expected bias-free Nemotron attention")
        if ps.cp_size != 1:
            raise ValueError("FP8 KV training attention currently requires CP1")
        self.head_dim = config.head_dim
        k_scale, v_scale = fp8_kv_scales
        self.kv_attention = Fa4Fp8KVAttention(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            k_scale.to(device=device),
            v_scale.to(device=device),
        )
        for name, heads in (
            ("q_proj", config.num_attention_heads),
            ("k_proj", config.num_key_value_heads),
            ("v_proj", config.num_key_value_heads),
        ):
            setattr(
                self,
                name,
                projection_layer(
                    projection_factory,
                    f"{hf_prefix}.{name}",
                    config.hidden_size,
                    heads * config.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )
        self.o_proj = projection_layer(
            projection_factory,
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
