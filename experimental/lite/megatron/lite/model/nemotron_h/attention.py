"""Cache-free packed attention with native mlite sequence/head exchange."""

import torch
from torch import nn

from .functional import projection, visible_forward
from .mamba import SSMMeta, exchange_sequence_channels
from .quantization import projection_layer


def packed_attention(q, k, v, meta: SSMMeta, *, scale):
    """THD causal GQA with linear-memory fused forward and VJP."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    meta.validate_tokens(q.shape[0])
    if k.shape != v.shape or k.shape[0] != q.shape[0] or q.shape[1] % k.shape[1]:
        raise ValueError("Packed GQA requires matching KV and divisible query heads")
    if q.shape[2] != k.shape[2]:
        raise ValueError("Query and key head dimensions must match")
    cu = torch.tensor(meta.boundaries, device=q.device, dtype=torch.int32)
    max_length = max(b - a for a, b in zip(meta.boundaries, meta.boundaries[1:]))

    def visible(q, k, v):
        return flash_attn_varlen_func(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_length,
            max_seqlen_k=max_length,
            softmax_scale=scale,
            causal=True,
            fa_version=2,
            num_splits=1,
        )

    def native(q, k, v):
        outputs = []
        for a, b in zip(meta.boundaries, meta.boundaries[1:]):
            query, key, value = (x[a:b].transpose(0, 1)[None] for x in (q, k, v))
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=scale,
                enable_gqa=q.shape[1] != k.shape[1],
            )
            outputs.append(output.transpose(1, 2).squeeze(0))
        return torch.cat(outputs)

    return visible_forward(visible, native, q, k, v)


class Attention(nn.Module):
    """Nemotron's non-rotary attention; query width is independent of hidden size."""

    def __init__(
        self,
        config,
        ps,
        *,
        device=None,
        dtype=torch.bfloat16,
        fp8_kv_scales=None,
        projection_factory=None,
        hf_prefix=None,
    ):
        super().__init__()
        if ps.tp_size != 1:
            raise ValueError("Nemotron attention currently requires TP1")
        if config.attention_bias:
            raise ValueError("Expected bias-free Nemotron attention")
        if config.num_key_value_heads % ps.cp_size:
            raise ValueError("CP must divide KV heads")
        if fp8_kv_scales is not None and ps.cp_size != 1:
            raise ValueError("FP8 KV training attention currently requires CP1")
        self.cp_group = ps.cp_group if ps.cp_size > 1 else None
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.kv_attention = None
        if fp8_kv_scales is not None:
            import vllm.envs as envs

            from .fp8_attention import Fa4Fp8KVAttention, Fp8KVAttention

            k_scale, v_scale = fp8_kv_scales
            attention_class = (
                Fa4Fp8KVAttention if envs.VLLM_BATCH_INVARIANT else Fp8KVAttention
            )
            self.kv_attention = attention_class(
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
                    f"{hf_prefix}.{name}" if hf_prefix else None,
                    config.hidden_size,
                    heads * config.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )
        self.o_proj = projection_layer(
            projection_factory,
            f"{hf_prefix}.o_proj" if hf_prefix else None,
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, meta: SSMMeta):
        q, k, v = (
            exchange_sequence_channels(
                projection(x, proj).view(x.shape[0], -1, self.head_dim),
                self.cp_group,
            )
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        if self.kv_attention is None:
            output = packed_attention(q, k, v, meta, scale=self.scale)
        else:
            output = self.kv_attention(q, k, v, meta)
        output = exchange_sequence_channels(output, self.cp_group, reverse=True)
        return projection(output.flatten(1), self.o_proj)
