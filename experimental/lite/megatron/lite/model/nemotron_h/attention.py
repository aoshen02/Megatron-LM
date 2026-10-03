"""Nemotron-H attention over the checkpoint's static FP8 KV cache."""

import torch
from torch import nn

from .functional import projection
from .mamba import SSMMeta
from .quantization import projection_layer


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
        import vllm.envs as envs

        from .fp8_attention import Fa4Fp8KVAttention, Fp8KVAttention

        if ps.tp_size != 1:
            raise ValueError("Nemotron attention currently requires TP1")
        if config.attention_bias:
            raise ValueError("Expected bias-free Nemotron attention")
        if ps.cp_size != 1:
            raise ValueError("FP8 KV training attention currently requires CP1")
        self.head_dim = config.head_dim
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
