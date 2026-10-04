"""Packed Mamba2 mixer with explicit request boundaries."""

from dataclasses import dataclass

import torch

from megatron.lite.model.nemotron_h.vllm.primitive.dense import (
    GatedRMSNorm,
    projection,
)
from megatron.lite.model.nemotron_h.vllm.primitive.mamba.backward import (
    _PackedConvVJP,
    _PackedScanVJP,
)


@dataclass(frozen=True)
class SSMMeta:
    """Global packed request boundaries; every request starts from zero SSM state."""

    boundaries: tuple[int, ...]
    chunk_size: int = 128

    def __post_init__(self):
        if (
            self.chunk_size <= 0
            or len(self.boundaries) < 2
            or self.boundaries[0] != 0
            or any(a >= b for a, b in zip(self.boundaries, self.boundaries[1:]))
        ):
            raise ValueError("SSM boundaries must start at zero and strictly increase")

    def chunks(self):
        starts, last, sequence_ids = [], [], []
        for sequence, (start, end) in enumerate(
            zip(self.boundaries, self.boundaries[1:])
        ):
            offsets = list(range(start, end, self.chunk_size))
            starts.extend(offsets)
            last.append(len(starts) - 1)
            sequence_ids.extend([sequence] * len(offsets))
        return (*starts, self.boundaries[-1]), tuple(last), tuple(sequence_ids)

    def validate_tokens(self, tokens):
        if tokens != self.boundaries[-1]:
            raise ValueError("Packed SSM token count disagrees with request boundaries")

    def seq_idx(self, device):
        """Per-token request index [1, T], as MCore ``PackedSeqParams.seq_idx``."""
        lengths = [b - a for a, b in zip(self.boundaries, self.boundaries[1:])]
        return torch.repeat_interleave(
            torch.arange(len(lengths), dtype=torch.int32, device=device),
            torch.tensor(lengths, device=device),
            output_size=self.boundaries[-1],
        )[None]


def packed_conv(x, weight, bias, meta: SSMMeta):
    """Causal SiLU convolution, x[T,C], weight[C,K]; no cross-request history."""
    import causal_conv1d.cpp_functions  # noqa: F401

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    meta.validate_tokens(x.shape[0])

    def visible(x, weight, bias):
        sequences = len(meta.boundaries) - 1
        states = x.new_zeros(sequences + 1, x.shape[1], weight.shape[1] - 1)
        return causal_conv1d_fn(
            x.T,
            weight,
            bias,
            states,
            torch.tensor(meta.boundaries, dtype=torch.int32, device=x.device),
            cache_indices=torch.arange(
                1, sequences + 1, dtype=torch.int32, device=x.device
            ),
            has_initial_state=torch.zeros(sequences, dtype=torch.bool, device=x.device),
            activation="silu",
        ).T

    if not torch.is_grad_enabled():
        return visible(x, weight, bias)
    return _PackedConvVJP.apply(visible, x, weight, bias, meta.seq_idx(x.device))


def packed_scan(x, dt, A, B, C, D, dt_bias, meta: SSMMeta):
    """Cache-free SSD, x[T,H,P], B/C[T,G,N]; inference forward, mamba_ssm VJP."""
    import mamba_ssm.ops.triton.ssd_combined  # noqa: F401

    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )

    meta.validate_tokens(x.shape[0])

    def visible(x, dt, A, B, C, D, dt_bias):
        chunks, last, sequence_ids = meta.chunks()

        def i32(values):
            return torch.tensor(values, dtype=torch.int32, device=x.device)

        output = torch.empty_like(x)
        mamba_chunk_scan_combined_varlen(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=meta.chunk_size,
            cu_seqlens=i32(meta.boundaries),
            cu_chunk_seqlens=i32(chunks),
            last_chunk_indices=i32(last),
            seq_idx=i32(sequence_ids),
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=output,
            state_dtype=torch.float32,
        )
        return output

    if not torch.is_grad_enabled():
        return visible(x, dt, A, B, C, D, dt_bias)
    return _PackedScanVJP.apply(
        visible, x, dt, A, B, C, D, dt_bias, meta.seq_idx(x.device), meta.chunk_size
    )


class MambaMixer(torch.nn.Module):
    """Native TP1 Mamba2 mixer; weights retain HF names for lossless checkpoint IO."""

    def __init__(
        self,
        config,
        parallel_state,
        *,
        device=None,
        dtype=torch.bfloat16,
        projection_factory,
        hf_prefix,
    ):
        super().__init__()
        if config.mamba_hidden_act != "silu":
            raise ValueError("Mamba visible convolution requires SiLU")
        self.config = config
        factory = dict(device=device, dtype=dtype)
        self.in_proj = projection_factory(
            f"{hf_prefix}.in_proj",
            config.hidden_size,
            config.mamba_in_proj_size,
            bias=config.use_bias,
            **factory,
        )
        self.conv1d = torch.nn.Conv1d(
            config.mamba_conv_dim,
            config.mamba_conv_dim,
            config.conv_kernel,
            groups=config.mamba_conv_dim,
            bias=config.use_conv_bias,
            **factory,
        )
        self.dt_bias = torch.nn.Parameter(
            torch.empty(config.mamba_num_heads, **factory)
        )
        self.A_log = torch.nn.Parameter(torch.empty(config.mamba_num_heads, **factory))
        self.D = torch.nn.Parameter(torch.empty(config.mamba_num_heads, **factory))
        self.norm = GatedRMSNorm(
            config.mamba_inner_size,
            config.mamba_inner_size // config.n_groups,
            config.layer_norm_epsilon,
            **factory,
        )
        self.out_proj = projection_factory(
            f"{hf_prefix}.out_proj",
            config.mamba_inner_size,
            config.hidden_size,
            bias=config.use_bias,
            **factory,
        )

    def forward(self, hidden, meta: SSMMeta):
        c = self.config
        meta.validate_tokens(hidden.shape[0])
        if meta.chunk_size != c.chunk_size:
            raise ValueError("SSM metadata chunk size differs from model config")
        projected = projection(hidden, self.in_proj)
        gate, xbc, dt = projected.split(
            (c.mamba_inner_size, c.mamba_conv_dim, c.mamba_num_heads), dim=-1
        )
        sections = [
            c.mamba_inner_size,
            c.n_groups * c.ssm_state_size,
            c.n_groups * c.ssm_state_size,
        ]
        conv = packed_conv(
            xbc.contiguous(), self.conv1d.weight[:, 0], self.conv1d.bias, meta
        )
        x, B, C = conv.split(sections, dim=-1)
        tokens = conv.shape[0]
        scanned = packed_scan(
            x.reshape(tokens, c.mamba_num_heads, c.mamba_head_dim),
            dt,
            -torch.exp(self.A_log.float()),
            B.reshape(tokens, c.n_groups, c.ssm_state_size),
            C.reshape(tokens, c.n_groups, c.ssm_state_size),
            self.D,
            self.dt_bias,
            meta,
        )
        normalized = self.norm(
            scanned.reshape(hidden.shape[0], c.mamba_inner_size), gate
        )
        return projection(normalized, self.out_proj)
