"""Packed Mamba2 mixer with explicit request boundaries."""

from dataclasses import dataclass, field, replace

import torch

from megatron.lite.model.nemotron_h.vllm.primitive.cp import (
    full_sequence_to_heads,
    heads_to_full_sequence,
    headwise,
)
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
    """Global packed request boundaries; every request starts from zero SSM state.

    With ``cp_size > 1`` this rank holds the contiguous token shard
    ``local_range()`` of the packed sequence.
    """

    boundaries: tuple[int, ...]
    chunk_size: int = 128
    cp_size: int = 1
    cp_rank: int = 0
    cp_group: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if (
            self.chunk_size <= 0
            or len(self.boundaries) < 2
            or self.boundaries[0] != 0
            or any(a >= b for a, b in zip(self.boundaries, self.boundaries[1:]))
        ):
            raise ValueError("SSM boundaries must start at zero and strictly increase")
        if not 0 <= self.cp_rank < self.cp_size or self.boundaries[-1] % self.cp_size:
            raise ValueError("Packed tokens must split evenly over the CP ranks")
        if self.cp_size > 1 and self.cp_group is None:
            raise ValueError("CP>1 needs the context-parallel group")

    def full(self):
        """The whole packed sequence on one rank."""
        return replace(self, cp_size=1, cp_rank=0, cp_group=None)

    def local_range(self):
        size = self.boundaries[-1] // self.cp_size
        return self.cp_rank * size, (self.cp_rank + 1) * size

    def attention_spans(self):
        """The local queries of every request this shard touches, and the keys
        they attend: ``(query_bounds, key_bounds, key_start, key_end)``. Keys
        run from the request start to the last local query, so each request's
        queries are the last ``q`` of its ``k`` keys."""
        if self.cp_size == 1:
            return self.boundaries, self.boundaries, 0, self.boundaries[-1]
        lo, hi = self.local_range()
        spans = [
            (start, end)
            for start, end in zip(self.boundaries, self.boundaries[1:])
            if end > lo and start < hi
        ]
        key_start = spans[0][0]
        queries, keys = [0], [0]
        for start, end in spans:
            queries.append(queries[-1] + min(end, hi) - max(start, lo))
            keys.append(min(end, hi) - key_start)
        return tuple(queries), tuple(keys), key_start, hi

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
        if tokens != self.boundaries[-1] // self.cp_size:
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
        projected = projection(hidden, self.in_proj)
        gate, xbc, dt = projected.split(
            (c.mamba_inner_size, c.mamba_conv_dim, c.mamba_num_heads), dim=-1
        )
        sections = [
            c.mamba_inner_size,
            c.n_groups * c.ssm_state_size,
            c.n_groups * c.ssm_state_size,
        ]
        weight, bias = self.conv1d.weight[:, 0], self.conv1d.bias
        A_log, D, dt_bias = self.A_log, self.D, self.dt_bias
        heads, groups, core = c.mamba_num_heads, c.n_groups, meta
        if meta.cp_size > 1:
            # Head-parallel CP: this rank runs heads [cp_rank * H / cp, ...)
            # over the whole packed sequence, so the convolution and the
            # scan's chunks are the CP=1 ones.
            cp = meta.cp_size
            xbc_dt = heads_to_full_sequence(
                torch.cat((xbc, dt), dim=-1), [*sections, heads], meta
            )
            xbc, dt = xbc_dt.split((c.mamba_conv_dim // cp, heads // cp), dim=-1)
            weight, bias = (
                None if p is None else headwise(p, meta, sections)
                for p in (weight, bias)
            )
            A_log, D, dt_bias = (headwise(p, meta) for p in (A_log, D, dt_bias))
            sections = [width // cp for width in sections]
            heads, groups, core = heads // cp, groups // cp, meta.full()
        conv = packed_conv(xbc.contiguous(), weight, bias, core)
        x, B, C = conv.split(sections, dim=-1)
        tokens = conv.shape[0]
        scanned = packed_scan(
            x.reshape(tokens, heads, c.mamba_head_dim),
            dt,
            -torch.exp(A_log.float()),
            B.reshape(tokens, groups, c.ssm_state_size),
            C.reshape(tokens, groups, c.ssm_state_size),
            D,
            dt_bias,
            core,
        )
        scanned = scanned.reshape(tokens, heads * c.mamba_head_dim)
        if meta.cp_size > 1:
            scanned = full_sequence_to_heads(scanned, meta)
        normalized = self.norm(scanned, gate)
        return projection(normalized, self.out_proj)
