"""Nemotron-H MTP block (``mtp.layers.{0,1}``), trained on detached inputs.

vLLM's ``NemotronHMTP`` drafts with it: ``eh_proj([enorm(embed(t[i+1])),
hnorm(h[i])])``, an attention layer and an MoE layer (pattern ``*E``), then
``final_layernorm`` and the target's LM head predict ``t[i+2]``. ``h`` is the
target's last hidden state after ``norm_f``.

The block only proposes tokens, so it trains on plain PyTorch BF16 ops with
ordinary autograd; it needs no batch invariance, and it shares no compiled
kernel (whose recompilation would change the policy's numerics) with the
policy's layers. Its inputs (``h``, the
embedding, the LM head weight) are detached, so its loss reaches only the
``mtp.*`` parameters. The routed experts are BF16 and expert parallel: every
EP rank holds ``n_routed_experts / ep`` of them, sees the tokens of the whole
EP group (all-gather) and returns each rank its partial outputs
(reduce-scatter).
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from megatron.lite.model.nemotron_h.vllm.primitive.attention.module import Fa4KVAttention

MTP_PREFIX = "mtp.layers"
# Tokens per LM-head chunk of the MTP loss.
MTP_LOSS_CHUNK = 8192


def _gather_sizes(n, group):
    sizes = torch.tensor([n], device="cuda", dtype=torch.int64)
    out = [torch.empty_like(sizes) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, sizes, group=group)
    return [int(s.item()) for s in out]


def _all_gather_rows(x, sizes, group):
    width = max(sizes)
    padded = x.new_zeros((width, *x.shape[1:]))
    padded[: x.shape[0]] = x
    out = x.new_empty((len(sizes) * width, *x.shape[1:]))
    dist.all_gather_into_tensor(out, padded.contiguous(), group=group)
    return torch.cat([out[r * width : r * width + n] for r, n in enumerate(sizes)])


def _reduce_scatter_rows(y, sizes, rank, group):
    width = max(sizes)
    padded = y.new_zeros((len(sizes) * width, *y.shape[1:]))
    start = 0
    for r, n in enumerate(sizes):
        padded[r * width : r * width + n] = y[start : start + n]
        start += n
    out = y.new_empty((width, *y.shape[1:]))
    dist.reduce_scatter_tensor(out, padded, group=group)
    return out[: sizes[rank]]


class _GatherTokens(torch.autograd.Function):
    """EP all-gather of token rows; backward sums the row gradients back."""

    @staticmethod
    def forward(ctx, x, sizes, rank, group):
        ctx.sizes, ctx.rank, ctx.group = sizes, rank, group
        return _all_gather_rows(x, sizes, group)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter_rows(grad.contiguous(), ctx.sizes, ctx.rank, ctx.group), None, None, None


class _ScatterSum(torch.autograd.Function):
    """Sum every rank's partial outputs for this rank's tokens."""

    @staticmethod
    def forward(ctx, y, sizes, rank, group):
        ctx.sizes, ctx.group = sizes, group
        return _reduce_scatter_rows(y.contiguous(), sizes, rank, group)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather_rows(grad.contiguous(), ctx.sizes, ctx.group), None, None, None


class RMSNorm(nn.Module):
    """vLLM's RMSNorm rounding (FP32 statistics, residual added in FP32)."""

    def __init__(self, width, eps, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.eps = eps

    def _norm(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(self.weight.dtype) * self.weight

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x.float())
        total = x.float() + residual.float()
        return self._norm(total), total.to(x.dtype)


def projection(x, module):
    return torch.nn.functional.linear(x, module.weight)


class SharedExperts(nn.Module):
    def __init__(self, config, *, projection_factory, hf_prefix, device=None, dtype=torch.bfloat16):
        super().__init__()
        size = config.moe_shared_expert_intermediate_size
        self.up_proj = projection_factory(
            f"{hf_prefix}.up_proj", config.hidden_size, size, device=device, dtype=dtype
        )
        self.down_proj = projection_factory(
            f"{hf_prefix}.down_proj", size, config.hidden_size, device=device, dtype=dtype
        )

    def forward(self, x):
        return projection(torch.nn.functional.relu(projection(x, self.up_proj)).square(), self.down_proj)


class MTPRouter(nn.Module):
    """Sigmoid top-k with the score-correction bias, renormalized (vLLM's
    grouped top-k with one group). No router replay: the rollout returns
    routes for the target's MoE layers only."""

    def __init__(self, config, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size, device=device, dtype=dtype)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.n_routed_experts, device=device, dtype=torch.float32),
        )

    def forward(self, x):
        scores = torch.nn.functional.linear(x.float(), self.weight.float()).sigmoid()
        with torch.no_grad():
            ids = (scores + self.e_score_correction_bias).topk(
                self.config.num_experts_per_tok, dim=-1
            ).indices
        weights = scores.gather(-1, ids)
        if self.config.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        return ids, weights


class Bf16RoutedExperts(nn.Module):
    """This EP rank's BF16 routed experts, ``[local, out, in]`` per projection."""

    def __init__(self, config, ps, *, hf_prefix, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = config.n_routed_experts
        if self.num_experts % ps.ep_size:
            raise ValueError("EP must divide the MTP routed experts")
        self.num_local = self.num_experts // ps.ep_size
        self.offset = ps.ep_rank * self.num_local
        self.prefix = hf_prefix
        self.ep_size, self.ep_rank = ps.ep_size, ps.ep_rank
        self.ep_group = ps.ep_group if ps.ep_size > 1 else None
        hidden, inter = config.hidden_size, config.moe_intermediate_size
        self.up_proj = nn.Parameter(
            torch.empty(self.num_local, inter, hidden, device=device, dtype=dtype)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_local, hidden, inter, device=device, dtype=dtype)
        )

    def hf_names(self, expert, *, local=False):
        index = expert if local else self.offset + expert
        return {
            p: f"{self.prefix}.{index}.{p}.weight" for p in ("up_proj", "down_proj")
        }

    def forward(self, x, ids, weights):
        sizes = None
        if self.ep_group is not None:
            sizes = _gather_sizes(x.shape[0], self.ep_group)
            x = _GatherTokens.apply(x, sizes, self.ep_rank, self.ep_group)
            ids = _all_gather_rows(ids, sizes, self.ep_group)
            weights = _GatherTokens.apply(weights, sizes, self.ep_rank, self.ep_group)
        out = x.new_zeros(x.shape, dtype=torch.float32)
        local = ids - self.offset
        for expert in range(self.num_local):
            rows, slots = (local == expert).nonzero(as_tuple=True)
            if rows.numel() == 0:
                # Keep every local expert in the graph so FSDP sees its gradient.
                out = out + 0.0 * (self.up_proj[expert].sum() + self.down_proj[expert].sum())
                continue
            h = torch.nn.functional.relu(x[rows] @ self.up_proj[expert].T).square()
            y = (h @ self.down_proj[expert].T).float() * weights[rows, slots, None]
            out = out.index_add(0, rows, y)
        out = out.to(x.dtype)
        if self.ep_group is not None:
            out = _ScatterSum.apply(out, sizes, self.ep_rank, self.ep_group)
        return out


class MTPMoE(nn.Module):
    def __init__(self, config, ps, *, projection_factory, hf_prefix, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.gate = MTPRouter(config, device=device, dtype=dtype)
        self.experts = Bf16RoutedExperts(
            config, ps, hf_prefix=f"{hf_prefix}.experts", device=device, dtype=dtype
        )
        self.shared_experts = SharedExperts(
            config,
            device=device,
            dtype=dtype,
            projection_factory=projection_factory,
            hf_prefix=f"{hf_prefix}.shared_experts",
        )
        self.scale = config.routed_scaling_factor

    def forward(self, x):
        ids, weights = self.gate(x)
        return self.shared_experts(x) + self.experts(x, ids, weights) * self.scale


class MTPAttention(nn.Module):
    """Non-rotary attention over a BF16 KV cache (the drafter's own layer)."""

    def __init__(self, config, *, projection_factory, hf_prefix, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.head_dim = config.head_dim
        self.kv_attention = Fa4KVAttention(
            config.num_attention_heads, config.num_key_value_heads, config.head_dim, fp8=False
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
                    f"{hf_prefix}.{name}", config.hidden_size, heads * config.head_dim,
                    device=device, dtype=dtype,
                ),
            )
        self.o_proj = projection_factory(
            f"{hf_prefix}.o_proj", config.num_attention_heads * config.head_dim,
            config.hidden_size, device=device, dtype=dtype,
        )

    def forward(self, x, meta):
        q, k, v = (
            projection(x, proj).view(x.shape[0], -1, self.head_dim)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        return projection(self.kv_attention(q, k, v, meta).flatten(1), self.o_proj)


class MTPLayer(nn.Module):
    def __init__(self, config, ps, index, kind, *, projection_factory, device=None, dtype=torch.bfloat16):
        super().__init__()
        prefix = f"{MTP_PREFIX}.{index}"
        eps = config.layer_norm_epsilon
        last = index == len(config.mtp_layers_block_type) - 1
        if index == 0:
            self.enorm = RMSNorm(config.hidden_size, eps, device=device, dtype=dtype)
            self.hnorm = RMSNorm(config.hidden_size, eps, device=device, dtype=dtype)
            self.eh_proj = projection_factory(
                f"{prefix}.eh_proj", 2 * config.hidden_size, config.hidden_size,
                device=device, dtype=dtype,
            )
        self.norm = RMSNorm(config.hidden_size, eps, device=device, dtype=dtype)
        self.kind = kind
        if kind == "attention":
            self.mixer = MTPAttention(
                config, projection_factory=projection_factory, hf_prefix=f"{prefix}.mixer",
                device=device, dtype=dtype,
            )
        elif kind == "moe":
            self.mixer = MTPMoE(
                config, ps, projection_factory=projection_factory, hf_prefix=f"{prefix}.mixer",
                device=device, dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported Nemotron MTP block: {kind}")
        if last:
            self.final_layernorm = RMSNorm(config.hidden_size, eps, device=device, dtype=dtype)

    def forward(self, hidden, residual, meta, embed=None):
        if embed is not None:
            hidden = projection(
                torch.cat((self.enorm(embed), self.hnorm(hidden)), dim=-1), self.eh_proj
            )
        if residual is None:
            residual, hidden = hidden, self.norm(hidden)
        else:
            hidden, residual = self.norm(hidden, residual)
        hidden = self.mixer(hidden, meta) if self.kind == "attention" else self.mixer(hidden)
        if hasattr(self, "final_layernorm"):
            hidden, residual = self.final_layernorm(hidden + residual), None
        return hidden, residual


class NemotronHMTP(nn.Module):
    """``mtp.layers.*``; one FSDP2 unit, outside the policy's units."""

    def __init__(self, config, ps, *, projection_factory, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(i): MTPLayer(
                    config, ps, i, kind, projection_factory=projection_factory,
                    device=device, dtype=dtype,
                )
                for i, kind in enumerate(config.mtp_layers_block_type)
            }
        )

    def forward(self, hidden, embed, meta):
        residual = None
        for index, layer in enumerate(self.layers.values()):
            hidden, residual = layer(hidden, residual, meta, embed=embed if index == 0 else None)
        return hidden


class _FrozenHead(torch.autograd.Function):
    """The serving LM head on a constant weight: gradient for the input only."""

    @staticmethod
    def forward(ctx, x, visible, weight):
        ctx.save_for_backward(weight)
        return visible(x)

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return (grad.to(weight.dtype) @ weight), None, None


def _chunk_nll(hidden, visible, weight, ids):
    logits = _FrozenHead.apply(hidden, visible, weight).float()
    nll = torch.logsumexp(logits, -1) - logits.gather(-1, ids[:, None]).squeeze(-1)
    with torch.no_grad():
        hit = logits.argmax(-1) == ids
    return nll, hit


def mtp_token_nll(hidden, lm_head, ids):
    """Per-token NLL of ``ids`` under the frozen LM head, and the top-1 hits,
    one checkpointed ``MTP_LOSS_CHUNK`` at a time (logits are recomputed in
    backward instead of kept)."""
    weight = lm_head.weight.detach()
    visible = lm_head._inference if hasattr(lm_head, "_inference") else (
        lambda x: torch.nn.functional.linear(x, weight)
    )
    nll, hits = [], []
    for start in range(0, hidden.shape[0], MTP_LOSS_CHUNK):
        chunk = hidden[start : start + MTP_LOSS_CHUNK]
        ids_chunk = ids[start : start + MTP_LOSS_CHUNK]
        n, h = checkpoint(_chunk_nll, chunk, visible, weight, ids_chunk, use_reentrant=False)
        nll.append(n)
        hits.append(h)
    return torch.cat(nll), torch.cat(hits)


def roll_packed_left(tensor, boundaries, fill=0):
    """``tensor[j] = tensor[j + 1]`` inside every packed sequence; the last
    position of each sequence takes ``fill``."""
    rolled = torch.roll(tensor, shifts=-1, dims=0)
    ends = torch.tensor(boundaries[1:], device=tensor.device, dtype=torch.long) - 1
    rolled[ends] = fill
    return rolled
