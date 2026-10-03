"""Nemotron native mlite protocol; no HF model wrapper or custom scheduler."""

from dataclasses import dataclass, field
from functools import partial

from megatron.lite.model.protocol_utils import nested_from_packed
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.loss import get_loss_context

from .checkpoint import load_hf_weights as _load_weights
from .checkpoint import refresh_quantized_projections
from .config import NemotronHConfig
from .mamba import SSMMeta


# Tokens per LM-head/log-probability chunk (DS4 default).
LOGPROB_CHUNK_SIZE = 8192


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    optimizer_config: OptimizerConfig | None = None
    deterministic: bool = True
    hf_path: str | None = None
    routed_forward_reduction: str | None = None

    def __post_init__(self):
        from .nvfp4_ep4 import validate_reduction

        validate_reduction(self.routed_forward_reduction)


def build_model_config(source, **overrides):
    config = (
        NemotronHConfig._from_hf_dict(source)
        if isinstance(source, dict)
        else NemotronHConfig.from_hf(source)
    )
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise ValueError(f"Unknown Nemotron config override: {name}")
        setattr(config, name, value)
    config.__post_init__()
    return config


def is_expert(name):
    return ".mixer.experts." in name


EXPERT_CLASSIFIER = is_expert


def PLACEMENT_FN(name):
    from torch.distributed.tensor import Replicate, Shard

    return [
        Replicate(),
        Replicate(),
        Shard(0) if is_expert(name) else Replicate(),
        Replicate(),
    ]


def _token_mean_loss(log_probs, local_mask, full_mask, cp_size):
    # Megatron averages dense and expert gradients over DP*CP.
    return -(log_probs * local_mask).sum() * cp_size / full_mask.sum().clamp_min(1)


def _base(module):
    while hasattr(module, "module"):
        module = module.module
    return module


def forward_step(model, batch):
    from .logprob import aligned_selected_log_probs

    ps = parallel_state_from_model(model)
    loss_mask = batch.loss_mask
    if loss_mask is None:
        loss_mask = batch.input_ids.new_ones(batch.input_ids.shape)
    packed = pack_nested_thd(
        nested_from_packed(batch.input_ids, batch.seq_lens),
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
        split_cp=False,
        labels=nested_from_packed(batch.labels, batch.seq_lens),
        loss_mask=nested_from_packed(loss_mask, batch.seq_lens),
        roll_labels=batch.labels is not None,
        roll_loss_mask=True,
    )

    def local(tensor):
        if tensor is None:
            return None
        return contiguous_slice_for_cp(
            tensor, ps.cp_rank, ps.cp_size, seq_dim=1
        ).reshape(-1)

    meta = SSMMeta(tuple(packed.cu_seqlens_padded.cpu().tolist()))
    labels = local(packed.labels)
    output = model(local(packed.input_ids), meta=meta, return_logits=labels is None)
    if not ps.pp_is_last:
        return {"hidden_states": output}
    if labels is None:
        return {"logits": output}
    context = get_loss_context()
    valid = labels >= 0
    calculate_entropy = context is not None and context.calculate_entropy
    log_probs, entropy = aligned_selected_log_probs(
        output,
        _base(model).lm_head,
        labels.clamp_min(0),
        1.0 if context is None else context.temperature,
        LOGPROB_CHUNK_SIZE,
        calculate_entropy=calculate_entropy,
        tp_group=ps.tp_group,
    )
    log_probs = log_probs.masked_fill(~valid, 0)
    mask = local(packed.loss_mask)
    mask = valid if mask is None else mask * valid
    full_mask = packed.labels >= 0
    if packed.loss_mask is not None:
        full_mask = full_mask * packed.loss_mask
    loss = _token_mean_loss(log_probs, mask, full_mask, ps.cp_size)
    result = {"log_probs": log_probs[None], "loss": loss}
    if calculate_entropy:
        result["entropy"] = entropy[None]
    return result


def unpack_forward_output(model, batch, output):
    ps = parallel_state_from_model(model)
    meta = thd_pack_meta(
        batch.seq_lens, tp_size=ps.tp_size, cp_size=ps.cp_size, cp_group=ps.cp_group
    )
    return unpack_thd_to_nested(output, meta, contiguous=True)


def build_model(model_cfg, *, impl_cfg):
    if model_cfg.quantization_config is None:
        raise ValueError(
            "Nemotron impl=vllm requires a ModelOpt MIXED_PRECISION checkpoint "
            "(NVFP4 experts/linears, FP8 Mamba projections and KV cache)"
        )
    from .quantized_proxy import build_quantized_proxy, validate_proxy_config

    validate_proxy_config(model_cfg, impl_cfg)
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    init_batch_invariance()
    ps = init_parallel(impl_cfg.parallel)
    count = model_cfg.num_hidden_layers
    start, end = (
        count * ps.pp_rank // ps.pp_size,
        count * (ps.pp_rank + 1) // ps.pp_size,
    )
    chunks = [build_quantized_proxy(model_cfg, impl_cfg, ps, layer_range=(start, end))]
    # Verify the checkpoint before an optimizer can bind these parameters.
    _load_weights(chunks[0], impl_cfg.hf_path)
    parameter_ids = {id(p) for chunk in chunks for p in chunk.parameters()}
    optimizer = finalize_grads = None
    if impl_cfg.optimizer == "dist_opt":
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_training_optimizer,
        )

        optimizer, finalize_grads = build_dist_opt_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=impl_cfg,
            ps=ps,
            model_name="nemotron_h",
            is_expert=is_expert,
            deterministic=impl_cfg.deterministic,
        )
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict

        attach_model_sharded_state_dict(
            chunks, ps, get_placements=PLACEMENT_FN, is_expert=is_expert
        )
        if parameter_ids != {id(p) for chunk in chunks for p in chunk.parameters()}:
            raise RuntimeError("Optimizer replaced quantized Parameter identities")
        # DDP may rebind storage; never replace the Parameter objects.
        refresh_quantized_projections(chunks)
        for chunk in chunks:
            current = chunk
            while hasattr(current, "module"):
                current = current.module
            _load_weights(current, impl_cfg.hf_path)
    elif impl_cfg.optimizer is not None:
        raise ValueError(
            "Native Nemotron uses dist_opt, not the historical FSDP adapter"
        )
    extras = {
        "model_cfg": model_cfg,
        "optimizer_backend": impl_cfg.optimizer or "none",
        "post_optimizer_step_hook": partial(_refresh_quantized, chunks),
    }
    if optimizer is not None:
        extras["post_model_load_hook"] = partial(_refresh_after_model_load, chunks)
    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=forward_step,
        extras=extras,
    )


def _refresh_quantized(chunks):
    # Post-optimizer hook: the checkpoint bytes are only valid for the initial
    # weights (DeepSeek-V4 invalidates its bound scales after an update too).
    refresh_quantized_projections(chunks, recompute_scales=True)


def _refresh_after_model_load(chunks):
    # The runtime's HF loader may rebind the masters after build_model;
    # reinstall the deployments before the first forward.
    refresh_quantized_projections(chunks)


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    while hasattr(chunk, "module"):
        chunk = chunk.module
    from pathlib import Path

    if str(Path(hf_path).resolve()) != chunk._quantized_proxy_root:
        raise ValueError("Quantized proxy loader must use its construction checkpoint")
    _load_weights(chunk, hf_path)


def vocab_size(model_cfg):
    return model_cfg.vocab_size


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as export

    from .checkpoint import NemotronExport

    if kwargs.get("export_dtype") is not None:
        # The deployment bytes (FP4/FP8 weights, FP32 scales) are exported
        # as stored; any cast would change what the rollout serves.
        raise ValueError("Nemotron exports deployment bytes; set export_dtype=None")
    yield from export(chunks, NemotronExport(model_cfg), ps, **kwargs)
