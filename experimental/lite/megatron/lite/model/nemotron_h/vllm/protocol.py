"""Nemotron-H vLLM-aligned mlite protocol (``impl="vllm"``)."""

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch

from megatron.lite.model.nemotron_h.checkpoint import (
    NemotronExport,
    is_expert,
    load_hf_weights as _load_weights,
    refresh_quantized_projections,
)
from megatron.lite.model.nemotron_h.config import NemotronHConfig
from megatron.lite.model.nemotron_h.vllm.model import NemotronHLayer, build_stage
from megatron.lite.model.nemotron_h.vllm.primitive.logprob import (
    aligned_selected_log_probs,
)
from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import SSMMeta
from megatron.lite.model.protocol_utils import nested_from_packed
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.loss import get_loss_context

# Tokens per LM-head/log-probability chunk (DS4 default).
LOGPROB_CHUNK_SIZE = 8192


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    optimizer_config: OptimizerConfig | None = None
    deterministic: bool = True
    hf_path: str | None = None
    # BF16 release hf_path was quantized from; theta0 deploys requant(master).
    bf16_master_path: str | None = None
    recompute: str | list[str] | None = None


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


EXPERT_CLASSIFIER = is_expert


def PLACEMENT_FN(name):
    from torch.distributed.tensor import Replicate, Shard

    return [
        Replicate(),
        Replicate(),
        Shard(0) if is_expert(name) else Replicate(),
        Replicate(),
    ]


def _post_optimizer_step(chunks, *, release_grads=False):
    # dist_opt overlaps the parameter all-gather with the next forward; the
    # requantization below needs every rank's updated shard now.
    for chunk in chunks:
        start_param_sync = getattr(chunk, "start_param_sync", None)
        if callable(start_param_sync):
            with torch.no_grad():
                start_param_sync(force_sync=True)
    refresh_quantized_projections(chunks, recompute_scales=True)
    if release_grads:
        # FSDP2 gradients are plain .grad tensors and the rollout runs next.
        for chunk in chunks:
            for parameter in chunk.parameters():
                parameter.grad = None


def _post_dist_opt_model_load(chunks):
    # The runtime's HF loader may rebind the masters after build_model;
    # reinstall the deployments before the first forward.
    refresh_quantized_projections(chunks)


def _validate_full_depth(config, source):
    if config != NemotronHConfig._from_hf_dict(source):
        raise ValueError("Full-depth configuration must match the checkpoint")
    pattern = "".join(
        {"linear_attention": "M", "moe": "E", "full_attention": "A"}.get(kind, "?")
        for kind in config.layers_block_type
    )
    if pattern != "MEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMEMAEMEMEMEME":
        raise ValueError("Expected the complete 52-layer Lightning pattern")
    if (
        config.vocab_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.mamba_num_heads,
        config.mamba_head_dim,
        config.n_groups,
        config.ssm_state_size,
        config.conv_kernel,
        config.chunk_size,
        config.n_shared_experts,
        config.moe_shared_expert_intermediate_size,
    ) != (131072, 32, 2, 128, 64, 64, 8, 128, 4, 128, 1, 3712):
        raise ValueError("Expected full Lightning attention/SSM/shared geometry")
    expected = {"lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}
    for index, kind in enumerate(config.layers_block_type):
        prefix = f"backbone.layers.{index}.mixer"
        if kind == "linear_attention":
            for projection in ("in_proj", "out_proj"):
                expected[f"{prefix}.{projection}"] = {"quant_algo": "FP8"}
        elif kind == "moe":
            for owner in [
                "shared_experts",
                *(f"experts.{expert}" for expert in range(config.n_routed_experts)),
            ]:
                for projection in ("up_proj", "down_proj"):
                    expected[f"{prefix}.{owner}.{projection}"] = {
                        "quant_algo": "W4A16_NVFP4",
                        "group_size": 16,
                    }
    # The authentic full checkpoint's MTP tensors are not quantized recipe entries.
    # Do not silently exclude unknown prefixes from the coverage contract.
    if config.quantization_config.get("quantized_layers") != expected:
        raise ValueError("Expected the exact full Lightning projection recipe")


def _validate_contract(config, impl):
    p = impl.parallel
    if not config.quantization_config:
        raise ValueError("Nemotron impl=vllm requires a ModelOpt MIXED_PRECISION checkpoint")
    if not impl.hf_path:
        raise ValueError("Quantized Nemotron requires an explicit hf_path")
    if impl.optimizer_config is None:
        raise ValueError("Quantized Nemotron requires explicit optimizer_config")
    if any(value != 1 for value in (p.tp, p.etp or 1, p.cp, p.vpp)):
        raise ValueError("Quantized Nemotron requires TP/ETP/CP/VPP1")
    if (p.pp, p.ep) not in ((1, 1), (4, 1), (1, 4)):
        raise ValueError("Quantized Nemotron supports PP1/EP1, PP4/EP1 or PP1/EP4")
    if (
        config.hidden_size,
        config.n_routed_experts,
        config.moe_intermediate_size,
        config.num_experts_per_tok,
    ) != (2688, 128, 1856, 6):
        raise ValueError("Quantized Nemotron requires Lightning operator geometry")
    recipe = config.quantization_config
    if recipe.get("quant_algo") != "MIXED_PRECISION" or recipe.get(
        "kv_cache_scheme"
    ) != {"dynamic": False, "num_bits": 8, "type": "float"}:
        raise ValueError("Expected checkpoint mixed precision and static FP8 KV")
    source = json.loads((Path(impl.hf_path) / "config.json").read_text())
    _validate_full_depth(config, source)
    if not impl.bf16_master_path:
        raise ValueError("Quantized Nemotron requires impl_cfg.bf16_master_path")
    master = json.loads((Path(impl.bf16_master_path) / "config.json").read_text())
    if "quantization_config" in master or any(
        master.get(key) != source.get(key)
        for key in ("num_hidden_layers", "layers_block_type", "hidden_size")
    ):
        raise ValueError("BF16 master source must be the unquantized same model")


def _check_runtime(ps):
    import vllm.envs as envs

    if not envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError("Quantized Nemotron requires BI=1 before startup")
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() != ps.pp_size * ps.ep_size
    ):
        raise RuntimeError("Torch world must be PP x EP")


@torch.no_grad()
def _check_router_gemm_rows_invariant():
    """The router's torch.mm(out_dtype=fp32) must not depend on the row count.

    That holds only if init_batch_invariance ran before this process's first
    cuBLAS call (the workspace is fixed then); check it in this process.
    """
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(8192, 2688, generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn(128, 2688, generator=g, device="cuda").to(torch.bfloat16)
    full = torch.mm(x, w.T, out_dtype=torch.float32)
    for rows in (1, 7, 64, 513):
        if not torch.equal(torch.mm(x[:rows], w.T, out_dtype=torch.float32), full[:rows]):
            raise RuntimeError(
                "Router GEMM depends on the row count: run init_batch_invariance "
                "before the process's first cuBLAS call"
            )


def _token_mean_loss(log_probs, local_mask, full_mask, cp_size):
    # Megatron averages dense and expert gradients over DP*CP.
    return -(log_probs * local_mask).sum() * cp_size / full_mask.sum().clamp_min(1)


def _forward_step(model, batch):
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
        unwrap_model(model).lm_head,
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


def _build_fsdp2(chunks, impl_cfg, ps):
    """DS4's deferred FSDP2 wrap after the HF load: FP32 shards, FP32
    parameters replicated, BF16 compute. The quantized layers requantize from
    the gathered BF16 matrix the unsharded forward sees."""
    from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

    optimizer = build_fsdp2_training_optimizer(
        chunks,
        impl_cfg.optimizer_config,
        ps,
        unit_modules=(NemotronHLayer,),
        expert_classifier=is_expert,
        replicated_param_classifier=lambda _name, param: param.dtype == torch.float32,
        deterministic=impl_cfg.deterministic,
        vpp=impl_cfg.parallel.vpp,
        leaf_module_names=(),
        use_fp32_shards=True,
        cast_forward_inputs=False,
    )
    for chunk in chunks:
        for module in chunk.modules():
            bind = getattr(module, "bind_master", None)
            if callable(bind):
                bind()
    refresh_quantized_projections(chunks)
    return {"optimizer": optimizer}


def build_model(model_cfg, *, impl_cfg):
    _validate_contract(model_cfg, impl_cfg)
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    init_batch_invariance()
    if impl_cfg.deterministic:
        # mamba_ssm's SSD backward reduces dA/dD/ddt_bias with atomics unless
        # its deterministic mode is on; set it here rather than rely on the
        # caller's torch.use_deterministic_algorithms.
        from mamba_ssm.utils.determinism import set_deterministic_mode

        set_deterministic_mode(True)
    ps = init_parallel(impl_cfg.parallel)
    _check_runtime(ps)
    _check_router_gemm_rows_invariant()
    count = model_cfg.num_hidden_layers
    start, end = (
        count * ps.pp_rank // ps.pp_size,
        count * (ps.pp_rank + 1) // ps.pp_size,
    )
    chunks = [build_stage(model_cfg, impl_cfg, ps, layer_range=(start, end))]
    recompute = parse_recompute_spec(impl_cfg.recompute)
    if recompute:
        if recompute != ["full"]:
            raise ValueError("Nemotron supports recompute='full' only")
        apply_recompute(list(chunks[0].layers.values()), recompute, {})
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
            _load_weights(unwrap_model(chunk), impl_cfg.hf_path)
    elif impl_cfg.optimizer not in (None, "fsdp2"):
        raise ValueError(f"Nemotron optimizers are dist_opt and fsdp2, not {impl_cfg.optimizer}")
    fsdp2 = impl_cfg.optimizer == "fsdp2"
    extras = {
        "model_cfg": model_cfg,
        "optimizer_backend": impl_cfg.optimizer or "none",
        "post_optimizer_step_hook": partial(_post_optimizer_step, chunks, release_grads=fsdp2),
    }
    if optimizer is not None:
        extras["post_model_load_hook"] = partial(_post_dist_opt_model_load, chunks)
    elif fsdp2:
        extras["post_model_load_hook"] = partial(_build_fsdp2, chunks, impl_cfg, ps)
    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step,
        extras=extras,
    )


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    chunk = unwrap_model(chunk)
    if str(Path(hf_path).resolve()) != chunk._hf_root:
        raise ValueError("Load from the checkpoint the model was constructed from")
    _load_weights(chunk, hf_path)


def vocab_size(model_cfg):
    return model_cfg.vocab_size


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as export

    if kwargs.get("export_dtype") is not None:
        # The deployment bytes (FP4/FP8 weights, FP32 scales) are exported
        # as stored; any cast would change what the rollout serves.
        raise ValueError("Nemotron exports deployment bytes; set export_dtype=None")
    yield from export(chunks, NemotronExport(model_cfg), ps, **kwargs)
