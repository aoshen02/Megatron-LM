"""Explicit quantized diagnostic construction, not a production recipe.

Torch distributed must be initialized. The vLLM config, groups and workspace
are either caller-owned or created by vllm_runtime.ensure_vllm_runtime().
"""

import json
from pathlib import Path

import torch

from .checkpoint import load_fp8_kv_scales
from .nvfp4_moe_vjp import SURROGATE_CONTRACT, validate_matmul_policy


def _reject_grad_enabled_forward(module, args):
    if torch.is_grad_enabled():
        raise RuntimeError("Full-depth diagnostic requires torch.no_grad()")


def _validate_full_depth(config, source):
    from .config import NemotronHConfig

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


def validate_proxy_config(config, impl):
    p = impl.parallel
    if not impl.hf_path:
        raise ValueError("Quantized Nemotron requires an explicit hf_path")
    forward_only = impl.diagnostic_forward_only
    full_training = impl.diagnostic_full_training
    # Full training runs the formal PP4 recipe on the full model or on the
    # 4/5-layer proxy; forward-only diagnostics always cover full depth.
    proxy_depth = config.num_hidden_layers in (4, 5)
    full_depth = forward_only or (full_training and not proxy_depth)
    if not forward_only and impl.surrogate_contract != SURROGATE_CONTRACT:
        raise ValueError("Quantized proxy requires explicit diagnostic V2 contract")
    if not forward_only and impl.optimizer_config is None:
        raise ValueError("Quantized proxy requires explicit optimizer_config")
    if any(value != 1 for value in (p.tp, p.etp or 1, p.ep, p.cp, p.vpp)):
        raise ValueError("Quantized proxy requires TP/ETP/EP/CP/VPP1")
    if p.pp != 1 and not ((forward_only or full_training) and p.pp == 4):
        raise ValueError(
            "Quantized PP4 requires explicit forward-only or full-training diagnostic"
        )
    if not full_depth and (
        config.num_hidden_layers not in (4, 5)
        or not {
            "linear_attention",
            "full_attention",
            "moe",
        }.issubset(config.layers_block_type)
    ):
        raise ValueError(
            "Quantized proxy requires four or five layers covering all operators"
        )
    if (
        config.hidden_size,
        config.n_routed_experts,
        config.moe_intermediate_size,
        config.num_experts_per_tok,
    ) != (2688, 128, 1856, 6):
        raise ValueError("Quantized proxy requires full Lightning operator geometry")
    recipe = config.quantization_config
    if recipe.get("quant_algo") != "MIXED_PRECISION" or recipe.get(
        "kv_cache_scheme"
    ) != {"dynamic": False, "num_bits": 8, "type": "float"}:
        raise ValueError("Expected checkpoint mixed precision and static FP8 KV")
    source = json.loads((Path(impl.hf_path) / "config.json").read_text())
    if (
        source.get("num_hidden_layers")
        != (52 if full_depth else config.num_hidden_layers)
        or source.get("quantization_config") != recipe
    ):
        raise ValueError("Proxy checkpoint metadata disagrees with requested recipe")
    if full_depth:
        _validate_full_depth(config, source)


def caller_runtime(*, pipeline_size=1):
    import vllm.envs as envs
    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed.parallel_state import (
        get_ep_group,
        get_pp_group,
        get_tp_group,
        get_world_group,
    )
    from vllm.v1.worker.workspace import current_workspace_manager

    if pipeline_size not in (1, 4):
        raise ValueError("Quantized runtime supports pipeline size 1 or 4")
    if not envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError("Quantized proxy requires BI=1 before startup")
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() != pipeline_size
    ):
        raise RuntimeError("Caller torch world must match the pipeline size")
    cfg = get_current_vllm_config_or_none()
    if cfg is None or cfg.kernel_config.moe_backend != "humming":
        raise RuntimeError("Caller must provide current Humming VllmConfig")
    if cfg.parallel_config.pipeline_parallel_size != pipeline_size:
        raise RuntimeError("Caller VllmConfig must match the pipeline size")
    try:
        groups = (get_world_group(), get_tp_group(), get_ep_group(), get_pp_group())
        workspace = current_workspace_manager()
    except AssertionError as error:
        raise RuntimeError(
            "Caller must initialize vLLM groups and workspace"
        ) from error
    if tuple(group.world_size for group in groups) != (
        pipeline_size,
        1,
        1,
        pipeline_size,
    ):
        raise RuntimeError("Quantized runtime requires matching world/PP and TP1/EP1")
    if workspace._device != torch.device("cuda", torch.cuda.current_device()):
        raise RuntimeError("Caller workspace must use the current CUDA device")
    validate_matmul_policy(torch.device("cuda"))
    return cfg


def stage_quantization_contract(config, layer_range):
    """Select stage-owned recipe entries without discarding unknown prefixes."""
    start, end = layer_range
    if not 0 <= start < end <= config.num_hidden_layers:
        raise ValueError("Invalid quantized stage layer range")
    prefixes = set()
    for prefix in config.quantization_config["quantized_layers"]:
        if prefix == "lm_head":
            if end == config.num_hidden_layers:
                prefixes.add(prefix)
            continue
        parts = prefix.split(".")
        if (
            len(parts) < 5
            or parts[:2] != ["backbone", "layers"]
            or not parts[2].isdigit()
            or parts[3] != "mixer"
            or not 0 <= int(parts[2]) < config.num_hidden_layers
        ):
            raise ValueError(f"Unknown quantized stage prefix: {prefix}")
        if start <= int(parts[2]) < end:
            prefixes.add(prefix)
    attention_ids = [
        i for i in range(start, end) if config.layers_block_type[i] == "full_attention"
    ]
    return prefixes, attention_ids


def build_quantized_proxy(config, impl, ps, *, layer_range):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    from .fp8_training import Fp8TrainingLinear
    from .model import NemotronModel
    from .nvfp4_experts import Nvfp4ExpertWeights
    from .nvfp4_moe import Nvfp4RoutedDeployment
    from .quantization import CheckpointProjectionFactory, Nvfp4TrainingLinear

    cfg = caller_runtime(pipeline_size=ps.pp_size)
    recipe = config.quantization_config
    quant = ModelOptMixedPrecisionConfig.from_config(recipe)
    factory = CheckpointProjectionFactory(
        impl.hf_path, recipe["quantized_layers"], quant
    )

    def routed_factory(prefix, model_cfg, parallel, *, device, dtype):
        if dtype != torch.bfloat16 or parallel is not ps or model_cfg is not config:
            raise ValueError("Unexpected proxy factory configuration")
        weights = Nvfp4ExpertWeights(
            impl.hf_path,
            prefix,
            recipe["quantized_layers"],
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            tp_size=ps.tp_size,
            ep_size=ps.ep_size,
            device=device,
        )
        return Nvfp4RoutedDeployment(
            weights,
            config,
            cfg,
            quant,
            surrogate_contract=impl.surrogate_contract,
            routed_vjp_backend=impl.routed_vjp_backend,
            routed_vjp_kernel_source=impl.routed_vjp_kernel_source,
            routed_vjp_token_limit=impl.routed_vjp_token_limit,
            routed_forward_reduction=impl.routed_forward_reduction,
            recompute_surrogate=impl.diagnostic_full_training,
        )

    expected_prefixes, attention_ids = stage_quantization_contract(config, layer_range)
    kv = load_fp8_kv_scales(impl.hf_path, attention_ids, device="cuda")
    with set_default_torch_dtype(torch.bfloat16):
        model = NemotronModel(
            config,
            ps,
            layer_range=layer_range,
            device="cuda",
            dtype=torch.bfloat16,
            projection_factory=factory,
            routed_factory=routed_factory,
            fp8_kv_scales=kv,
        )
    covered = set()
    for name, module in model.named_modules():
        prefix = name if name == "lm_head" else "backbone." + name
        if isinstance(module, Fp8TrainingLinear | Nvfp4TrainingLinear):
            covered.add(prefix)
        if isinstance(module, Nvfp4RoutedDeployment):
            covered.update(
                f"{prefix}.{e}.{projection}"
                for e in range(config.n_routed_experts)
                for projection in ("up_proj", "down_proj")
            )
    if covered != expected_prefixes:
        raise ValueError(
            "Quantized proxy construction did not cover every stage recipe prefix"
        )
    model._quantized_proxy_root = str(Path(impl.hf_path).resolve())
    model._vllm_config = cfg
    if impl.diagnostic_forward_only:
        model.register_forward_pre_hook(_reject_grad_enabled_forward)
    return model
