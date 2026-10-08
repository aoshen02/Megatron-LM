"""Which Nemotron-H Lightning checkpoints the vLLM-aligned actor accepts.

The release serves Mamba in/out_proj as FP8 W8A8 and attention over a static
FP8 KV cache. A derived checkpoint may serve either with BF16 activations;
nothing else is accepted.
"""

import json

import pytest

from megatron.lite.model.nemotron_h.quantization import STATIC_FP8_KV_CACHE, fp8_kv_cache

LIGHTNING_PATTERN = "MEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMEMAEMEMEMEME"
NVFP4 = {"quant_algo": "W4A16_NVFP4", "group_size": 16}


def _lightning_source(*, fp8_mamba, **kv_cache):
    """A full-depth Lightning config.json with the release's projection recipe."""
    kinds = [{"M": "mamba", "E": "moe", "A": "attention"}[c] for c in LIGHTNING_PATTERN]
    quantized = {"lm_head": NVFP4}
    for index, kind in enumerate(kinds):
        prefix = f"backbone.layers.{index}.mixer"
        if kind == "mamba" and fp8_mamba:
            for projection in ("in_proj", "out_proj"):
                quantized[f"{prefix}.{projection}"] = {"quant_algo": "FP8"}
        elif kind == "moe":
            for owner in ["shared_experts", *(f"experts.{e}" for e in range(128))]:
                for projection in ("up_proj", "down_proj"):
                    quantized[f"{prefix}.{owner}.{projection}"] = NVFP4
    return {
        "model_type": "nemotron_h", "layers_block_type": kinds, "num_hidden_layers": 52,
        "hidden_size": 2688, "vocab_size": 131072, "num_attention_heads": 32,
        "num_key_value_heads": 2, "head_dim": 128, "mamba_num_heads": 64,
        "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4,
        "chunk_size": 128, "n_routed_experts": 128, "num_experts_per_tok": 6,
        "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712,
        "n_shared_experts": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": True,
        "layer_norm_epsilon": 1e-5, "residual_in_fp32": False, "mlp_hidden_act": "relu2",
        "mamba_hidden_act": "silu", "attention_bias": False, "mlp_bias": False,
        "mamba_proj_bias": False, "use_bias": False, "use_conv_bias": True,
        "tie_word_embeddings": False, "n_group": 1, "topk_group": 1,
        "quantization_config": {
            "quant_algo": "MIXED_PRECISION", "quantized_layers": quantized, **kv_cache
        },
    }


def _validate(tmp_path, source):
    from megatron.lite.model.nemotron_h.vllm import protocol
    from megatron.lite.runtime.contracts import OptimizerConfig

    master = {k: v for k, v in source.items() if k != "quantization_config"}
    for name, config in (("checkpoint", source), ("master", master)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(config))
    impl = protocol.ImplConfig(
        hf_path=str(tmp_path / "checkpoint"),
        bf16_master_path=str(tmp_path / "master"),
        optimizer_config=OptimizerConfig(),
    )
    protocol._validate_contract(protocol.build_model_config(source), impl)


@pytest.mark.parametrize("fp8_mamba", [True, False], ids=["fp8_mamba", "bf16_mamba"])
@pytest.mark.parametrize(
    "kv_cache", [{"kv_cache_scheme": STATIC_FP8_KV_CACHE}, {}], ids=["fp8_kv", "bf16_kv"]
)
def test_contract_accepts_fp8_or_bf16_mamba_and_kv_cache(tmp_path, fp8_mamba, kv_cache):
    _validate(tmp_path, _lightning_source(fp8_mamba=fp8_mamba, **kv_cache))


@pytest.mark.parametrize(
    "kv_cache",
    [
        {"kv_cache_scheme": {"dynamic": True, "num_bits": 8, "type": "float"}},
        {"kv_cache_scheme": {"dynamic": False, "num_bits": 4, "type": "float"}},
        {"kv_cache_scheme": {}},
        # vLLM reads this ModelOpt key too, so it must not slip past.
        {"kv_cache_quant_algo": "FP8"},
    ],
    ids=["dynamic_fp8", "fp4", "empty", "quant_algo"],
)
def test_contract_rejects_other_kv_cache_recipes(tmp_path, kv_cache):
    with pytest.raises(ValueError, match="static FP8 KV or no KV cache scheme"):
        _validate(tmp_path, _lightning_source(fp8_mamba=True, **kv_cache))


def test_kv_cache_scheme_selects_the_attention_cache():
    assert fp8_kv_cache({"kv_cache_scheme": dict(STATIC_FP8_KV_CACHE)}) is True
    assert fp8_kv_cache({}) is False
    assert fp8_kv_cache({"kv_cache_scheme": None, "kv_cache_quant_algo": None}) is False


@pytest.mark.parametrize("stray", [".k_proj.k_scale", ".v_proj.v_scale"])
def test_bf16_kv_checkpoint_rejects_kv_scales(tmp_path, stray):
    """Serving a BF16 KV cache ignores KV scales; a checkpoint carrying them
    was derived incompletely."""
    from megatron.lite.model.nemotron_h.checkpoint import check_no_kv_scales

    weight_map = {"backbone.layers.5.mixer.k_proj.weight": "a.safetensors"}
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    check_no_kv_scales(tmp_path)
    weight_map[f"backbone.layers.5.mixer{stray}"] = "a.safetensors"
    index.write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="KV scales without an FP8 KV cache scheme"):
        check_no_kv_scales(tmp_path)
