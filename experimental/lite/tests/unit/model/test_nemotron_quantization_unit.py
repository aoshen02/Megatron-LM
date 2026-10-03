"""Checkpoint-domain NVFP4/FP8 encoding and the BF16 master-weight VJPs.

The actor forward and the rollout export read the same encoding, so these tests
pin the encoding itself and the gradients the actor feeds the optimizer.
"""

import os

import pytest
import torch
from megatron.lite.model.nemotron_h.quantization import (
    QuantizedWeight,
    check_reversible,
    requantize,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _nvfp4(weight, scale, global_scale):
    return QuantizedWeight(
        "W4A16_NVFP4",
        {"weight": weight, "weight_scale": scale, "weight_scale_2": global_scale},
    )


def test_zero_group_scale_decodes_an_all_zero_block():
    weight = torch.full((2, 16), 0x21, dtype=torch.uint8)  # codes 1, 2
    scale = torch.tensor([[1.0, 1.0], [0.0, 2.0]]).to(torch.float8_e4m3fn)
    master = _nvfp4(weight, scale, torch.tensor(0.5)).initial_master()
    assert torch.equal(master[1, :16], torch.zeros(16))
    assert torch.equal(master[0, :2], torch.tensor([0.25, 0.5]))


def test_nonpositive_global_scale_is_rejected():
    weight = torch.zeros((1, 8), dtype=torch.uint8)
    scale = torch.ones((1, 1)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        _nvfp4(weight, scale, torch.tensor(0.0)).initial_master()


@cuda
def test_nvfp4_requantization_is_idempotent_on_its_own_output():
    """The checkpoint rule reproduces its own dequantized weights, so an
    unchanged master keeps its deployment values across refreshes."""
    g = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(256, 512, generator=g, device="cuda").to(torch.bfloat16)
    first = requantize("W4A16_NVFP4", weight)
    decoded = QuantizedWeight("W4A16_NVFP4", first).initial_master()
    second = requantize("W4A16_NVFP4", decoded.to(torch.bfloat16))
    assert torch.equal(QuantizedWeight("W4A16_NVFP4", second).initial_master(), decoded)


@cuda
def test_fp8_requantization_uses_the_tensor_amax():
    weight = torch.tensor([[1.0, -2.0], [0.5, 3.0]], device="cuda").to(torch.bfloat16)
    out = requantize("FP8", weight)
    assert torch.isclose(out["weight_scale"].cpu(), torch.tensor(3.0 / 448))
    assert out["weight"].float().abs().max() == 448


REQUANT_SHAPES = [(16, 64), (128, 1024), (1856, 2688), (2688, 1856)]


def _requant_input(kind, shape):
    rows, columns = shape
    if kind == "random":
        g = torch.Generator(device="cuda").manual_seed(rows * 7919 + columns)
        return torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    if kind == "boundary":
        # Pairs straddling block and FP4 decision boundaries.
        base = torch.linspace(-12.0, 12.0, columns // 2, device="cuda")
        row = torch.empty(columns, device="cuda")
        row[0::2], row[1::2] = base - 1e-3, base + 1e-3
        return row.expand(rows, columns).to(torch.bfloat16).contiguous()
    if kind == "zeros":
        return torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    if kind == "maxes":
        return torch.full(shape, torch.finfo(torch.bfloat16).max, device="cuda").to(
            torch.bfloat16
        )
    raise ValueError(kind)


@cuda
@pytest.mark.parametrize("shape", REQUANT_SHAPES)
@pytest.mark.parametrize("kind", ["random", "boundary", "zeros", "maxes"])
def test_nvfp4_requantize_is_te_4over6_mse_bytewise(kind, shape):
    """requantize == TE's reference 4over6 (E4M3 bound 256, MSE choice), and the
    global scale is amax / 1536 per HF tensor, byte for byte."""
    from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import (
        NVFP4QuantizerRef,
    )

    weight = _requant_input(kind, shape)
    out = requantize("W4A16_NVFP4", weight)
    amax = weight.float().abs().amax().reshape(1)
    packed, scale = NVFP4QuantizerRef._quantize_blockwise_reference(
        weight, amax, 16, 1, pow_2_scales=False, nvfp4_use_4over6=True,
        nvfp4_e4m3_max=256, nvfp4_4over6_err_mode="MSE", eps=0.0,
    )
    rows, columns = shape
    assert torch.equal(out["weight"], packed.view(torch.uint8)[:rows, : columns // 2])
    assert torch.equal(
        out["weight_scale"].view(torch.uint8),
        scale.view(torch.uint8)[:rows, : columns // 16],
    )
    expected = amax.reshape(()) / torch.tensor(1536.0, device="cuda")
    assert out["weight_scale_2"].dtype == torch.float32
    global_scale = out["weight_scale_2"].view(torch.int32)
    assert torch.equal(global_scale, expected.view(torch.int32))


@cuda
@pytest.mark.parametrize("kind", ["random", "boundary", "maxes"])
def test_fp8_requantize_rounds_the_quotient_through_bf16(kind):
    """ModelOpt encodes E4M3(BF16(w / scale)) with scale = amax / 448."""
    weight = _requant_input(kind, (64, 256))
    out = requantize("FP8", weight)
    scale = weight.float().abs().amax() / torch.tensor(448.0, device="cuda")
    assert torch.equal(out["weight_scale"], scale)
    quotient = (weight.float() / scale).to(torch.bfloat16).float()
    expected = quotient.clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(out["weight"].view(torch.uint8), expected.view(torch.uint8))
    assert out["weight"].float().abs().max() == 448


@cuda
@pytest.mark.parametrize("algorithm", ["W4A16_NVFP4", "FP8"])
def test_check_reversible_accepts_the_source_and_rejects_a_changed_master(algorithm):
    weight = _requant_input("random", (128, 1024))
    tensors = requantize(algorithm, weight)
    check_reversible(algorithm, weight, tensors, "w", exact_global=True)
    dequantized = QuantizedWeight(algorithm, tensors).initial_master()
    check_reversible(algorithm, dequantized.to(torch.bfloat16), tensors, "w")
    changed = weight.clone()
    changed[0, :16] *= 1.5
    with pytest.raises(RuntimeError):
        check_reversible(algorithm, changed, tensors, "w", exact_global=True)


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_requantized_bytes_are_what_the_vllm_loaders_hold(vllm_oracle_runtime):
    """Loading requantize's tensors through vLLM's ModelOpt weight loaders keeps
    them byte for byte: per-tensor global scales, one per routed expert and
    projection (relu2, no gate/up fusion), as the checkpoint stores them."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear

    up = requantize("W4A16_NVFP4", _requant_input("random", (1856, 2688)))
    down = requantize("W4A16_NVFP4", _requant_input("random", (2688, 1856)))
    with set_current_vllm_config(vllm_oracle_runtime), torch.device("cuda"):
        linear = ReplicatedLinear(
            2688, 1856, bias=False, params_dtype=torch.bfloat16,
            quant_config=_oracle_quant_config(["model.proj"]), prefix="model.proj",
            return_bias=False, disable_tp=True,
        )
    with torch.no_grad():
        for name, tensor in up.items():
            parameter = getattr(linear, name)
            parameter.weight_loader(parameter, tensor.clone())
            assert torch.equal(
                parameter.data.reshape(-1).view(torch.uint8),
                tensor.reshape(-1).view(torch.uint8),
            ), name

    experts, local = _empty_oracle_experts(vllm_oracle_runtime)
    expert = 5
    with set_current_vllm_config(local), torch.no_grad():
        for shard, stem, tensors in (("w1", "w13", up), ("w2", "w2", down)):
            for suffix, tensor in tensors.items():
                parameter = getattr(experts, f"{stem}_{suffix}")
                experts.weight_loader(
                    parameter, tensor.clone(), f"experts.{expert}.{suffix}",
                    shard_id=shard, expert_id=expert,
                )
                loaded = parameter.data[expert].reshape(-1)[: tensor.numel()]
                assert torch.equal(
                    loaded.view(torch.uint8), tensor.reshape(-1).view(torch.uint8)
                ), (stem, suffix)


_LIGHTNING_BF16 = os.environ.get("MLITE_LIGHTNING_BF16")
_LIGHTNING_NVFP4 = os.environ.get("MLITE_LIGHTNING_NVFP4")


@cuda
@pytest.mark.skipif(
    not (_LIGHTNING_BF16 and _LIGHTNING_NVFP4),
    reason="set MLITE_LIGHTNING_BF16/MLITE_LIGHTNING_NVFP4 to the a9904d2/bee7596 dirs",
)
def test_requantize_reproduces_a_lightning_checkpoint_tensor():
    """Experts 10 down_proj of layer 1 is one of the 1317 checkpoint tensors the
    rule reproduces byte for byte from the BF16 release."""
    import json
    from pathlib import Path

    from safetensors import safe_open

    def read(root, key):
        root = Path(root)
        index = json.loads((root / "model.safetensors.index.json").read_text())
        with safe_open(root / index["weight_map"][key], "pt", device="cuda") as f:
            return f.get_tensor(key)

    name = "backbone.layers.1.mixer.experts.10.down_proj"
    out = requantize("W4A16_NVFP4", read(_LIGHTNING_BF16, f"{name}.weight"))
    for suffix, tensor in out.items():
        stored = read(_LIGHTNING_NVFP4, f"{name}.{suffix}")
        assert torch.equal(
            stored.reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8)
        ), suffix


@cuda
def test_linear_vjp_is_the_bf16_master_weight_gradient():
    from megatron.lite.model.nemotron_h.functional import native_linear_vjp

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(64, 128, generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn(96, 128, generator=g, device="cuda").to(torch.bfloat16)
    dy = torch.randn(64, 96, generator=g, device="cuda").to(torch.bfloat16)
    dx, dw = native_linear_vjp(dy, x, w)
    torch.testing.assert_close(dx.float(), dy.float() @ w.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dw.float(), dy.float().T @ x.float(), rtol=1e-2, atol=1e-2)


def _routed_reference(x, fc1, visible, up, down, routes, ids, dy):
    """The per-input VJP contract, in FP32 autograd: FC1 and expert outputs
    take their visible values (identity straight-through), so the route
    gradient is <dy, visible> and the rest is the master-weight VJP."""
    ref = [t.detach().float().requires_grad_() for t in (x, up, down, routes)]
    u = torch.einsum("mk,msik->msi", ref[0], ref[1][ids])
    u = u + (fc1.float() - u).detach()
    v = torch.einsum("msi,mski->msk", u.relu().square(), ref[2][ids])
    v = v + (visible.float() - v).detach()
    (ref[3][..., None] * v).sum(1).backward(dy.float())
    return [t.grad for t in ref]


@cuda
def test_routed_vjp_follows_the_per_input_contract():
    """Route weights: <dy, visible expert output> (exact). Inputs and expert
    weights: the BF16-master VJP through the visible FC1 output."""
    from megatron.lite.model.nemotron_h.nvfp4_moe_vjp import routed_vjp

    g = torch.Generator(device="cuda").manual_seed(0)
    m, k, i, e, topk = 48, 64, 32, 8, 3

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)

    x, dy = randn(m, k), randn(m, k)
    up, down = randn(e, i, k, scale=0.1), randn(e, k, i, scale=0.1)
    ids = torch.stack([torch.randperm(e, generator=g, device="cuda")[:topk] for _ in range(m)])
    ids[0] = torch.tensor([0, 1, 2])  # expert 7 may receive no rows
    routes = torch.rand(m, topk, generator=g, device="cuda")
    fc1 = torch.einsum("mk,msik->msi", x.float(), up[ids].float()).to(torch.bfloat16)
    # A deployment output that differs from the master product, as quantized
    # serving does.
    visible = (
        torch.einsum("msi,mski->msk", fc1.float().relu().square(), down[ids].float())
        * (1 + 0.05 * torch.randn(m, topk, k, generator=g, device="cuda"))
    ).to(torch.bfloat16)
    got = routed_vjp(x, fc1.reshape(m * topk, i), visible.reshape(m * topk, k),
                     up, down, routes, ids, dy)
    expected = _routed_reference(x, fc1, visible, up, down, routes, ids, dy)
    for actual, reference in zip(got, expected, strict=True):
        torch.testing.assert_close(actual.float(), reference, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(got[3], expected[3], rtol=1e-5, atol=1e-5)


# The direct kernel calls must reproduce the vLLM layer objects bit for bit.
# The oracle builds those objects (ReplicatedLinear + ModelOpt, FusedMoE with
# the Humming backend) on the same checkpoint bytes, as serving does.
ORACLE_ROWS = (1, 7, 64, 513, 8192)
LIGHTNING_MOE = dict(experts=128, hidden=2688, intermediate=1856, topk=6)


def _random_nvfp4(rows, columns, generator, *, global_scale=2e-3):
    packed = torch.randint(
        0, 256, (rows, columns // 2), generator=generator, device="cuda", dtype=torch.uint8
    )
    scale = (
        torch.rand(rows, columns // 16, generator=generator, device="cuda") * 3 + 0.25
    ).to(torch.float8_e4m3fn)
    return packed, scale, torch.tensor([global_scale], device="cuda")


@pytest.fixture(scope="module")
def vllm_oracle_runtime():
    import os

    import torch.distributed as dist
    from vllm.config import (
        CompilationConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    for name, value in (
        ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29517"),
    ):
        os.environ.setdefault(name, value)
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    config = VllmConfig(
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        compilation_config=CompilationConfig(custom_ops=["none", "+quant_fp8"]),
    )
    config.kernel_config.moe_backend = "humming"
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    init_batch_invariance()
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        ensure_model_parallel_initialized(1, 1)
        if not is_workspace_manager_initialized():
            init_workspace_manager(torch.device("cuda", 0))
    # The direct kernels run outside any current vLLM config.
    return config


def _oracle_quant_config(prefixes):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
            "quantized_layers": {
                p: {"quant_algo": "W4A16_NVFP4", "group_size": 16} for p in prefixes
            },
        }
    )


def _oracle_linear(prefix, tensors, *, shared):
    from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        NemotronSharedNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.linear import ReplicatedLinear

    n, packed_k = tensors["weight"].shape
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            packed_k * 2, n, bias=False, params_dtype=torch.bfloat16,
            quant_config=_oracle_quant_config([prefix]), prefix=prefix,
            return_bias=False, disable_tp=True,
        )
    if shared:
        layer.quant_method.kernel = NemotronSharedNvFp4LinearKernel(
            NvFp4LinearLayerConfig()
        )
    assert shared or type(layer.quant_method.kernel).__name__ == "HummingNvFp4LinearKernel"
    with torch.no_grad():
        for name, tensor in tensors.items():
            parameter = getattr(layer, name)
            parameter.weight_loader(parameter, tensor.clone())
        layer.quant_method.process_weights_after_loading(layer)
    return layer


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("shared", [False, True], ids=["humming", "cute_dsl"])
@pytest.mark.parametrize("shape", [(2688, 3712), (3712, 2688), (4096, 2688)])
def test_direct_nvfp4_linear_matches_vllm_layer_bitwise(vllm_oracle_runtime, shared, shape):
    from megatron.lite.model.nemotron_h.kernels import (
        CuteDslNvfp4Linear,
        HummingNvfp4Linear,
    )
    from vllm.config import set_current_vllm_config

    g = torch.Generator(device="cuda").manual_seed(1)
    n, k = shape
    packed, scale, global_scale = _random_nvfp4(n, k, g)
    tensors = {"weight": packed, "weight_scale": scale, "weight_scale_2": global_scale}
    with set_current_vllm_config(vllm_oracle_runtime):
        oracle = _oracle_linear("model.proj", tensors, shared=shared)
    direct = (CuteDslNvfp4Linear if shared else HummingNvfp4Linear)(
        packed, scale, global_scale
    )
    for rows in ORACLE_ROWS:
        x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
        with set_current_vllm_config(vllm_oracle_runtime):
            expected = oracle(x)
        assert torch.equal(direct(x), expected), rows


def _empty_oracle_experts(config):
    from copy import copy

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

    e, h, i = (LIGHTNING_MOE[k] for k in ("experts", "hidden", "intermediate"))
    prefix = "backbone.layers.1.mixer.experts"
    quant = _oracle_quant_config(
        [f"{prefix}.{x}.{p}" for x in range(e) for p in ("up_proj", "down_proj")]
    )
    local = copy(config)
    local.compilation_config = copy(config.compilation_config)
    local.compilation_config.static_forward_context = {}
    local.compilation_config.static_all_moe_layers = []
    with set_current_vllm_config(local), torch.device("cuda"):
        return FusedMoEFactory(
            num_experts=e, top_k=LIGHTNING_MOE["topk"], hidden_size=h,
            intermediate_size=i, params_dtype=torch.bfloat16, quant_config=quant,
            prefix=prefix, ckpt_names=("up_proj", "down_proj", ""),
            activation="relu2_no_mul", apply_router_weight_on_input=False,
            shared_experts=None, enable_eplb=False, num_redundant_experts=0,
            use_grouped_topk=True, num_expert_group=1, topk_group=1,
            renormalize=True, scoring_func="sigmoid",
            e_score_correction_bias=torch.zeros(e, dtype=torch.float32),
            routed_scaling_factor=1.0, apply_routed_scale_to_output=True,
            router_logits_dtype=torch.float32, skip_padding=True,
        ).routed_experts, local


def _oracle_experts(stacks, config):
    from vllm.config import set_current_vllm_config

    layer, local = _empty_oracle_experts(config)
    with set_current_vllm_config(local), torch.device("cuda"), torch.no_grad():
        for stem, (packed, scale, global_scale) in stacks.items():
            getattr(layer, f"{stem}_weight").copy_(packed)
            getattr(layer, f"{stem}_weight_scale").copy_(scale)
            target = getattr(layer, f"{stem}_weight_scale_2")
            target.copy_(global_scale.reshape(target.shape))
        layer.w13_input_scale.fill_(float("nan"))
        layer.w2_input_scale.fill_(float("nan"))
        layer.quant_method.process_weights_after_loading(layer)
    return layer, local


def _oracle_routes(layer, config, x, ids):
    from vllm.forward_context import set_forward_context

    experts = layer.quant_method.moe_kernel.fused_experts
    rows, topk = ids.shape
    with set_forward_context(None, config, num_tokens=rows):
        metas, required = experts.get_buffer_metas(rows, topk, layer.activation)
        buffers = {
            name: torch.empty(metas[name]["shape"], dtype=metas[name]["dtype"], device="cuda")
            for name in required
            if name != "output"
        }
        kwargs1, kwargs2, scatter_idx = experts.prepare_humming_moe_kwargs(
            topk_ids=ids, expert_map=None, expert_tokens_meta=None
        )
        inputs, scale, scale_2 = experts.process_input(
            "w13", inputs=x, input_scale=None,
            quanted_input=buffers["quanted_gate_up_input"],
        )
        experts.humming_forward(
            "w13", inputs=inputs, weight=layer.w13_weight, input_scale=scale,
            input_scale_2=scale_2, outputs=buffers["gate_up_output"], **kwargs1,
        )
        inputs, scale, scale_2 = experts.process_input(
            "w2", inputs=buffers["gate_up_output"],
            quanted_input=buffers["quanted_down_input"], activation=layer.activation,
            scatter_idx=scatter_idx,
        )
        experts.humming_forward(
            "w2", inputs=inputs, weight=layer.w2_weight, input_scale=scale,
            input_scale_2=scale_2, outputs=buffers["down_output"].view(-1, x.shape[1]),
            **kwargs2,
        )
    return buffers["gate_up_output"], buffers["down_output"].view(rows, topk, -1)


def _lightning_stacks(generator):
    e, h, i = (LIGHTNING_MOE[k] for k in ("experts", "hidden", "intermediate"))
    stacks = {}
    for stem, (rows, columns) in (("w13", (i, h)), ("w2", (h, i))):
        packed, scale, _ = _random_nvfp4(e * rows, columns, generator)
        global_scale = torch.rand(e, generator=generator, device="cuda") * 2e-3 + 1e-3
        stacks[stem] = (
            packed.view(e, rows, -1), scale.view(e, rows, -1), global_scale
        )
    return stacks


def _route_ids(rows, generator, *, experts=128, topk=6):
    scores = torch.rand(rows, experts, generator=generator, device="cuda")
    return scores.topk(topk, dim=-1).indices.to(torch.int32)


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_direct_routed_experts_match_vllm_fused_moe_bitwise(vllm_oracle_runtime):
    """Per-route FC1 and down outputs for every M."""
    from megatron.lite.model.nemotron_h.kernels import HummingRoutedExperts

    g = torch.Generator(device="cuda").manual_seed(2)
    stacks = _lightning_stacks(g)
    oracle, config = _oracle_experts(stacks, vllm_oracle_runtime)
    direct = HummingRoutedExperts(
        stacks["w13"], stacks["w2"], num_experts=128, offset=0,
        layer_name="backbone.layers.1.mixer.experts",
    )
    for rows in ORACLE_ROWS:
        x = torch.randn(rows, 2688, generator=g, device="cuda").to(torch.bfloat16)
        ids = _route_ids(rows, g)
        fc1, down = direct.routes(x, ids)
        oracle_fc1, oracle_down = _oracle_routes(oracle, config, x, ids)
        assert torch.equal(fc1, oracle_fc1), rows
        assert torch.equal(down, oracle_down), rows


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_direct_query_fp8_quant_matches_vllm_quant_fp8_bitwise(vllm_oracle_runtime):
    from megatron.lite.model.nemotron_h.kernels import scaled_fp8_quant
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    with set_current_vllm_config(vllm_oracle_runtime):
        oracle = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    g = torch.Generator(device="cuda").manual_seed(3)
    scale = torch.ones(1, device="cuda")
    for rows in ORACLE_ROWS:
        q = (torch.randn(rows, 4096, generator=g, device="cuda") * 50).to(torch.bfloat16)
        expected, _ = oracle.forward_cuda(q, scale)
        actual, _ = scaled_fp8_quant(q, scale)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), rows


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_routed_vjp_on_the_humming_deployment_at_updated_weights():
    """The training VJP on the real deployment, at theta0 and two updated
    snapshots whose deployment is requantized from the updated masters."""
    from megatron.lite.model.nemotron_h.kernels import HummingRoutedExperts
    from megatron.lite.model.nemotron_h.nvfp4_ep4 import (
        EP4_ONESIDED_REDUCTION,
        ep4_routed_experts,
        reduce_ep4_parts,
    )
    from megatron.lite.model.nemotron_h.nvfp4_moe_vjp import routed_vjp

    g = torch.Generator(device="cuda").manual_seed(5)
    up = (torch.randn(128, 1856, 2688, generator=g, device="cuda") * 0.02).bfloat16()
    down = (torch.randn(128, 2688, 1856, generator=g, device="cuda") * 0.02).bfloat16()
    rows = 33
    x = torch.randn(rows, 2688, generator=g, device="cuda").to(torch.bfloat16)
    ids = _route_ids(rows, g)
    routes = torch.rand(rows, 6, generator=g, device="cuda")
    dy = torch.randn(rows, 2688, generator=g, device="cuda").to(torch.bfloat16)
    for snapshot in range(3):
        if snapshot:
            up = up + (torch.randn(up.shape, generator=g, device="cuda") * 1e-3).bfloat16()
            down = down + (torch.randn(down.shape, generator=g, device="cuda") * 1e-3).bfloat16()
        stacks = []
        for master in (up, down):
            parts = [requantize("W4A16_NVFP4", master[e]) for e in range(128)]
            stacks.append(tuple(
                torch.stack([q[name] for q in parts])
                for name in ("weight", "weight_scale", "weight_scale_2")
            ))
        experts = HummingRoutedExperts(*stacks, num_experts=128, offset=0, layer_name="experts")
        out, fc1, visible = ep4_routed_experts(
            experts, x, routes, ids, EP4_ONESIDED_REDUCTION, return_fc1=True
        )
        # The saved per-route output is the one the forward combined.
        per_route = visible.view(rows, 6, 2688)
        parts = []
        for rank in range(4):
            mapping = torch.full((128,), -1, dtype=torch.int32, device="cuda")
            mapping[rank * 32 : (rank + 1) * 32] = torch.arange(32, device="cuda")
            parts.append(experts.rank_partial(per_route, routes, ids, mapping))
        assert torch.equal(reduce_ep4_parts(parts, ids, EP4_ONESIDED_REDUCTION), out)
        got = routed_vjp(x, fc1, visible, up, down, routes, ids, dy)
        expected = _routed_reference(
            x, fc1.view(rows, 6, -1), per_route, up, down, routes, ids.long(), dy
        )
        torch.testing.assert_close(got[3], expected[3], rtol=1e-5, atol=1e-4)
        for name, actual, reference in zip(("dx", "d_up", "d_down"), got, expected):
            error = ((actual.float() - reference).norm() / reference.norm()).item()
            assert error < 1e-2, (snapshot, name, error)


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_direct_nvfp4_linears_reject_inputs_serving_rejects(monkeypatch):
    from megatron.lite.model.nemotron_h.kernels import CuteDslNvfp4Linear, HummingNvfp4Linear

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    g = torch.Generator(device="cuda").manual_seed(6)
    packed, scale, global_scale = _random_nvfp4(256, 512, g)
    for linear in (HummingNvfp4Linear, CuteDslNvfp4Linear):
        unloaded = scale.clone()
        unloaded.view(torch.uint8)[0, 0] = 0x7F  # E4M3 NaN
        with pytest.raises(RuntimeError, match="never loaded"):
            linear(packed, unloaded, global_scale)
        with pytest.raises(ValueError, match="global scale"):
            linear(packed, scale, global_scale.repeat(2))


@cuda
@pytest.mark.gpus(2, min_architecture="blackwell")
def test_cute_dsl_linear_builds_on_its_weight_device(monkeypatch):
    """vLLM's swizzle allocates on the current device; build on the weight's."""
    from megatron.lite.model.nemotron_h.kernels import CuteDslNvfp4Linear

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    g = torch.Generator(device="cuda:1").manual_seed(7)
    packed = torch.randint(0, 256, (256, 256), generator=g, device="cuda:1", dtype=torch.uint8)
    scale = (torch.rand(256, 32, generator=g, device="cuda:1") + 0.5).to(torch.float8_e4m3fn)
    with torch.cuda.device(0):
        linear = CuteDslNvfp4Linear(packed, scale, torch.tensor([2e-3], device="cuda:1"))
    assert {t.device for t in (linear.weight, linear.weight_scale, linear.alpha)} == {
        torch.device("cuda", 1)
    }


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_ep4_rank_humming_schedule_matches_the_full_deployment_bitwise():
    """The serving EP4 rank (32 local experts, configure_nemotron_humming,
    valid_shape_m from the global token count, expert_map, only the tokens
    routed to it) gives the routes and rank partial the trainer's 128-expert
    deployment computes, bit for bit."""
    from megatron.lite.model.nemotron_h.kernels import HummingRoutedExperts

    g = torch.Generator(device="cuda").manual_seed(8)
    stacks = _lightning_stacks(g)
    full = HummingRoutedExperts(
        stacks["w13"], stacks["w2"], num_experts=128, offset=0, layer_name="experts"
    )
    ranks = [
        HummingRoutedExperts(
            *(tuple(t[r * 32 : (r + 1) * 32] for t in stacks[s]) for s in ("w13", "w2")),
            num_experts=128, offset=r * 32, layer_name="experts",
        )
        for r in range(4)
    ]
    for rows in ORACLE_ROWS:
        x = torch.randn(rows, 2688, generator=g, device="cuda").to(torch.bfloat16)
        ids = _route_ids(rows, g)
        weights = torch.rand(rows, 6, generator=g, device="cuda")
        _, down = full.routes(x, ids)
        for r, rank in enumerate(ranks):
            owned = ids // 32 == r
            tokens = owned.any(1).nonzero()[:, 0]
            if tokens.numel() == 0:
                continue
            _, local = rank.routes(x[tokens], ids[tokens], global_tokens=4 * rows)
            assert torch.equal(local[owned[tokens]], down[tokens][owned[tokens]]), (rows, r)
            assert torch.equal(
                rank.rank_partial(local, weights[tokens], ids[tokens], rank.expert_map),
                full.rank_partial(down, weights, ids, rank.expert_map)[tokens],
            ), (rows, r)


def _theta0_case(tmp_path, masters, stored):
    """Write ``stored`` as a one-shard checkpoint; run the theta0 check."""
    import json

    from megatron.lite.model.nemotron_h.checkpoint import _check_theta0_agreement
    from safetensors.torch import save_file

    flat = {name: t.contiguous().cpu() for name, t in stored.items()}
    save_file(flat, str(tmp_path / "model.safetensors"))
    index = {name: "model.safetensors" for name in flat}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    _check_theta0_agreement(masters, tmp_path, index)


def _nvfp4_entry(prefix, master, deployed):
    stored = {f"{prefix}.{k}": v for k, v in requantize("W4A16_NVFP4", master).items()}
    return (prefix, "W4A16_NVFP4", master, lambda: deployed), stored


@cuda
def test_theta0_check_accepts_the_requantization_and_zero_blocks(tmp_path):
    """Requantizing the source passes; blocks that underflow to zero are
    excluded from the code/scale comparison (ModelOpt 2^-9 vs TE 0)."""
    g = torch.Generator(device="cuda").manual_seed(9)
    master = (torch.randn(256, 512, generator=g, device="cuda") * 0.02).bfloat16()
    master[:, :16] = 0
    deployed = requantize("W4A16_NVFP4", master)
    entry, stored = _nvfp4_entry("m.a", master, deployed)
    stored["m.a.weight_scale"][:, 0] = torch.tensor(2.0**-9).to(torch.float8_e4m3fn)
    _theta0_case(tmp_path, [entry], stored)


@cuda
def test_theta0_check_fails_one_bad_tensor(tmp_path):
    from megatron.lite.model.nemotron_h.checkpoint import THETA0_NVFP4_TENSOR_MIN_VALUES

    g = torch.Generator(device="cuda").manual_seed(10)
    entries, stored = [], {}
    for i in range(8):
        master = (torch.randn(256, 512, generator=g, device="cuda") * 0.02).bfloat16()
        deployed = requantize("W4A16_NVFP4", master)
        if i == 3:  # one tensor with 6% of its code bytes changed
            flip = torch.rand(deployed["weight"].shape, generator=g, device="cuda") < 0.06
            deployed["weight"] = torch.where(flip, deployed["weight"] ^ 0x11, deployed["weight"])
        entry, part = _nvfp4_entry(f"m.t{i}", master, deployed)
        entries.append(entry)
        stored.update(part)
    with pytest.raises(RuntimeError, match=r"m\.t3"):
        _theta0_case(tmp_path, entries, stored)
    assert THETA0_NVFP4_TENSOR_MIN_VALUES > 0.9


@cuda
def test_theta0_check_allows_calibrated_fp8_and_rejects_standard_mismatch(tmp_path):
    """FP8 tensors whose checkpoint scale is not amax/448 (ModelOpt-calibrated)
    may differ at theta0 (scheme A); amax/448 tensors must match exactly."""
    from megatron.lite.model.nemotron_h.quantization import fp8_encode

    g = torch.Generator(device="cuda").manual_seed(11)
    master = (torch.randn(128, 256, generator=g, device="cuda") * 0.02).bfloat16()
    deployed = requantize("FP8", master)
    calibrated = deployed["weight_scale"] * 1.5
    stored = {"m.f.weight": fp8_encode(master, calibrated), "m.f.weight_scale": calibrated}
    entry = ("m.f", "FP8", master, lambda: deployed)
    _theta0_case(tmp_path, [entry], stored)

    wrong = dict(deployed, weight=fp8_encode(master, deployed["weight_scale"] * 1.01))
    stored = {"m.f.weight": deployed["weight"], "m.f.weight_scale": deployed["weight_scale"]}
    with pytest.raises(RuntimeError, match="amax/448"):
        _theta0_case(tmp_path, [("m.f", "FP8", master, lambda: wrong)], stored)


@cuda
def test_bf16_source_check_rejects_another_release():
    """A master that is not the checkpoint's source fails the encoding check."""
    from megatron.lite.model.nemotron_h.quantization import check_reversible

    g = torch.Generator(device="cuda").manual_seed(12)
    source = (torch.randn(256, 512, generator=g, device="cuda") * 0.02).bfloat16()
    checkpoint = requantize("W4A16_NVFP4", source)
    check_reversible("W4A16_NVFP4", source, checkpoint, "m", exact_global=True)
    other = (source.float() + torch.randn(256, 512, generator=g, device="cuda") * 2e-3)
    with pytest.raises((RuntimeError, ValueError)):
        check_reversible("W4A16_NVFP4", other.bfloat16(), checkpoint, "m", exact_global=True)


def _deployment_chunk(nvfp4, fp8):
    from megatron.lite.model.nemotron_h.fp8_training import Fp8TrainingLinear
    from megatron.lite.model.nemotron_h.quantization import Nvfp4TrainingLinear

    return torch.nn.ModuleDict(
        {
            "lin": Nvfp4TrainingLinear(nvfp4, lambda **_: None, device="cuda"),
            "fp8": Fp8TrainingLinear(fp8, device="cuda"),
        }
    )


def _exported_bytes(chunk):
    from megatron.lite.model.nemotron_h.checkpoint import NemotronExport

    return {
        name: tensor.contiguous().reshape(-1).view(torch.uint8).cpu()
        for name, tensor in NemotronExport.iter_export_tensors(None, chunk)
    }


@cuda
def test_checkpoint_restore_reinstalls_the_last_deployed_bytes():
    """A training-checkpoint restore serves the bytes deployed when it was saved:
    the checkpoint bytes at step 0 (requant(master) differs there) and
    requant(master) after an update."""
    from megatron.lite.model.nemotron_h.protocol import (
        _refresh_quantized,
        _restore_quantized,
    )
    from megatron.lite.model.nemotron_h.quantization import fp8_encode

    g = torch.Generator(device="cuda").manual_seed(13)
    nvfp4 = _nvfp4(*_random_nvfp4(128, 256, g))
    master = (torch.randn(64, 128, generator=g, device="cuda") * 0.02).bfloat16()
    calibrated = master.float().abs().amax() / 448 * 1.5
    fp8 = QuantizedWeight(
        "FP8",
        {
            "weight": fp8_encode(master, calibrated),
            "weight_scale": calibrated,
            "input_scale": torch.tensor(0.01, device="cuda"),
        },
    )
    trained = _deployment_chunk(nvfp4, fp8)
    for name, module in trained.items():
        algorithm = "W4A16_NVFP4" if name == "lin" else "FP8"
        weight = requantize(algorithm, module.weight)["weight"]
        assert not torch.equal(weight.view(torch.uint8), module._packed.view(torch.uint8))
    for step in range(2):
        if step:
            with torch.no_grad():
                for parameter in trained.parameters():
                    noise = torch.randn(parameter.shape, generator=g, device="cuda")
                    parameter.add_((noise * 1e-3).to(parameter.dtype))
            _refresh_quantized([trained])
        saved, deployed = trained.state_dict(), _exported_bytes(trained)
        restored = _deployment_chunk(nvfp4, fp8)
        restored.load_state_dict(saved, strict=False)
        _restore_quantized([restored])
        exported = _exported_bytes(restored)
        assert exported.keys() == deployed.keys()
        for name, value in deployed.items():
            assert torch.equal(exported[name], value), (step, name)


