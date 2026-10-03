"""Checkpoint-domain NVFP4/FP8 encoding and the BF16 master-weight VJPs.

The actor forward and the rollout export read the same encoding, so these tests
pin the encoding itself and the gradients the actor feeds the optimizer.
"""

import pytest
import torch
from megatron.lite.model.nemotron_h.quantization import QuantizedWeight, requantize

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


def _oracle_experts(stacks, config):
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
        layer = FusedMoEFactory(
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
        ).routed_experts
        with torch.no_grad():
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
