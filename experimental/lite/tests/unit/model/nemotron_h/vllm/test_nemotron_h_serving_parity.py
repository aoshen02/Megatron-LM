"""Nemotron-H forward parity with vLLM batch-invariant serving.

The actor calls the serving kernels directly. These tests compare them bit for
bit with the vLLM layers and runners serving uses, the EP4 routed experts with
the rollout's EP4 combine, and the bytes the actor exports with the bytes its
forward reads.
"""

import pytest
import torch
from megatron.lite.model.nemotron_h.quantization import QuantizedWeight, requantize

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_actor_runtime_guard_accepts_the_installed_flashinfer(monkeypatch):
    pytest.importorskip("flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16")
    from megatron.lite.model.nemotron_h.vllm.primitive.moe import grouped as kernels

    monkeypatch.setattr(kernels, "_FLASHINFER_W4A16_CHECKED", False)
    kernels.check_flashinfer_w4a16()


def test_actor_runtime_guard_rejects_a_changed_helper(monkeypatch):
    module = pytest.importorskip("flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16")
    from megatron.lite.model.nemotron_h.vllm.primitive.moe import grouped as kernels

    def _run_grouped_gemm(weight, weight_sf, activations):  # noqa: ARG001
        raise AssertionError

    monkeypatch.setattr(module, "_run_grouped_gemm", _run_grouped_gemm)
    monkeypatch.setattr(kernels, "_FLASHINFER_W4A16_CHECKED", False)
    with pytest.raises(RuntimeError, match="_run_grouped_gemm"):
        kernels.check_flashinfer_w4a16()


# The direct kernel calls must reproduce the vLLM layer objects bit for bit.
# The oracle builds those objects (ReplicatedLinear + ModelOpt, and FusedMoE
# for its weight loaders) on the same checkpoint bytes, as serving does.
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
def vllm_oracle_runtime(tmp_path_factory):
    import os

    import torch.distributed as dist
    from vllm.config import (
        CompilationConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    for name, value in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
        os.environ.setdefault(name, value)
    torch.cuda.set_device(0)
    assert not dist.is_initialized(), "the vLLM oracle owns the default process group"
    # A file store: under torchrun a tcp:// init would wait for the agent's store.
    store = tmp_path_factory.mktemp("vllm_oracle") / "pg"
    dist.init_process_group("nccl", init_method=f"file://{store}", rank=0, world_size=1)
    config = VllmConfig(
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        compilation_config=CompilationConfig(custom_ops=["none", "+quant_fp8"]),
    )
    # Only the FusedMoE weight loaders are exercised; any NVFP4 backend loads.
    config.kernel_config.moe_backend = "humming"
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    init_batch_invariance()
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        ensure_model_parallel_initialized(1, 1)
        if not is_workspace_manager_initialized():
            init_workspace_manager(torch.device("cuda", 0))
    # The direct kernels run outside any current vLLM config.
    yield config
    destroy_model_parallel()
    destroy_distributed_environment()


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
    from megatron.lite.model.nemotron_h.vllm.primitive.dense import CuteDslNvfp4Linear, HummingNvfp4Linear
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


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_direct_query_fp8_quant_matches_vllm_quant_fp8_bitwise(vllm_oracle_runtime):
    from vllm import _custom_ops as ops
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
        actual, _ = ops.scaled_fp8_quant(q, scale)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), rows


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_requantized_bytes_are_what_the_vllm_loaders_hold(vllm_oracle_runtime):
    """Loading requantize's tensors through vLLM's ModelOpt weight loaders keeps
    them byte for byte: per-tensor global scales, one per routed expert and
    projection (relu2, no gate/up fusion), as the checkpoint stores them."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear

    g = torch.Generator(device="cuda").manual_seed(0)
    up, down = (
        requantize(
            "W4A16_NVFP4",
            torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16),
        )
        for shape in ((1856, 2688), (2688, 1856))
    )
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


def _stacks(generator, experts=128):
    def nvfp4(rows, columns):
        packed = torch.randint(
            0, 256, (experts, rows, columns // 2), generator=generator,
            device="cuda", dtype=torch.uint8,
        )
        scale = (
            torch.rand(experts, rows, columns // 16, generator=generator, device="cuda")
            * 3 + 0.25
        ).to(torch.float8_e4m3fn)
        alpha = torch.rand(experts, generator=generator, device="cuda") * 2e-3 + 1e-3
        return packed, scale, alpha

    return nvfp4(1856, 2688), nvfp4(2688, 1856)


def _route_ids(rows, generator, *, experts=128, topk=6):
    scores = torch.rand(rows, experts, generator=generator, device="cuda")
    return scores.topk(topk, dim=-1).indices.to(torch.int32)


def _serving_rank(stacks, x, ids, routes, rank, ranks=4):
    """One EP rank of vLLM's FlashInferCuteDSLW4A16Experts under BI."""
    from flashinfer.fused_moe.cute_dsl.tuner import CuteDslFusedMoEW4A16Runner
    from flashinfer.tllm_enums import ActivationType
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
        BATCH_INVARIANT_TACTIC,
        prepare_w4a16_scales,
    )

    local = 128 // ranks
    s = slice(rank * local, (rank + 1) * local)
    (w1, s1, a1), (w2, s2, a2) = stacks
    runner = CuteDslFusedMoEW4A16Runner(
        num_experts=128, top_k=6, num_local_experts=local,
        local_expert_offset=rank * local, use_fused_finalize=False,
        output_dtype=torch.bfloat16, activation_type=ActivationType.Relu2.value,
    )
    out = torch.empty_like(x)
    runner.forward(
        [x, ids, routes, w1[s], prepare_w4a16_scales(s1[s]), a1[s], w2[s],
         prepare_w4a16_scales(s2[s]), a2[s], out],
        tactic=(BATCH_INVARIANT_TACTIC, BATCH_INVARIANT_TACTIC),
    )
    return out


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("rows", [1, 7, 64, 65, 513, 4096])
def test_cutedsl_rank_partials_equal_the_serving_runner_bitwise(rows, monkeypatch):
    """The actor's one-launch composition gives each EP rank's serving output,
    and the EP1 (all-expert) output, bitwise."""
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import CuteDslRoutedExperts

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    g = torch.Generator(device="cuda").manual_seed(rows)
    stacks = _stacks(g)
    experts = CuteDslRoutedExperts(*stacks, num_experts=128)
    x = (torch.randn(rows, 2688, generator=g, device="cuda") * 0.5).bfloat16()
    ids = _route_ids(rows, g)
    routes = torch.softmax(torch.randn(rows, 6, generator=g, device="cuda"), -1)
    parts, fc1, visible, activated = experts.ep_partials(
        x, routes, ids, return_fc1=True
    )
    assert all(
        torch.equal(a, b) for a, b in zip(parts, experts.ep_partials(x, routes, ids))
    )
    for rank in range(4):
        assert torch.equal(parts[rank], _serving_rank(stacks, x, ids, routes, rank)), rank
    # An EP rank's own deployment (its 32 experts, as under EP4 training):
    # the same partial, and the saved per-route tensors of its slots.
    for rank in range(4):
        s = slice(rank * 32, (rank + 1) * 32)
        local = CuteDslRoutedExperts(
            *(tuple(t[s] for t in stack) for stack in stacks),
            num_experts=128, offset=rank * 32,
        )
        part, *saved = local.rank_partial(x, routes, ids, save=True)
        assert torch.equal(part, parts[rank]), rank
        assert torch.equal(local.rank_partial(x, routes, ids), part), rank
        mine = ((ids // 32) == rank).reshape(-1)
        for got, ref in zip(saved, (fc1, visible, activated), strict=True):
            assert torch.equal(got, ref[mine]), rank
    full = experts.ep_partials(x, routes, ids, ranks=1)[0]
    assert torch.equal(full, _serving_rank(stacks, x, ids, routes, 0, ranks=1))
    # The saved per-route output is the one the partials combine: where a
    # rank owns a single slot of a token, its partial is that slot's output
    # times the route weight.
    owner = (ids // 32).long()
    per_route = visible.view(rows, 6, 2688).float() * routes[..., None]
    for rank in range(4):
        owned = owner == rank
        single = owned.sum(-1) == 1
        slot = owned.float().argmax(-1)
        expected = per_route[torch.arange(rows, device="cuda"), slot].bfloat16()
        assert torch.equal(expected[single], parts[rank][single]), rank
    assert fc1.shape == activated.shape == (rows * 6, 1856)
    # The saved fused activation is relu(a)^2 of an FP32 a that rounds to fc1:
    # it lies between the activations of fc1's BF16 rounding-interval ends.
    u = fc1.float()
    ulp = 2.0 ** (torch.floor(torch.log2(u.abs().clamp_min(2.0**-126))) - 7)
    lo = ((u - ulp / 2).clamp_min(0) ** 2).bfloat16().float()
    hi = ((u + ulp / 2).clamp_min(0) ** 2).bfloat16().float()
    h = activated.float()
    assert bool(((h >= lo) & (h <= hi)).all())


def test_export_rounds_kv_scales_to_bf16():
    """vLLM loads KV scales into BF16 at first load and FP32 on reload; the
    exporter sends BF16-representable FP32 so both hold the same value."""
    from megatron.lite.model.nemotron_h.checkpoint import NemotronExport

    model = torch.nn.Module()
    attention = torch.nn.Module()
    attention.register_buffer("k_scale", torch.tensor(0.0123456789))
    attention.register_buffer("v_scale", torch.tensor([0.0987654321]))
    model.add_module("layers", torch.nn.ModuleDict({"0": torch.nn.Module()}))
    model.layers["0"].add_module("mixer", torch.nn.Module())
    model.layers["0"].mixer.add_module("kv_attention", attention)
    exported = dict(NemotronExport.iter_export_tensors(None, model))
    for name, source in (("k", attention.k_scale), ("v", attention.v_scale)):
        value = exported[f"backbone.layers.0.mixer.{name}_proj.{name}_scale"]
        assert value.dtype == torch.float32 and value.shape == source.shape
        assert torch.equal(value, source.bfloat16().float())
        assert not torch.equal(value, source)


# Per-rank source rows; a source rank with no tokens, and (last case) routes
# that avoid rank 3's experts, so it receives no rows.
EP4_ROWS = ((513, 7, 64, 1), (8192, 4096, 1, 2048), (0, 7, 64, 1), (33, 5, 0, 17))


def _gather_rows(tensor):
    import torch.distributed as dist

    sizes = [None] * 4
    dist.all_gather_object(sizes, tensor.shape[0])
    out = [tensor.new_empty(n, *tensor.shape[1:]) for n in sizes]
    dist.all_gather(out, tensor.contiguous())
    return torch.cat(out), sum(sizes[: dist.get_rank()])


def _ep4_deepep_worker(rank):
    from types import SimpleNamespace

    import torch.distributed as dist
    from megatron.lite.model.nemotron_h.vllm.primitive.moe import communication as ep
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import (
        CuteDslRoutedExperts,
        ep4_routed_experts,
        routed_vjp,
    )

    g = torch.Generator(device="cuda").manual_seed(2)
    up = (torch.randn(128, 1856, 2688, generator=g, device="cuda") * 0.02).bfloat16()
    down = (torch.randn(128, 2688, 1856, generator=g, device="cuda") * 0.02).bfloat16()
    mine_experts = slice(rank * 32, (rank + 1) * 32)
    # theta0 and two updated snapshots; the deployment follows the masters.
    for snapshot in range(3):
        if snapshot:
            up = up + (torch.randn(up.shape, generator=g, device="cuda") * 1e-3).bfloat16()
            down = down + (torch.randn(down.shape, generator=g, device="cuda") * 1e-3).bfloat16()
        stacks = {}
        for stem, master in (("w13", up), ("w2", down)):
            parts = [requantize("W4A16_NVFP4", master[e]) for e in range(128)]
            stacks[stem] = tuple(
                torch.stack([q[name] for q in parts]).float()
                if name == "weight_scale_2"
                else torch.stack([q[name] for q in parts])
                for name in ("weight", "weight_scale", "weight_scale_2")
            )
        mine_stacks = [tuple(t[mine_experts] for t in stacks[s]) for s in ("w13", "w2")]
        full = CuteDslRoutedExperts(stacks["w13"], stacks["w2"], num_experts=128)
        local = CuteDslRoutedExperts(*mine_stacks, num_experts=128, offset=rank * 32)
        up_local = up[mine_experts].clone().requires_grad_()
        down_local = down[mine_experts].clone().requires_grad_()
        owner = SimpleNamespace(
            _experts=local, ep_group=dist.group.WORLD,
            weights=SimpleNamespace(_versions=lambda: 0),
        )
        for rows_per_rank in EP4_ROWS if snapshot == 0 else EP4_ROWS[:1]:
            _check_ep4_rows(rank, rows_per_rank, ep, owner, full, up, down, up_local,
                            down_local, mine_experts, ep4_routed_experts, routed_vjp)


def _check_ep4_rows(rank, rows_per_rank, ep, owner, full, up, down, up_local,
                    down_local, mine_experts, ep4_routed_experts, routed_vjp):
    rows = rows_per_rank[rank]
    mine = torch.Generator(device="cuda").manual_seed(100 + rank)
    x = torch.randn(rows, 2688, generator=mine, device="cuda").to(torch.bfloat16)
    experts = 96 if rows_per_rank is EP4_ROWS[-1] else 128
    ids = _route_ids(rows, mine, experts=experts)
    weights = torch.rand(rows, 6, generator=mine, device="cuda")
    expected = ep4_routed_experts(full, x, weights, ids) if rows else x.clone()
    with torch.no_grad():
        actual = ep._forward(owner._experts, owner.ep_group, x, ids, weights)[0]
    assert torch.equal(actual, expected), (rank, rows)

    dy = torch.randn(rows, 2688, generator=mine, device="cuda").to(torch.bfloat16)
    grads = []
    for _ in range(2):
        xs, ws = x.clone().requires_grad_(), weights.clone().requires_grad_()
        out = ep.EPRoutedExpertsVJP.apply(xs, up_local, down_local, ws, ids, owner)
        assert torch.equal(out, expected)
        grads.append(torch.autograd.grad(out, (xs, up_local, down_local, ws), dy))
    # A retained graph runs the backward again from the same saved state.
    xs, ws = x.clone().requires_grad_(), weights.clone().requires_grad_()
    out = ep.EPRoutedExpertsVJP.apply(xs, up_local, down_local, ws, ids, owner)
    inputs = (xs, up_local, down_local, ws)
    grads.append(torch.autograd.grad(out, inputs, dy, retain_graph=True))
    grads.append(torch.autograd.grad(out, inputs, dy))
    for other in grads[1:]:
        for a, b in zip(grads[0], other, strict=True):
            assert torch.equal(a, b), "EP4 backward is not run-to-run deterministic"

    # Single-rank reference on the whole EP batch.
    all_x, start = _gather_rows(x)
    all_ids, _ = _gather_rows(ids)
    all_w, _ = _gather_rows(weights)
    all_dy, _ = _gather_rows(dy)
    _, fc1, visible, activated = full.ep_partials(all_x, all_w, all_ids, return_fc1=True)
    dx, d_up, d_down, d_w = routed_vjp(
        all_x, fc1, visible.view(-1, 2688), up, down, all_w, all_ids, all_dy, activated
    )
    mine_rows = slice(start, start + rows)
    expect = (dx[mine_rows], d_up[mine_experts], d_down[mine_experts], d_w[mine_rows])
    # Route gradients come from the visible outputs, and the input gradient
    # adds the same per-route rows in the same order: both bitwise as EP1.
    assert torch.equal(grads[0][3], expect[3])
    assert torch.equal(grads[0][0], expect[0]), (rank, rows_per_rank)
    # Only rounding differs: the wgrad GEMMs see other row groupings.
    for name, a, b in zip(("d_up", "d_down"), grads[0][1:3], expect[1:3]):
        if b.float().norm() == 0:
            assert a.float().norm() == 0, name
            continue
        error = ((a.float() - b.float()).norm() / b.float().norm()).item()
        assert error < 1e-2, (name, error)


@cuda
@pytest.mark.gpus(4, min_architecture="blackwell")
def test_ep4_deepep_routed_experts_match_serving_reduction_bitwise(ep4, monkeypatch):
    """Real EP4 over DeepEP: forward equals the single-rank EP4 serving
    reduction bitwise; backward is deterministic and matches the EP1 VJP.
    Each rank launches CuTe-DSL over its own 32 experts, as a serving rank
    does."""
    pytest.importorskip("deep_ep")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    ep4(_ep4_deepep_worker)


def _flashinfer_combine_worker(rank):
    import torch.distributed as dist
    from flashinfer.comm import Mapping
    from flashinfer.comm.comm_backend import TorchDistBackend
    from flashinfer.comm.mnnvl import MnnvlConfig
    from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.communication import reduce_ep4_parts

    hidden, max_tokens = 2688, 1024
    a2a = MoeAlltoAll(
        Mapping(world_size=4, rank=rank, gpus_per_node=4, tp_size=4, moe_ep_size=4),
        max_num_tokens=max_tokens, top_k=6, num_experts=128, hidden_size=hidden,
        mnnvl_config=MnnvlConfig(comm_backend=TorchDistBackend(dist.group.WORLD)),
    )
    for rows in ((513, 7, 64, 1), (1024, 1024, 300, 1)):
        mine = rows[rank]
        g = torch.Generator(device="cuda").manual_seed(10 + rank)
        ids = _route_ids(mine, g)
        token = torch.arange(mine, dtype=torch.int32, device="cuda")[:, None]
        recv_ids, recv_token = a2a.dispatch(ids, [ids, token], max(rows))

        def partial(source, dest, tokens):
            # The BF16 partial rank `dest` returns for `tokens` of `source`.
            gen = torch.Generator(device="cuda").manual_seed(1000 * source + dest)
            table = torch.randn(rows[source], hidden, generator=gen, device="cuda")
            return table.to(torch.bfloat16)[tokens]

        payload = torch.zeros(4, max(rows), hidden, dtype=torch.bfloat16, device="cuda")
        for source in range(4):
            valid = (recv_ids[source] // 32 == rank).any(1)
            tokens = recv_token[source, :, 0].long().clamp(0, rows[source] - 1)
            payload[source][valid] = partial(source, rank, tokens)[valid]
        combined = a2a.combine(payload, max(rows))
        parts = [partial(rank, dest, torch.arange(mine, device="cuda")) for dest in range(4)]
        assert torch.equal(combined, reduce_ep4_parts(parts, ids))


@cuda
@pytest.mark.gpus(4, min_architecture="blackwell")
def test_flashinfer_one_sided_combine_matches_ep4_reduction_bitwise(ep4, monkeypatch):
    """Guard against a FlashInfer upgrade changing the serving combine order."""
    pytest.importorskip("flashinfer.comm.trtllm_moe_alltoall")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    ep4(_flashinfer_combine_worker)


def _fsdp_requant_worker(rank):
    from megatron.lite.model.nemotron_h.vllm.primitive.dense import (
        Fp8TrainingLinear,
        HummingNvfp4Linear,
        Nvfp4TrainingLinear,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    mesh = init_device_mesh("cuda", (4,))
    g = torch.Generator(device="cuda").manual_seed(4)
    master = (torch.randn(512, 2688, generator=g, device="cuda") * 0.02).bfloat16()
    for algorithm in ("W4A16_NVFP4", "FP8"):
        start = requantize(algorithm, master)
        if algorithm == "FP8":
            start["input_scale"] = torch.tensor([0.05], device="cuda")
            module = Fp8TrainingLinear(QuantizedWeight("FP8", start), device="cuda")
        else:
            module = Nvfp4TrainingLinear(
                QuantizedWeight(algorithm, start), HummingNvfp4Linear, device="cuda"
            )
        # The FP32 FSDP2 shard of the master, as fully_shard leaves it.
        module.weight = torch.nn.Parameter(
            distribute_tensor(module.weight.detach().float(), mesh, [Shard(0)])
        )
        module.bind_master()
        module.refresh_deployment()
        update = (torch.randn(512, 2688, generator=g, device="cuda") * 1e-3).float()
        with torch.no_grad():
            module.weight.add_(distribute_tensor(update, mesh, [Shard(0)]))
        # The optimizer's in-place update of the shard is seen as stale.
        with pytest.raises(RuntimeError, match="Refresh deployment"):
            module.export_quantized()
        module.refresh_deployment(recompute_scales=True)
        exported = module.export_quantized()
        expected = requantize(algorithm, module.weight.full_tensor().bfloat16())
        for name, tensor in expected.items():
            assert torch.equal(
                exported[name].reshape(-1).view(torch.uint8),
                tensor.reshape(-1).view(torch.uint8),
            ), (algorithm, name)


@cuda
@pytest.mark.gpus(4, min_architecture="blackwell")
def test_fsdp2_shard_requantizes_from_the_full_matrix(ep4, monkeypatch):
    """Under FSDP2 a quantized layer tracks its FP32 shard: an optimizer update
    makes it stale, and the refresh requantizes the gathered BF16 matrix (the
    one the unsharded forward sees), so its scales come from the full matrix."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    ep4(_fsdp_requant_worker)
