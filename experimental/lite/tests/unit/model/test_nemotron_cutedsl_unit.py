"""FlashInfer CuTe-DSL W4A16 routed experts called directly by the actor.

``kernels.CuteDslRoutedExperts`` re-composes FlashInfer's
``launch_w4a16_moe`` from its stages and two private helpers. These tests
pin the helpers and the composition they mirror, and check the composition
bitwise against the serving runner.
"""

import hashlib
import inspect

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# flashinfer-python 0.7.0.post1, flashinfer/fused_moe/cute_dsl/blackwell/moe_w4a16.py
# (_W4A16Workspace :49, launch_w4a16_moe :358)
PINNED_SIGNATURES = {
    # :59
    "_get_workspace": (
        "x", "top_k", "num_experts", "num_local_experts", "intermediate_size",
        "route_tile",
    ),
    # :195
    "_run_grouped_gemm": (
        "weight", "weight_sf", "activations", "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit", "num_non_exiting_tiles", "alpha", "output",
        "num_local_experts", "activation_type", "swiglu_alpha", "swiglu_beta",
        "swiglu_limit", "situ_beta", "situ_linear_beta", "use_fused_finalize",
        "permuted_idx_to_expanded_idx", "token_final_scales", "enable_pdl", "tactic",
    ),
}
# Source of the private helpers and of the launcher whose stage order the
# actor mirrors; any change needs a re-review of CuteDslRoutedExperts.
PINNED_SOURCE_SHA256 = {
    "_W4A16Workspace": "9a4534c8fd89a1093c3007874c77f39cfb200bdc74036f0f6f6a05d9e73aed87",
    "_get_workspace": "c5acedc5bb61dffbb894aba3d015ade3c301886d9e671cb08ac7cd09235b1fa4",
    "_run_grouped_gemm": "430265499f90cf3552e2dfbf2bc4d58649d6d814423563843579bd55df771fc1",
    "launch_w4a16_moe": "ae7abf3e50ea6ef411240f602a9b90980f8994faaefe10e47c5726c3c9abbc55",
}


def _module():
    return pytest.importorskip("flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16")


def test_flashinfer_private_w4a16_helpers_keep_their_signatures():
    module = _module()
    for name, parameters in PINNED_SIGNATURES.items():
        actual = tuple(inspect.signature(getattr(module, name)).parameters)
        assert actual == parameters, f"FlashInfer {name} signature changed: {actual}"
    workspace_fields = tuple(module._W4A16Workspace.__dataclass_fields__)
    assert workspace_fields == ("moe_sort_buffers", "hidden_workspace", "intermediate")


def test_actor_runtime_guard_pins_the_same_flashinfer_w4a16_helpers():
    """The actor checks the same pins at construction (check_flashinfer_w4a16)."""
    _module()
    from megatron.lite.model.nemotron_h import kernels

    assert kernels.FLASHINFER_W4A16_SIGNATURES == PINNED_SIGNATURES
    assert kernels.FLASHINFER_W4A16_SOURCE_SHA256 == PINNED_SOURCE_SHA256
    kernels.check_flashinfer_w4a16()


def test_actor_runtime_guard_rejects_a_changed_helper(monkeypatch):
    module = _module()
    from megatron.lite.model.nemotron_h import kernels

    def _run_grouped_gemm(weight, weight_sf, activations):  # noqa: ARG001
        raise AssertionError

    monkeypatch.setattr(module, "_run_grouped_gemm", _run_grouped_gemm)
    monkeypatch.setattr(kernels, "_FLASHINFER_W4A16_CHECKED", False)
    with pytest.raises(RuntimeError, match="_run_grouped_gemm"):
        kernels.check_flashinfer_w4a16()


def test_flashinfer_w4a16_launcher_source_is_the_reviewed_one():
    module = _module()
    for name, digest in PINNED_SOURCE_SHA256.items():
        source = inspect.getsource(getattr(module, name))
        actual = hashlib.sha256(source.encode()).hexdigest()
        assert actual == digest, (
            f"FlashInfer {name} changed (sha256 {actual}); re-review "
            "megatron.lite.model.nemotron_h.kernels.CuteDslRoutedExperts"
        )


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
    from megatron.lite.model.nemotron_h.kernels import CuteDslRoutedExperts

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


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_routed_vjp_on_the_cutedsl_deployment_at_updated_weights(monkeypatch):
    """The training VJP on the CuTe-DSL deployment, at theta0 and two updated
    snapshots requantized from the updated masters (as the Humming test)."""
    from megatron.lite.model.nemotron_h.kernels import CuteDslRoutedExperts
    from megatron.lite.model.nemotron_h.nvfp4_ep4 import (
        EP4_ONESIDED_REDUCTION,
        ep4_routed_experts,
    )
    from megatron.lite.model.nemotron_h.nvfp4_moe_vjp import routed_vjp
    from megatron.lite.model.nemotron_h.quantization import requantize

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
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
                torch.stack([q[name] for q in parts]).float()
                if name == "weight_scale_2"
                else torch.stack([q[name] for q in parts])
                for name in ("weight", "weight_scale", "weight_scale_2")
            ))
        experts = CuteDslRoutedExperts(*stacks, num_experts=128)
        out, fc1, visible, activated = ep4_routed_experts(
            experts, x, routes, ids, EP4_ONESIDED_REDUCTION, return_fc1=True,
            return_activated=True,
        )
        assert torch.equal(
            out, ep4_routed_experts(experts, x, routes, ids, EP4_ONESIDED_REDUCTION)
        )
        got = routed_vjp(x, fc1, visible, up, down, routes, ids, dy, activated)
        ref = [t.detach().float().requires_grad_() for t in (x, up, down, routes)]
        idx = ids.long()
        u = torch.einsum("mk,msik->msi", ref[0], ref[1][idx])
        u = u + (fc1.view(rows, 6, -1).float() - u).detach()
        a = u.relu().square()
        a = a + (activated.view(rows, 6, -1).float() - a).detach()
        v = torch.einsum("msi,mski->msk", a, ref[2][idx])
        v = v + (visible.view(rows, 6, -1).float() - v).detach()
        (ref[3][..., None] * v).sum(1).backward(dy.float())
        expected = [t.grad for t in ref]
        torch.testing.assert_close(got[3], expected[3], rtol=1e-5, atol=1e-4)
        for name, actual, reference in zip(("dx", "d_up", "d_down"), got, expected):
            error = ((actual.float() - reference).norm() / reference.norm()).item()
            assert error < 1e-2, (snapshot, name, error)
