"""Nemotron protocol loss normalization and FP8 attention VJP accuracy."""

import pytest
import torch


@pytest.mark.parametrize("mask", [[1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0] * 6])
def test_cp_loss_unequal_valid_counts_matches_unsharded_gradient(mask):
    from megatron.lite.model.nemotron_h.protocol import _token_mean_loss

    scores = -torch.arange(1.0, 7.0, requires_grad=True)
    weights = torch.tensor(mask, dtype=torch.float32)
    reference = -(scores * weights).sum() / weights.sum().clamp_min(1)
    # Reproduce Megatron's CP-averaged parameter gradient, including an empty rank.
    actual = (
        sum(
            _token_mean_loss(local, local_mask, weights, 2)
            for local, local_mask in zip(scores.chunk(2), weights.chunk(2), strict=True)
        )
        / 2
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference, scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_unquantized_checkpoint_is_rejected():
    from types import SimpleNamespace

    from megatron.lite.model.nemotron_h.protocol import ImplConfig, build_model

    with pytest.raises(ValueError, match="MIXED_PRECISION"):
        build_model(
            SimpleNamespace(quantization_config=None),
            impl_cfg=ImplConfig(hf_path="/nonexistent"),
        )


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
@pytest.mark.parametrize("attention", ["Fa4Fp8KVAttention", "Fp8KVAttention"])
def test_fp8_attention_vjp_within_bf16_noise_floor(attention, kind, monkeypatch):
    """Lightning Q32/KV2/D128 vs per-request SDPA on the dequantized Q/K/V."""
    from megatron.lite.model.nemotron_h import fp8_attention
    from megatron.lite.model.nemotron_h.mamba import SSMMeta
    from test_nemotron_mamba_unit import (
        LENGTHS,
        assert_within_noise_floor,
        per_request,
        upstream_gradient,
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from vllm.config import VllmConfig, set_current_vllm_config

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(42)
    meta = SSMMeta((0, *torch.tensor(LENGTHS).cumsum(0).tolist()))
    with set_current_vllm_config(VllmConfig()):
        module = getattr(fp8_attention, attention)(
            32,
            2,
            128,
            torch.tensor(0.02, device="cuda"),
            torch.tensor(0.01, device="cuda"),
        )
    q, k, v = (
        torch.randn(
            meta.boundaries[-1], heads, 128, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for heads in (32, 2, 2)
    )
    output = module(q, k, v, meta)
    with torch.no_grad():
        serving = module._visible(q, k, v, meta.boundaries)[0]
        _, *references = module._visible(q, k, v, meta.boundaries, return_query=True)
    assert torch.equal(output, serving)
    upstream = upstream_gradient(kind, output)
    actual = torch.autograd.grad(output, (q, k, v), upstream)

    def sdpa_vjp(dtype):
        def attend(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                *(x.transpose(0, 1)[None] for x in (q, k, v)),
                is_causal=True,
                scale=module.scale,
                enable_gqa=True,
            )[0].transpose(0, 1)

        inputs = [x.to(dtype).requires_grad_() for x in references]
        output = per_request(attend, meta, *inputs)
        return torch.autograd.grad(output, inputs, upstream.to(dtype))

    old = sdpa_vjp(torch.bfloat16)
    with sdpa_kernel(SDPBackend.MATH):
        reference = sdpa_vjp(torch.float64)
    assert_within_noise_floor(("dq", "dk", "dv"), actual, old, reference, kind)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
def test_bf16_linear_vjp_within_bf16_noise_floor(kind):
    """lm_head-like BF16 projection: batch-invariant forward, TE GEMM VJP."""
    from megatron.lite.model.nemotron_h.functional import linear
    from test_nemotron_mamba_unit import upstream_gradient, assert_within_noise_floor

    from vllm.model_executor.determinism.batch_invariant import (
        init_batch_invariance,
        linear_batch_invariant,
    )

    init_batch_invariance()
    torch.manual_seed(0)
    leaves = (
        torch.randn(1506, 2688, device="cuda"),
        torch.randn(8192, 2688, device="cuda") * 0.02,
    )

    def cast(dtype):
        return [t.to(dtype).requires_grad_() for t in leaves]

    inputs = cast(torch.bfloat16)
    output = linear(*inputs)
    with torch.no_grad():
        assert torch.equal(output, linear_batch_invariant(*inputs))
    upstream = upstream_gradient(kind, output)
    actual = torch.autograd.grad(output, inputs, upstream)
    old_inputs, reference_inputs = cast(torch.bfloat16), cast(torch.float64)
    old = torch.autograd.grad(
        torch.nn.functional.linear(*old_inputs), old_inputs, upstream
    )
    reference = torch.autograd.grad(
        torch.nn.functional.linear(*reference_inputs),
        reference_inputs,
        upstream.double(),
    )
    assert_within_noise_floor(("dx", "dweight"), actual, old, reference, kind)
