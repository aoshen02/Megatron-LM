"""Request boundaries and packed conv/SSD VJP accuracy."""

import pytest
import torch
from megatron.lite.model.nemotron_h.mamba import SSMMeta


def test_chunks_restart_at_every_request_not_packed_offset():
    meta = SSMMeta((0, 127, 256, 259))
    assert meta.chunks() == ((0, 127, 255, 256, 259), (0, 2, 3), (0, 1, 1, 2))
    with pytest.raises(ValueError, match="token count"):
        meta.validate_tokens(258)


@pytest.mark.parametrize("boundaries", [(1, 4), (0,), (0, 2, 2), (0, 3, 1)])
def test_invalid_request_boundaries_fail_closed(boundaries):
    with pytest.raises(ValueError):
        SSMMeta(boundaries)


# Uneven packed requests with non-chunk-aligned tails (chunk 128).
LENGTHS = (300, 77, 1000, 129)


def upstream_gradient(kind, like):
    generator = torch.Generator(device=like.device).manual_seed(7)
    grad = torch.randn(
        like.shape, generator=generator, device=like.device, dtype=like.dtype
    )
    if kind == "zero":
        return torch.zeros_like(like)
    if kind == "sparse":
        keep = torch.rand(like.shape, generator=generator, device=like.device) < 0.01
        return grad * keep
    return grad


def assert_within_noise_floor(names, actual, old, reference, kind, ratio=2.0):
    """New BF16 VJP error vs the high-precision reference stays within ``ratio``
    x the old BF16 error. References run in FP64 where cuBLAS/cuDNN could use
    TF32 (the image sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1)."""
    for name, a, b, r in zip(names, actual, old, reference, strict=True):
        assert torch.isfinite(a).all(), name
        if kind == "zero":
            assert not a.any() and not b.any(), name
            continue
        r = r.double()
        new_error = ((a.double() - r).norm() / r.norm()).item()
        old_error = ((b.double() - r).norm() / r.norm()).item()
        print(f"{kind:6} {name:8} new={new_error:.3e} old={old_error:.3e}")
        assert new_error <= ratio * old_error + 1e-5, (name, new_error, old_error)


def per_request(function, meta, *tensors):
    return torch.cat(
        [
            function(*(t[a:b] for t in tensors))
            for a, b in zip(meta.boundaries, meta.boundaries[1:])
        ]
    )


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
def test_packed_conv_vjp_within_bf16_noise_floor(kind):
    from megatron.lite.model.nemotron_h.mamba import packed_conv
    from transformers.models.nemotron_h.modeling_nemotron_h import causal_conv1d_fn

    hf_conv = getattr(causal_conv1d_fn, "__wrapped__", causal_conv1d_fn)
    torch.manual_seed(42)
    meta = SSMMeta((0, *torch.tensor(LENGTHS).cumsum(0).tolist()))
    leaves = (
        torch.randn(meta.boundaries[-1], 6144, device="cuda"),
        torch.randn(6144, 4, device="cuda") * 0.5,
        torch.randn(6144, device="cuda") * 0.1,
    )

    def cast(dtype):
        return [t.to(dtype).requires_grad_() for t in leaves]

    def native(x, weight, bias):
        def conv(x):
            return hf_conv(x.T[None], weight, bias, activation="silu")[0].T

        return per_request(conv, meta, x)

    inputs = cast(torch.bfloat16)
    output = packed_conv(*inputs, meta)
    independent = per_request(
        lambda x: packed_conv(x, *inputs[1:], SSMMeta((0, len(x)))), meta, inputs[0]
    )
    assert torch.equal(output, independent)
    upstream = upstream_gradient(kind, output)
    actual = torch.autograd.grad(output, inputs, upstream)
    old_inputs, reference_inputs = cast(torch.bfloat16), cast(torch.float64)
    old = torch.autograd.grad(native(*old_inputs), old_inputs, upstream)
    reference = torch.autograd.grad(
        native(*reference_inputs), reference_inputs, upstream.double()
    )
    assert_within_noise_floor(("dx", "dweight", "dbias"), actual, old, reference, kind)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
def test_packed_ssd_vjp_within_bf16_noise_floor(kind):
    """Lightning shapes: 64 heads x 64, 8 groups, state 128, chunk 128."""
    from megatron.lite.model.nemotron_h.mamba import packed_scan
    from transformers.models.nemotron_h.modeling_nemotron_h import mamba2_chunk_scan

    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    hf_scan = getattr(mamba2_chunk_scan, "__wrapped__", mamba2_chunk_scan)
    init_batch_invariance()
    torch.manual_seed(42)
    meta = SSMMeta((0, *torch.tensor(LENGTHS).cumsum(0).tolist()))
    tokens = meta.boundaries[-1]
    heads = 64
    leaves = (
        torch.randn(tokens, heads, 64, device="cuda"),
        torch.randn(tokens, heads, device="cuda"),
        -torch.empty(heads, device="cuda").uniform_(1, 16),
        torch.randn(tokens, 8, 128, device="cuda"),
        torch.randn(tokens, 8, 128, device="cuda"),
        torch.empty(heads, device="cuda").uniform_(0.5, 1.5),
        torch.empty(heads, device="cuda").uniform_(1e-3, 0.1).expm1().log(),
    )

    def cast(dtype):
        return [
            t.to(torch.float32 if i == 2 else dtype).requires_grad_()
            for i, t in enumerate(leaves)
        ]

    def native(x, dt, A, B, C, D, dt_bias):
        def scan(x, dt, B, C):
            return hf_scan(
                x[None],
                dt[None],
                A,
                B[None],
                C[None],
                chunk_size=128,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
            )[0]

        return per_request(scan, meta, x, dt, B, C)

    inputs = cast(torch.bfloat16)
    output = packed_scan(*inputs, meta)
    x, dt, A, B, C, D, dt_bias = inputs
    independent = per_request(
        lambda x, dt, B, C: packed_scan(
            x, dt, A, B, C, D, dt_bias, SSMMeta((0, len(x)))
        ),
        meta,
        x,
        dt,
        B,
        C,
    )
    assert torch.equal(output, independent)
    upstream = upstream_gradient(kind, output)
    actual = torch.autograd.grad(output, inputs, upstream)
    old_inputs, reference_inputs = cast(torch.bfloat16), cast(torch.float32)
    old = torch.autograd.grad(native(*old_inputs), old_inputs, upstream)
    reference = torch.autograd.grad(
        native(*reference_inputs), reference_inputs, upstream.float()
    )
    assert_within_noise_floor(
        ("dx", "ddt", "dA", "dB", "dC", "dD", "ddt_bias"),
        actual,
        old,
        reference,
        kind,
    )
