"""Request boundaries, native VJP ownership, and packed conv/SSD replay."""

import pytest
import torch
from megatron.lite.model.nemotron_h.functional import visible_forward
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


def test_visible_value_and_native_gradient_with_frozen_input():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    weight = torch.tensor([4.0, 5.0])
    native = lambda a, b: a.square() * b
    visible = lambda a, b: native(a, b) + 0.125
    result = visible_forward(visible, native, x, weight)
    assert torch.equal(result, visible(x, weight))
    upstream = torch.tensor([0.5, -2.0])
    assert torch.equal(
        torch.autograd.grad(result, x, upstream)[0],
        torch.autograd.grad(native(x, weight), x, upstream)[0],
    )


def test_parameter_mutation_before_backward_is_rejected():
    x = torch.tensor([2.0], requires_grad=True)
    result = visible_forward(torch.square, torch.square, x)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


@pytest.mark.parametrize("requires_grad", [False, True])
def test_scoring_does_not_call_native_backward_reference(requires_grad):
    x = torch.tensor([2.0], requires_grad=requires_grad)

    def native(_):
        raise AssertionError("Scoring must not build backward intermediates")

    with torch.no_grad():
        assert torch.equal(visible_forward(torch.square, native, x), x.square())


@pytest.mark.gpus(1)
@pytest.mark.parametrize("lengths", [(17, 19), (127, 129)])
def test_packed_conv_matches_independent_requests_and_native_vjp(lengths):
    from megatron.lite.model.nemotron_h.mamba import packed_conv
    from transformers.models.nemotron_h.modeling_nemotron_h import causal_conv1d_fn

    causal_conv1d_fn = getattr(causal_conv1d_fn, "__wrapped__", causal_conv1d_fn)
    torch.manual_seed(42)
    first, second = lengths
    meta = SSMMeta((0, first, first + second))
    x = torch.randn(
        first + second, 6144, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        6144, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(6144, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = packed_conv(x, weight, bias, meta)
    independent = torch.cat(
        [
            packed_conv(x[a:b], weight, bias, SSMMeta((0, b - a)))
            for a, b in zip(meta.boundaries, meta.boundaries[1:])
        ]
    )
    assert torch.equal(output, independent)
    native = torch.cat(
        [
            causal_conv1d_fn(x[a:b].T.unsqueeze(0), weight, bias, activation="silu")
            .squeeze(0)
            .T
            for a, b in zip(meta.boundaries, meta.boundaries[1:])
        ]
    )
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, (x, weight, bias), upstream)
    expected = torch.autograd.grad(native, (x, weight, bias), upstream)
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)


@pytest.mark.parametrize("length", [1, 37, 256, 300, 1300])
def test_chunked_ssd_reference_matches_transformers_scan_bitwise(length):
    """The chunk-at-a-time contractions keep HF native SSD bits and VJP."""
    from megatron.lite.model.nemotron_h.ssd_reference import chunk_scan
    from transformers.models.nemotron_h.modeling_nemotron_h import mamba2_chunk_scan

    hf_scan = getattr(mamba2_chunk_scan, "__wrapped__", mamba2_chunk_scan)
    torch.manual_seed(0)

    def leaves():
        g = torch.Generator().manual_seed(1)
        x = torch.randn(1, length, 4, 8, generator=g, dtype=torch.bfloat16)
        dt = torch.randn(1, length, 4, generator=g, dtype=torch.bfloat16)
        B = torch.randn(1, length, 2, 16, generator=g, dtype=torch.bfloat16)
        C = torch.randn(1, length, 2, 16, generator=g, dtype=torch.bfloat16)
        A = -torch.arange(1, 5, dtype=torch.float32)
        D = torch.linspace(0.5, 1.5, 4)
        bias = torch.full((4,), -1.0)
        return [t.requires_grad_() for t in (x, dt, A, B, C, D, bias)]

    def run(scan, x, dt, A, B, C, D, bias):
        return scan(
            x, dt, A, B, C, chunk_size=128, D=D, dt_bias=bias, dt_softplus=True,
            dt_limit=(0.0, float("inf")),
        )

    ours, theirs = leaves(), leaves()
    actual, expected = run(chunk_scan, *ours), run(hf_scan, *theirs)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    upstream = torch.randn(expected.shape, generator=torch.Generator().manual_seed(2))
    for a, b in zip(
        torch.autograd.grad(actual, ours, upstream),
        torch.autograd.grad(expected, theirs, upstream),
        strict=True,
    ):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("lengths", [(17, 19), (127, 129)])
def test_packed_ssd_matches_independent_requests_and_native_vjp(lengths):
    from megatron.lite.model.nemotron_h.mamba import packed_scan
    from transformers.models.nemotron_h.modeling_nemotron_h import mamba2_chunk_scan

    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    native_scan = getattr(mamba2_chunk_scan, "__wrapped__", mamba2_chunk_scan)
    init_batch_invariance()
    torch.manual_seed(42)
    first, second = lengths
    tokens = first + second
    meta = SSMMeta((0, first, tokens))

    def rand(*shape):
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

    x, dt = rand(tokens, 64, 64), rand(tokens, 64)
    A = (-torch.arange(1, 65, device="cuda", dtype=torch.float32)).requires_grad_()
    B, C = rand(tokens, 8, 128), rand(tokens, 8, 128)
    D = torch.ones(64, device="cuda", requires_grad=True)
    bias = torch.full((64,), -4.0, device="cuda", requires_grad=True)
    inputs = (x, dt, A, B, C, D, bias)
    output = packed_scan(*inputs, meta)
    independent, native = [], []
    for a, b in zip(meta.boundaries, meta.boundaries[1:]):
        independent.append(
            packed_scan(
                x[a:b], dt[a:b], A, B[a:b], C[a:b], D, bias, SSMMeta((0, b - a))
            )
        )
        native.append(
            native_scan(
                x[a:b].unsqueeze(0),
                dt[a:b].unsqueeze(0),
                A,
                B[a:b].unsqueeze(0),
                C[a:b].unsqueeze(0),
                chunk_size=128,
                D=D,
                dt_bias=bias,
                dt_softplus=True,
            ).squeeze(0)
        )
    assert torch.equal(output, torch.cat(independent))
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, inputs, upstream)
    expected = torch.autograd.grad(torch.cat(native), inputs, upstream)
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)
