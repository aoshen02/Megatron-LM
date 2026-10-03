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
def test_fp8_attention_vjp_within_bf16_noise_floor(kind, monkeypatch):
    """Lightning Q32/KV2/D128 vs per-request SDPA on the dequantized Q/K/V.

    The VJP uses the visible output and LSE (DS4 semantics); the visible P@V
    is FP8, which puts dQ/dK about 3x the BF16 noise floor of a BF16 forward.
    """
    from megatron.lite.model.nemotron_h.fp8_attention import Fa4Fp8KVAttention
    from megatron.lite.model.nemotron_h.mamba import SSMMeta
    from test_nemotron_mamba_unit import (
        LENGTHS,
        assert_within_noise_floor,
        per_request,
        upstream_gradient,
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(42)
    meta = SSMMeta((0, *torch.tensor(LENGTHS).cumsum(0).tolist()))
    module = Fa4Fp8KVAttention(
        32, 2, 128, torch.tensor(0.02, device="cuda"), torch.tensor(0.01, device="cuda")
    )
    q, k, v = (
        torch.randn(
            meta.boundaries[-1], heads, 128, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for heads in (32, 2, 2)
    )
    output = module(q, k, v, meta)
    with torch.no_grad():
        serving, _, *references = module._visible(q, k, v, meta.boundaries)
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
    assert_within_noise_floor(("dq", "dk", "dv"), actual, old, reference, kind, ratio=4.0)


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


def _old_rms(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return weight * y.to(x.dtype)


def _old_residual_rms(x, residual, weight, eps):
    y = x.float() + residual.float()
    residual_out = y.to(weight.dtype)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return y.to(weight.dtype) * weight, residual_out


def _old_gated_rms(x, gate, weight, group_size, eps):
    y = x.float() * torch.nn.functional.silu(gate.float())
    groups = y.unflatten(-1, (-1, group_size))
    groups = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + eps)
    return weight * groups.flatten(-2).to(x.dtype)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
@pytest.mark.parametrize("norm", ["rms", "residual_rms", "gated_rms"])
def test_rms_norm_vjp_within_bf16_noise_floor(norm, kind):
    """Lightning widths: hidden 2688; Mamba inner 4096 in 8 groups."""
    from megatron.lite.model.nemotron_h.functional import GatedRMSNorm, RMSNorm
    from test_nemotron_mamba_unit import assert_within_noise_floor, upstream_gradient

    torch.manual_seed(0)
    eps, tokens = 1e-5, 1506
    width = 4096 if norm == "gated_rms" else 2688
    module = (
        GatedRMSNorm(width, 512, eps, device="cuda", dtype=torch.bfloat16)
        if norm == "gated_rms"
        else RMSNorm(width, eps, device="cuda")
    )
    with torch.no_grad():
        module.weight.copy_(torch.empty(width).uniform_(0.5, 1.5))
    leaves = [torch.randn(tokens, width, device="cuda") for _ in range(2)]
    leaves = [*(leaves[:1] if norm == "rms" else leaves), module.weight.float()]

    def old(*inputs):
        if norm == "rms":
            return _old_rms(*inputs, eps)
        if norm == "residual_rms":
            return torch.cat(_old_residual_rms(*inputs, eps), -1)
        return _old_gated_rms(*inputs, 512, eps)

    def cast(dtype):
        return [t.detach().to(dtype).requires_grad_() for t in leaves]

    inputs = [*cast(torch.bfloat16)[:-1], module.weight]
    with torch.no_grad():
        visible = module(*inputs[:-1])
    output = module(*inputs[:-1])
    if norm == "residual_rms":
        assert all(map(torch.equal, output, visible))
        output = torch.cat(output, -1)
    else:
        assert torch.equal(output, visible)
    upstream = upstream_gradient(kind, output)
    actual = torch.autograd.grad(output, inputs, upstream)
    old_inputs, reference_inputs = cast(torch.bfloat16), cast(torch.float64)
    expected = torch.autograd.grad(old(*old_inputs), old_inputs, upstream)
    reference = torch.autograd.grad(
        old(*reference_inputs), reference_inputs, upstream.double()
    )
    names = {"rms": ("dx",), "residual_rms": ("dx", "dresidual")}
    names = (*names.get(norm, ("dx", "dgate")), "dweight")
    assert_within_noise_floor(names, actual, expected, reference, kind)


@pytest.mark.gpus(1)
def test_parameter_mutation_before_backward_is_rejected():
    from megatron.lite.model.nemotron_h.functional import RMSNorm

    norm = RMSNorm(64, 1e-5, device="cuda")
    output = norm(torch.randn(4, 64, device="cuda", dtype=torch.bfloat16))
    with torch.no_grad():
        norm.weight.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def _lightning_router():
    from types import SimpleNamespace

    from megatron.lite.model.nemotron_h.experts import Router

    config = SimpleNamespace(
        n_routed_experts=128,
        hidden_size=2688,
        num_experts_per_tok=6,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
    )
    return Router(config, device="cuda")


@pytest.mark.gpus(1)
def test_router_requires_batch_invariance_initialized(monkeypatch):
    """The FP32-output router GEMM is M-invariant only after init_batch_invariance."""
    from vllm.model_executor.determinism import batch_invariant

    monkeypatch.setattr(batch_invariant, "_batch_invariant_MODE", False)
    router = _lightning_router()
    with pytest.raises(RuntimeError, match="init_batch_invariance"):
        router(torch.zeros(4, 2688, device="cuda", dtype=torch.bfloat16))


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
def test_router_vjp_within_bf16_noise_floor(kind, monkeypatch):
    """Lightning router: 128 experts, top-6 sigmoid, renormalized, fixed ids."""
    from test_nemotron_mamba_unit import assert_within_noise_floor, upstream_gradient
    from vllm.model_executor.determinism import batch_invariant

    # The VJP is measured against FP64; batch-invariant GEMM tiling is not.
    monkeypatch.setattr(batch_invariant, "_batch_invariant_MODE", True)
    torch.manual_seed(0)
    router = _lightning_router()
    with torch.no_grad():
        router.weight.normal_(std=0.02)
        router.e_score_correction_bias.normal_(std=0.01)
    leaves = (torch.randn(1506, 2688, device="cuda"), router.weight.float())
    inputs = [leaves[0].to(torch.bfloat16).requires_grad_(), router.weight]
    with torch.no_grad():
        visible_ids, visible = router(inputs[0])
    ids, weights = router(inputs[0])
    assert torch.equal(ids, visible_ids) and torch.equal(weights, visible)
    upstream = upstream_gradient(kind, weights)
    actual = torch.autograd.grad(weights, inputs, upstream)

    def old_vjp(dtype, logits_dtype):
        inputs = [t.detach().to(dtype).requires_grad_() for t in leaves]
        logits = torch.nn.functional.linear(*(t.to(logits_dtype) for t in inputs))
        selected = logits.sigmoid().gather(1, ids.long())
        selected = selected / selected.sum(-1, keepdim=True)
        return torch.autograd.grad(selected, inputs, upstream.to(logits_dtype))

    old = old_vjp(torch.bfloat16, torch.float32)
    reference = old_vjp(torch.float64, torch.float64)
    assert_within_noise_floor(("dx", "dweight"), actual, old, reference, kind)


def test_moe_combine_vjp_matches_autograd_bitwise():
    from megatron.lite.model.nemotron_h.experts import _CombineVJP

    def combine(shared, routed):
        return shared + routed * 2.5

    torch.manual_seed(0)
    inputs = [
        torch.randn(37, 64, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)
    ]
    upstream = torch.randn(37, 64, dtype=torch.bfloat16)
    actual = torch.autograd.grad(
        _CombineVJP.apply(combine, *inputs, 2.5), inputs, upstream
    )
    expected = torch.autograd.grad(combine(*inputs), inputs, upstream)
    assert all(map(torch.equal, actual, expected))


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_selected_log_probs_vjp_within_bf16_noise_floor(temperature, kind):
    """Chunked LM head + selected log-prob: rollout value unchanged, FP32 VJP."""
    from megatron.lite.model.nemotron_h.logprob import aligned_selected_log_probs
    from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
    from test_nemotron_mamba_unit import assert_within_noise_floor, upstream_gradient

    from vllm.model_executor.determinism.batch_invariant import (
        init_batch_invariance,
        linear_batch_invariant,
    )
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    init_batch_invariance()
    torch.manual_seed(0)
    tokens, vocab = 1506, 131072
    lm_head = torch.nn.Linear(2688, vocab, bias=False, device="cuda")
    with torch.no_grad():
        lm_head.weight.normal_(std=0.02)
    lm_head = lm_head.to(torch.bfloat16)
    hidden = torch.randn(tokens, 2688, device="cuda")
    labels = torch.randint(vocab, (tokens,), device="cuda")
    leaves = (hidden, lm_head.weight.float())
    inputs = [hidden.to(torch.bfloat16).requires_grad_(), lm_head.weight]
    log_probs, entropy = aligned_selected_log_probs(
        inputs[0],
        lm_head,
        labels,
        temperature,
        512,
        calculate_entropy=True,
        tp_group=None,
    )
    with torch.no_grad():
        logits = linear_batch_invariant(inputs[0], lm_head.weight)
        if temperature != 1.0:
            # vLLM's sampler: FP32 copy of the logits, then the temperature
            # (its processed_logprobs; the recipe's raw_logprobs uses T=1).
            logits = logits.float() / temperature
        assert torch.equal(
            log_probs, compute_token_logprobs(logits, labels[:, None])[:, 0]
        )
        # Entropy is training-only; its reductions see fewer rows per chunk.
        torch.testing.assert_close(
            entropy, vocab_parallel_entropy(logits), rtol=1e-6, atol=0
        )
    upstream = upstream_gradient(kind, log_probs)
    actual = torch.autograd.grad(log_probs, inputs, upstream)

    def old_vjp(dtype):
        inputs = [t.detach().to(dtype).requires_grad_() for t in leaves]
        logits = torch.nn.functional.linear(*inputs)
        if temperature != 1.0:
            logits = logits / temperature
        logits = logits.to(torch.promote_types(dtype, torch.float32))
        selected = logits.log_softmax(-1).gather(-1, labels[:, None])
        return torch.autograd.grad(selected[:, 0], inputs, upstream.to(selected.dtype))

    old, reference = old_vjp(torch.bfloat16), old_vjp(torch.float64)
    assert_within_noise_floor(("dhidden", "dweight"), actual, old, reference, kind)


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_fp8_attention_vjp_is_run_to_run_bitwise(monkeypatch):
    """full_determinism: the FlashAttention backward accumulates dQ in a fixed order."""
    from megatron.lite.model.nemotron_h.fp8_attention import Fa4Fp8KVAttention
    from megatron.lite.model.nemotron_h.mamba import SSMMeta

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(7)
    lengths = (8192, 3000, 1, 5000)
    meta = SSMMeta((0, *torch.tensor(lengths).cumsum(0).tolist()))
    module = Fa4Fp8KVAttention(
        32, 2, 128, torch.tensor(0.02, device="cuda"), torch.tensor(0.01, device="cuda")
    )
    q, k, v = (
        torch.randn(meta.boundaries[-1], h, 128, device="cuda", dtype=torch.bfloat16)
        .requires_grad_()
        for h in (32, 2, 2)
    )
    upstream = torch.randn(meta.boundaries[-1], 32, 128, device="cuda", dtype=torch.bfloat16)
    first, second = (
        torch.autograd.grad(module(q, k, v, meta), (q, k, v), upstream) for _ in range(2)
    )
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("init_first", [True, False])
def test_router_gemm_row_check_detects_a_missing_batch_invariance_init(init_first):
    """build_model's probe passes when init_batch_invariance precedes the first
    cuBLAS call and fails in a process that never ran it."""
    import os
    import subprocess
    import sys

    script = (
        "import torch\n"
        "from vllm.model_executor.determinism.batch_invariant import init_batch_invariance\n"
        "from megatron.lite.model.nemotron_h.protocol import "
        "_check_router_gemm_rows_invariant as check\n"
        f"{'init_batch_invariance()' if init_first else ''}\n"
        "torch.cuda.set_device(0)\n"
        "check()\n"
    )
    env = dict(os.environ, VLLM_BATCH_INVARIANT="1" if init_first else "0")
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    if init_first:
        assert result.returncode == 0, result.stderr[-2000:]
    else:
        assert "depends on the row count" in result.stderr, result.stderr[-2000:]


@pytest.mark.gpus(1)
def test_fp8_attention_cache_pages_follow_request_lengths(monkeypatch):
    """1024 one-token requests and one of 7000 tokens take 1026 cache pages of
    3.3 MiB (3.4 GiB; three pages per request alone would be 10 GiB; FA4's
    split scratch adds about 4 GiB), and each request's output is the one it
    gets alone."""
    from megatron.lite.model.nemotron_h.fp8_attention import Fa4Fp8KVAttention

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(3)
    module = Fa4Fp8KVAttention(
        32, 2, 128, torch.tensor(0.02, device="cuda"), torch.tensor(0.01, device="cuda")
    )
    lengths = [1] * 1024 + [7000]
    bounds = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    q, k, v = (
        torch.randn(bounds[-1], h, 128, device="cuda", dtype=torch.bfloat16)
        for h in (32, 2, 2)
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    packed = module._visible(q, k, v, bounds)[0]
    assert torch.cuda.max_memory_allocated() - base < 9 * 2**30
    for start, end in ((0, 1), (1023, 1024), (1024, bounds[-1])):
        alone = module._visible(q[start:end], k[start:end], v[start:end], [0, end - start])
        assert torch.equal(packed[start:end], alone[0])
