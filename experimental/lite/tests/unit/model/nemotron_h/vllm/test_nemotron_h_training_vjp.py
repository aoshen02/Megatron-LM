"""Nemotron-H training VJPs and the checkpoint quantization rule.

Every VJP is checked against an FP64 (or FP32) reference: its error must stay
within the BF16 noise floor of the plain BF16 computation. The requantization
must be the rule that produced the Lightning checkpoint, byte for byte.
"""

import pytest
import torch
from megatron.lite.model.nemotron_h.quantization import (
    QuantizedWeight,
    check_reversible,
    requantize,
)

from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import SSMMeta

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

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


def _deployment_chunk(nvfp4, fp8):
    from megatron.lite.model.nemotron_h.vllm.primitive.dense import (
        Fp8TrainingLinear,
        Nvfp4TrainingLinear,
    )

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


def _random_nvfp4(rows, columns, generator, *, global_scale=2e-3):
    packed = torch.randint(
        0, 256, (rows, columns // 2), generator=generator, device="cuda", dtype=torch.uint8
    )
    scale = (
        torch.rand(rows, columns // 16, generator=generator, device="cuda") * 3 + 0.25
    ).to(torch.float8_e4m3fn)
    return packed, scale, torch.tensor([global_scale], device="cuda")


@cuda
def test_checkpoint_load_redeploys_the_saved_bytes(monkeypatch, tmp_path):
    """A training-checkpoint load through the verl engine runs the
    post-optimizer-step hook (as DS4): the deployment is requant(master) from
    theta0 on, so requantizing the restored masters reproduces the bytes
    deployed when the checkpoint was saved, at theta0 and after an update."""
    from functools import partial
    from types import SimpleNamespace

    from megatron.lite.model.nemotron_h.quantization import fp8_encode
    from megatron.lite.model.nemotron_h.vllm.protocol import _post_optimizer_step
    from verl_mlite.engine import mlite_engine

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
    # theta0 deploys requant(master), as the BF16-master load does.
    _post_optimizer_step([trained])
    for step in range(2):
        if step:
            with torch.no_grad():
                for parameter in trained.parameters():
                    noise = torch.randn(parameter.shape, generator=g, device="cuda")
                    parameter.add_((noise * 1e-3).to(parameter.dtype))
            _post_optimizer_step([trained])
        saved, deployed = trained.state_dict(), _exported_bytes(trained)
        restored = _deployment_chunk(nvfp4, fp8)
        # The engine's load path with the checkpoint I/O reduced to the
        # module state it restores.
        monkeypatch.setattr(
            mlite_engine,
            "load_training_checkpoint",
            lambda *a, **k: restored.load_state_dict(saved, strict=False),
        )
        # Single process; other tests in this module may own a process group.
        monkeypatch.setattr(mlite_engine.dist, "is_initialized", lambda: False)
        engine = mlite_engine.MegatronLiteEngine.__new__(mlite_engine.MegatronLiteEngine)
        engine.runtime, engine.module = object(), restored
        engine.engine_config = SimpleNamespace(param_offload=False, optimizer_offload=False)
        engine.handle = SimpleNamespace(
            _extras={"post_optimizer_step_hook": partial(_post_optimizer_step, [restored])},
            _optimizer=None,
            _config=SimpleNamespace(parallel=None),
            _parallel_state=None,
            _lr_scheduler=None,
        )
        engine.load_checkpoint(str(tmp_path))
        exported = _exported_bytes(restored)
        assert exported.keys() == deployed.keys()
        for name, value in deployed.items():
            assert torch.equal(exported[name], value), (step, name)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
def test_fp8_attention_vjp_within_bf16_noise_floor(kind, monkeypatch):
    """Lightning Q32/KV2/D128 vs per-request SDPA on the dequantized Q/K/V.

    The VJP recomputes the BF16 output for D = rowsum(dO * O); the visible
    output (FP8 P@V) would put dQ/dK about 3x the BF16 noise floor.
    """
    from megatron.lite.model.nemotron_h.vllm.primitive.attention.module import Fa4Fp8KVAttention
    from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import SSMMeta
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
    assert_within_noise_floor(("dq", "dk", "dv"), actual, old, reference, kind)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
def test_bf16_linear_vjp_within_bf16_noise_floor(kind):
    """lm_head-like BF16 projection: batch-invariant forward, TE GEMM VJP."""
    from megatron.lite.model.nemotron_h.vllm.primitive.dense import linear

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
    from megatron.lite.model.nemotron_h.vllm.primitive.dense import GatedRMSNorm, RMSNorm

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


def _lightning_router():
    from types import SimpleNamespace

    from megatron.lite.model.nemotron_h.vllm.primitive.moe.module import Router

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
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
def test_router_vjp_within_bf16_noise_floor(kind, monkeypatch):
    """Lightning router: 128 experts, top-6 sigmoid, renormalized, fixed ids."""
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
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.module import _CombineVJP

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
    from megatron.lite.model.nemotron_h.vllm.primitive.logprob import aligned_selected_log_probs
    from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy

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

    def old_vjp(dtype, device="cuda"):
        # The FP64 reference runs on the CPU: batch invariance replaces the CUDA
        # matmul/log_softmax kernels, which have no FP64 variant.
        inputs = [t.detach().to(device, dtype).requires_grad_() for t in leaves]
        logits = torch.nn.functional.linear(*inputs)
        if temperature != 1.0:
            logits = logits / temperature
        logits = logits.to(torch.promote_types(dtype, torch.float32))
        selected = logits.log_softmax(-1).gather(-1, labels[:, None].to(device))
        grads = torch.autograd.grad(
            selected[:, 0], inputs, upstream.to(device, selected.dtype)
        )
        return [g.cuda() for g in grads]

    old, reference = old_vjp(torch.bfloat16), old_vjp(torch.float64, "cpu")
    assert_within_noise_floor(("dhidden", "dweight"), actual, old, reference, kind)


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_fp8_attention_vjp_is_run_to_run_bitwise(monkeypatch):
    """full_determinism: the FlashAttention backward accumulates dQ in a fixed order."""
    from megatron.lite.model.nemotron_h.vllm.primitive.attention.module import Fa4Fp8KVAttention
    from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import SSMMeta

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
        "from megatron.lite.model.nemotron_h.vllm.protocol import "
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
    from megatron.lite.model.nemotron_h.vllm.primitive.attention.module import Fa4Fp8KVAttention

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


def test_vendor_backward_signatures_match_the_positional_calls():
    """The attention and Mamba VJPs pass these vendor backwards positionally;
    a reordered signature would silently swap arguments."""
    import inspect

    flash = pytest.importorskip("vllm.vllm_flash_attn.cute.interface")
    conv = pytest.importorskip("causal_conv1d.cpp_functions")
    ssd = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    expected = {
        flash._flash_attn_bwd: (
            ["q", "k", "v", "out", "dout", "lse", "softmax_scale", "causal"],
            ["cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "deterministic"],
        ),
        conv.causal_conv1d_bwd_function: (
            ["x", "weight", "bias", "dout", "seq_idx", "initial_states",
             "dfinal_states", "dx", "return_dinitial_states", "silu_activation"],
            [],
        ),
        ssd._mamba_chunk_scan_combined_bwd: (
            ["dout", "x", "dt", "A", "B", "C", "out", "chunk_size"],
            ["D", "dt_bias", "seq_idx", "dt_softplus", "dt_limit", "state_dtype"],
        ),
    }
    for function, (positional, keywords) in expected.items():
        parameters = list(inspect.signature(function).parameters)
        assert parameters[: len(positional)] == positional, function.__name__
        assert set(keywords) <= set(parameters), function.__name__


def test_chunks_restart_at_every_request_not_packed_offset():
    meta = SSMMeta((0, 127, 256, 259))
    assert meta.chunks() == ((0, 127, 255, 256, 259), (0, 2, 3), (0, 1, 1, 2))
    with pytest.raises(ValueError, match="token count"):
        meta.validate_tokens(258)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("kind", ["random", "zero", "sparse"])
def test_packed_conv_vjp_within_bf16_noise_floor(kind):
    from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import packed_conv
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
    from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import packed_scan
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


def _route_ids(rows, generator, *, experts=128, topk=6):
    scores = torch.rand(rows, experts, generator=generator, device="cuda")
    return scores.topk(topk, dim=-1).indices.to(torch.int32)


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
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import routed_vjp

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


def test_route_input_gradients_sum_as_ds4():
    """DS4 rounds to BF16 after each slot add; an FP32 sum would keep 1/256."""
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import sum_route_grads

    per_route = torch.tensor([[1, 1 / 256, -1, 0, 0, 0]]).to(torch.bfloat16)[..., None]
    assert sum_route_grads(per_route).item() == 0
    assert per_route.float().sum().item() == 1 / 256


@cuda
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_routed_vjp_on_the_cutedsl_deployment_at_updated_weights(monkeypatch):
    """The training VJP on the CuTe-DSL deployment, at theta0 and two updated
    snapshots requantized from the updated masters."""
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import CuteDslRoutedExperts, ep4_routed_experts, routed_vjp

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
            experts, x, routes, ids, return_fc1=True
        )
        assert torch.equal(out, ep4_routed_experts(experts, x, routes, ids))
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
