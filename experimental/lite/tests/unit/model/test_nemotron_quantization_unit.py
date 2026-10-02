"""Checkpoint-domain NVFP4/FP8 encoding used by the Nemotron-H actor.

The actor forward and the rollout export read the same encoding, so these tests
pin the encoding itself: zero group scales, dynamic (post-update) scales, and
determinism. GPU cases call vLLM's native quantizers.
"""

import pytest
import torch
from megatron.lite.model.nemotron_h.quantization import QuantizedWeight, quantize_master

LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4(weight, scale, global_scale):
    return QuantizedWeight(
        "W4A16_NVFP4",
        {"weight": weight, "weight_scale": scale, "weight_scale_2": global_scale},
    )


def test_zero_group_scale_decodes_and_encodes_an_all_zero_block():
    """A zero group scale is valid NVFP4 for an all-zero block: decoding gives
    zeros and re-encoding must not divide by it."""
    weight = torch.full((2, 16), 0x21, dtype=torch.uint8)  # codes 1, 2
    scale = torch.tensor([[1.0, 1.0], [0.0, 2.0]]).to(torch.float8_e4m3fn)
    checkpoint = _nvfp4(weight, scale, torch.tensor(0.5))

    master = checkpoint.initial_master()

    assert torch.equal(master[1, :16], torch.zeros(16))
    assert torch.equal(master[0, :2], torch.tensor([0.25, 0.5]))
    assert torch.equal(checkpoint.encode_master(master), weight.where(
        torch.tensor([[True] * 16, [False] * 8 + [True] * 8]), 0
    ))


def test_nonpositive_global_scale_is_rejected():
    weight = torch.zeros((1, 8), dtype=torch.uint8)
    scale = torch.ones((1, 1)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        _nvfp4(weight, scale, torch.tensor(0.0)).initial_master()


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vLLM native quantizers need CUDA"
)


@requires_cuda
def test_dynamic_nvfp4_matches_checkpoint_format_and_bounds_error():
    torch.manual_seed(0)
    master = torch.randn(64, 256, device="cuda") * 0.02
    master[3, 32:48] = 0  # one all-zero group
    master[5, 7] = 0.5  # one outlier dominating the global scale

    tensors = quantize_master("W4A16_NVFP4", master)

    assert tensors["weight"].dtype == torch.uint8
    assert tensors["weight"].shape == (64, 128)
    assert tensors["weight_scale"].dtype == torch.float8_e4m3fn
    assert tensors["weight_scale"].shape == (64, 16)
    assert tensors["weight_scale_2"].dtype == torch.float32
    expected_global = master.to(torch.bfloat16).abs().max().float() / (6 * 448)
    torch.testing.assert_close(tensors["weight_scale_2"], expected_global)
    assert tensors["weight_scale"][3, 2].float() == 0

    decoded = QuantizedWeight("W4A16_NVFP4", tensors).initial_master()
    assert torch.equal(decoded[3, 32:48], torch.zeros(16, device="cuda"))
    # Each value lies within half the widest grid step (2 units) of its group.
    factors = tensors["weight_scale"].float().repeat_interleave(16, -1)
    factors = factors * tensors["weight_scale_2"]
    reference = master.to(torch.bfloat16).float()
    assert ((decoded - reference).abs() <= factors + 1e-12).all()
    assert torch.equal(decoded[5, 7], reference[5, 7])


@requires_cuda
def test_dynamic_quantization_is_deterministic_and_handles_all_zero():
    master = torch.randn(32, 64, device="cuda")
    first = quantize_master("W4A16_NVFP4", master)
    second = quantize_master("W4A16_NVFP4", master.clone())
    for name in first:
        assert torch.equal(
            first[name].reshape(-1).view(torch.uint8),
            second[name].reshape(-1).view(torch.uint8),
        ), name

    zeros = quantize_master("W4A16_NVFP4", torch.zeros(32, 64, device="cuda"))
    assert not zeros["weight"].any()
    assert not zeros["weight_scale"].float().any()
    assert torch.isfinite(zeros["weight_scale_2"]) and zeros["weight_scale_2"] > 0


@requires_cuda
def test_dynamic_fp8_scale_follows_the_master():
    master = torch.randn(32, 64, device="cuda") * 3
    tensors = quantize_master("FP8", master)

    assert tensors["weight"].dtype == torch.float8_e4m3fn
    expected = master.to(torch.bfloat16).abs().max().float() / 448
    torch.testing.assert_close(tensors["weight_scale"], expected)
    decoded = QuantizedWeight(
        "FP8",
        {"weight": tensors["weight"], "weight_scale": tensors["weight_scale"]},
    ).initial_master()
    assert decoded.abs().max() <= master.abs().max() * (1 + 2**-3)
