"""Checkpoint-domain NVFP4/FP8 encoding used by the Nemotron-H actor.

The actor forward and the rollout export read the same encoding, so these tests
pin the encoding itself: zero group scales and post-update scale growth.
"""

import pytest
import torch
from megatron.lite.model.nemotron_h.quantization import QuantizedWeight, grow_scales

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


def _checkpoint(rows=8, cols=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
    scale = (torch.rand(rows, cols // 16, generator=g) * 200 + 8).to(torch.float8_e4m3fn)
    return _nvfp4(weight, scale, torch.tensor(1e-4))


def test_unchanged_master_reencodes_to_identical_bytes():
    """Scales recomputed from a dequantized checkpoint would requantize every
    layer at the first update; an unchanged master must keep its bytes."""
    checkpoint = _checkpoint()
    out = grow_scales("W4A16_NVFP4", checkpoint.initial_master(), checkpoint.tensors)
    for name in ("weight", "weight_scale", "weight_scale_2"):
        assert torch.equal(
            out[name].reshape(-1).view(torch.uint8),
            checkpoint.tensors[name].reshape(-1).view(torch.uint8),
        ), name


def test_small_overflow_keeps_the_scale_large_overflow_grows_only_that_block():
    checkpoint = _checkpoint()
    master = checkpoint.initial_master()
    factors = checkpoint.tensors["weight_scale"].float() * 1e-4
    master[0, 0] = 6.9 * factors[0, 0]  # within the top code's rounding range
    master[1, 0] = 9.0 * factors[1, 0]  # beyond it
    out = grow_scales("W4A16_NVFP4", master, checkpoint.tensors)
    grown = out["weight_scale"].float() != checkpoint.tensors["weight_scale"].float()
    assert grown.nonzero().tolist() == [[1, 0]]
    decoded = QuantizedWeight("W4A16_NVFP4", out).initial_master()
    new_factors = (out["weight_scale"].float() * out["weight_scale_2"]).repeat_interleave(16, -1)
    assert ((decoded - master).abs() <= new_factors + 1e-12).all()


def test_block_beyond_e4m3_range_grows_the_global_scale():
    checkpoint = _checkpoint()
    master = checkpoint.initial_master()
    master[2, 5] = 6 * 448 * 1e-4 * 2  # needs a block scale of 896 at the old global
    out = grow_scales("W4A16_NVFP4", master, checkpoint.tensors)
    assert out["weight_scale_2"] > checkpoint.tensors["weight_scale_2"]
    decoded = QuantizedWeight("W4A16_NVFP4", out).initial_master()
    assert torch.isclose(decoded[2, 5], master[2, 5], rtol=1e-6, atol=0)


def test_fp8_scale_grows_only_past_the_top_rounding_range():
    master = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    current = {"weight_scale": torch.tensor(3.0 / 448)}
    assert torch.equal(grow_scales("FP8", master, current)["weight_scale"], current["weight_scale"])
    assert torch.equal(
        grow_scales("FP8", master * (464 / 448) * 0.999, current)["weight_scale"],
        current["weight_scale"],
    )
    grown = grow_scales("FP8", master * 2, current)
    assert torch.isclose(grown["weight_scale"], torch.tensor(6.0 / 448))
