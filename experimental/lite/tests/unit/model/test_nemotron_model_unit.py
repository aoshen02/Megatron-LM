"""Nemotron protocol loss normalization."""

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
