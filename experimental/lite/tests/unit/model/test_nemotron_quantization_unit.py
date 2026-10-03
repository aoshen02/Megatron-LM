"""Checkpoint-domain NVFP4/FP8 encoding and the BF16 master-weight VJPs.

The actor forward and the rollout export read the same encoding, so these tests
pin the encoding itself and the gradients the actor feeds the optimizer.
"""

import pytest
import torch
from megatron.lite.model.nemotron_h.quantization import QuantizedWeight, requantize

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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


def test_nonpositive_global_scale_is_rejected():
    weight = torch.zeros((1, 8), dtype=torch.uint8)
    scale = torch.ones((1, 1)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        _nvfp4(weight, scale, torch.tensor(0.0)).initial_master()


@cuda
def test_nvfp4_requantization_is_idempotent_on_its_own_output():
    """The checkpoint rule reproduces its own dequantized weights, so an
    unchanged master keeps its deployment values across refreshes."""
    g = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(256, 512, generator=g, device="cuda").to(torch.bfloat16)
    first = requantize("W4A16_NVFP4", weight)
    decoded = QuantizedWeight("W4A16_NVFP4", first).initial_master()
    second = requantize("W4A16_NVFP4", decoded.to(torch.bfloat16))
    assert torch.equal(QuantizedWeight("W4A16_NVFP4", second).initial_master(), decoded)


@cuda
def test_fp8_requantization_uses_the_tensor_amax():
    weight = torch.tensor([[1.0, -2.0], [0.5, 3.0]], device="cuda").to(torch.bfloat16)
    out = requantize("FP8", weight)
    assert torch.isclose(out["weight_scale"].cpu(), torch.tensor(3.0 / 448))
    assert out["weight"].float().abs().max() == 448


@cuda
def test_linear_vjp_is_the_bf16_master_weight_gradient():
    from megatron.lite.model.nemotron_h.functional import native_linear_vjp

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(64, 128, generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn(96, 128, generator=g, device="cuda").to(torch.bfloat16)
    dy = torch.randn(64, 96, generator=g, device="cuda").to(torch.bfloat16)
    dx, dw = native_linear_vjp(dy, x, w)
    torch.testing.assert_close(dx.float(), dy.float() @ w.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dw.float(), dy.float().T @ x.float(), rtol=1e-2, atol=1e-2)


@cuda
def test_routed_vjp_matches_the_dense_relu2_reference():
    from megatron.lite.model.nemotron_h.nvfp4_moe_vjp import routed_vjp

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

    ref = [t.detach().float().requires_grad_() for t in (x, up, down, routes)]
    u = torch.einsum("mk,msik->msi", ref[0], ref[1][ids])
    u = u + (fc1.float() - u).detach()  # the visible FC1 value, the native gradient
    v = torch.einsum("msi,mski->msk", u.relu().square(), ref[2][ids])
    (ref[3][..., None] * v).sum(1).backward(dy.float())

    got = routed_vjp(x, fc1.reshape(m * topk, i), up, down, routes, ids, dy)
    for actual, expected in zip(got, ref, strict=True):
        torch.testing.assert_close(actual.float(), expected.grad, rtol=5e-2, atol=5e-2)
