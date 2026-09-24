# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Differentiable DSv4 inference-precision boundaries."""

import torch
import torch.nn.functional as F


def _quantize_fp8(x: torch.Tensor, block_scaling_dim: int) -> torch.Tensor:
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

    quantizer = Float8BlockQuantizer(
        DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        amax_epsilon=1e-4,
        force_pow_2_scales=True,
        block_scaling_dim=block_scaling_dim,
    )
    quantizer.internal = False
    return quantizer.quantize(x)


def quantize_kv_nope(kv: torch.Tensor) -> torch.Tensor:
    """Round NoPE to block-64 FP8, preserving the final 64 RoPE channels.

    Transformer Engine supplies the quantizer's straight-through gradient.
    This returns dense values, not a packed inference cache.
    """
    if kv.shape[-1] != 512:
        raise ValueError("DSv4 KV quantization requires 448 NoPE and 64 RoPE channels")
    blocks = kv[..., :448].reshape(-1, 64).float()
    # TE uses blocks of 128; zero padding preserves each 64-channel amax.
    padded = F.pad(blocks, (0, 64))
    # Tensor dispatch preserves autograd; the storage dequantize method does not.
    restored = (_quantize_fp8(padded, 1) + 0)[:, :64]
    restored = restored.reshape(*kv.shape[:-1], 448).to(kv.dtype)
    return torch.cat((restored, kv[..., 448:]), dim=-1)


def inference_precision_grouped_projection(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply grouped projection with block128 activations and block128x128 weights.

    Tensor dispatch retains TE quantizer gradients. This is a dense projection
    on FP8-rounded operands, not a hardware FP8 GEMM implementation.
    """
    if x.shape[-1] % 128 or weight.shape[-2] % 128:
        raise ValueError("Inference output projection requires block128-aligned dimensions")
    if weight.ndim != 3 or x.shape[-2:] != (weight.shape[0], weight.shape[2]):
        raise ValueError("Expected inputs [..., groups, channels] and weights [groups, out, channels]")
    # Materialize BF16 operands to avoid dependence on global TF32 settings.
    qx = (_quantize_fp8(x.float(), 1) + 0).bfloat16()
    qw = (_quantize_fp8(weight.float(), 2) + 0).bfloat16()
    return torch.einsum("...gk,gok->...go", qx, qw)
