"""Nemotron-H checkpoint quantization rules for NVFP4/FP8 weight QAT."""

import torch


FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def requantize(algorithm, weight):
    """Quantize a BF16 master with the rule that produced the Lightning checkpoint.

    NVFP4: Transformer Engine 4over6 with the 256 E4M3 bound, choosing map4/map6
    by squared error (ModelOpt's static-MSE 4over6), global = amax / 1536. FP8:
    per-tensor scale = amax / 448, codes = E4M3(BF16(w / scale)) as ModelOpt
    rounds the quotient.
    """
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("Expected a BF16 master matrix")
    if algorithm == "FP8":
        values = weight.float()
        scale = values.abs().amax() / torch.tensor(448.0, device=values.device)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return {"weight": fp8_encode(weight, scale), "weight_scale": scale}
    if algorithm != "W4A16_NVFP4" or weight.shape[-1] % 16:
        raise ValueError(f"Unsupported requantization: {algorithm}")
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=256,
        nvfp4_4over6_err_mode="MSE",
    )
    quantized = quantizer(weight.contiguous())
    rows, cols = weight.shape
    amax = quantized._amax_rowwise.float().reshape(())
    return {
        "weight": quantized._rowwise_data.view(torch.uint8)[:rows, : cols // 2]
        .contiguous(),
        "weight_scale": quantized._rowwise_scale_inv.view(torch.float8_e4m3fn)[
            :rows, : cols // 16
        ].contiguous(),
        "weight_scale_2": amax / torch.tensor(1536.0, device=amax.device),
    }


def fp8_encode(weight, scale):
    """E4M3 codes of ``weight / scale`` with the quotient rounded to BF16 first."""
    quotient = (weight.float() / scale.float().reshape(())).to(torch.bfloat16)
    return quotient.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def nvfp4_decode_values(packed):
    """Signed FP4 values of packed NVFP4 codes (low nibble first)."""
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    levels = torch.tensor(FP4_LEVELS, device=packed.device)
    values = levels[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -values, values)
