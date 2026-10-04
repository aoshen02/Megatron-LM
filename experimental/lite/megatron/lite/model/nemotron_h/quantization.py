"""Nemotron-H checkpoint quantization policy shared by load and resync."""

from dataclasses import dataclass

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


def nvfp4_encode_values(weight, scale, global_scale):
    """FP4 values (round to nearest even) of a master on given NVFP4 scales."""
    levels = torch.tensor(FP4_LEVELS, device=weight.device)
    midpoints = (levels[1:] + levels[:-1]) / 2
    rows, cols = weight.shape
    blocks = weight.float().reshape(rows, cols // 16, 16)
    unit = (scale.float() * global_scale.float().reshape(()))[..., None]
    y = torch.where(unit > 0, blocks / unit, torch.zeros_like(blocks))
    magnitude = y.abs().clamp(max=6.0)
    index = torch.bucketize(magnitude, midpoints)
    tie = (index < 7) & (magnitude == midpoints[index.clamp(max=6)])
    index = torch.where(tie & (index % 2 == 1), index + 1, index)
    return torch.where(y < 0, -levels[index], levels[index]).reshape(rows, cols)


def check_reversible(algorithm, master, tensors, name, *, exact_global=False):
    """Compatibility: fail unless the checkpoint is the master encoded on its scales.

    Not an identity check (a master moved within its cells passes). FP8:
    E4M3(BF16(master / scale)) must give the checkpoint codes. NVFP4: FP4
    rounding of master / (block scale * global scale) must give the checkpoint
    FP4 values wherever that unit is positive (no product is compared); with
    ``exact_global`` (the master is the original BF16 source) the global scale
    must also be amax / 1536.
    """
    master = master.detach()
    tensors = {key: value.to(master.device) for key, value in tensors.items()}
    if algorithm == "FP8":
        codes = fp8_encode(master, tensors["weight_scale"]).view(torch.uint8)
        expected = tensors["weight"].view(torch.uint8)
        changed = int((codes != expected).sum())
    elif algorithm == "W4A16_NVFP4":
        global_scale = tensors["weight_scale_2"].float()
        scale = tensors["weight_scale"]
        values = nvfp4_encode_values(master, scale, global_scale)
        unit = (scale.float() * global_scale.reshape(())).repeat_interleave(16, -1)
        stored = nvfp4_decode_values(tensors["weight"])
        changed = int(((values != stored) & (unit > 0)).sum())
        if exact_global:
            amax = master.float().abs().amax()
            derived = amax / torch.tensor(1536.0, device=master.device)
            if not torch.equal(derived.reshape(()), global_scale.reshape(())):
                raise RuntimeError(f"{name}: checkpoint global scale is not amax/1536")
    else:
        raise ValueError(f"Unsupported quantization: {algorithm}")
    if changed:
        raise RuntimeError(
            f"{name} is not its BF16 master on the checkpoint scales: "
            f"{changed} values changed"
        )


@dataclass(frozen=True)
class QuantizedWeight:
    """Keep serialized scales separate from runtime-specific packed layouts."""

    algorithm: str
    tensors: dict[str, torch.Tensor]

    def initial_master(self):
        """Dequantize the checkpoint to FP32."""
        weight = self.tensors["weight"]
        scale = self.tensors["weight_scale"]
        if self.algorithm == "FP8":
            if weight.dtype != torch.float8_e4m3fn or scale.numel() != 1:
                raise ValueError("Expected tensor-scaled E4M3 FP8 weights")
            factors = scale.float()
            values = weight.float()
        elif self.algorithm == "W4A16_NVFP4":
            if weight.dtype != torch.uint8 or weight.ndim != 2:
                raise ValueError("Expected a packed uint8 NVFP4 matrix")
            if scale.dtype != torch.float8_e4m3fn:
                raise ValueError("Expected E4M3 group scales")
            if scale.shape != (weight.shape[0], weight.shape[1] // 8):
                raise ValueError("Expected group16 scales for packed NVFP4")
            if weight.shape[1] % 8:
                raise ValueError("NVFP4 K must be divisible by 16")
            global_scale = self.tensors["weight_scale_2"]
            if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
                raise ValueError("Expected one FP32 checkpoint global scale")
            values = nvfp4_decode_values(weight)
            factors = scale.float().repeat_interleave(16, -1) * global_scale
        else:
            raise ValueError(f"Unsupported quantization: {self.algorithm}")
        if not torch.isfinite(factors).all() or (factors < 0).any():
            raise ValueError("Weight scales must be finite and nonnegative")
        if self.algorithm == "FP8" and not (factors > 0).all():
            raise ValueError("FP8 weight scales must be positive")
        if self.algorithm == "W4A16_NVFP4" and not (global_scale > 0).all():
            raise ValueError("NVFP4 global scales must be positive")
        result = values * factors
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite checkpoint weights")
        return result


def full_master(master):
    """The BF16 master the forward sees: an FSDP2 shard is gathered, then cast
    as FSDP2's BF16 unshard casts it."""
    from torch.distributed.tensor import DTensor

    if isinstance(master, DTensor):
        master = master.full_tensor()
    return master.detach().to(torch.bfloat16)


def master_version(master):
    """Version of a master, advanced by any in-place update: an FSDP2 shard
    counts updates through the DTensor and through its local tensor apart."""
    return master._version, getattr(master, "_local_tensor", master)._version


def load_quantized_weight(root, prefix, recipe):
    """Read one explicitly described module without changing its recipe.

    Args:
        root: HF checkpoint directory.
        prefix: Full HF module name, without a tensor suffix.
        recipe: This module's entry from quantized_layers.
    """
    import json
    from pathlib import Path

    from safetensors import safe_open

    algorithm = recipe["quant_algo"]
    if algorithm == "W4A16_NVFP4":
        if recipe.get("group_size") != 16:
            raise ValueError("Only NVFP4 group16 is supported")
        suffixes = ("weight", "weight_scale", "weight_scale_2")
    elif algorithm == "FP8":
        suffixes = ("weight", "weight_scale", "input_scale")
    else:
        raise ValueError(f"Unsupported quantization: {algorithm}")
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = {}
    for suffix in suffixes:
        name = f"{prefix}.{suffix}"
        with safe_open(root / index[name], framework="pt", device="cpu") as handle:
            tensors[suffix] = handle.get_tensor(name)
    return QuantizedWeight(algorithm, tensors)
