"""Checkpoint-domain tensors for mixed-precision Nemotron training adapters."""

from dataclasses import dataclass

import torch


def projection_layer(factory, prefix, in_features, out_features, **kwargs):
    """Construct before optimizer binding, preserving the ordinary BF16 API."""
    if factory is None:
        return torch.nn.Linear(in_features, out_features, **kwargs)
    if not callable(factory) or not prefix:
        raise ValueError(
            "A callable projection factory and explicit HF prefix are required"
        )
    return factory(prefix, in_features, out_features, **kwargs)


class CheckpointProjectionFactory:
    """Strict checkpoint-aware projection construction, never module replacement.

    Plain BF16 tensors are validated here and loaded by the normal HF loader.
    Quantized adapters load their own checkpoint-domain tensors at construction.
    The caller owns vLLM configuration/parallel initialization for deployment.
    """

    def __init__(self, root, quantized_layers, quant_config):
        import json
        from pathlib import Path

        self.root = Path(root)
        self.recipes = dict(quantized_layers)
        self.quant_config = quant_config
        self.index = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )["weight_map"]

    def __call__(
        self,
        prefix,
        in_features,
        out_features,
        *,
        bias=False,
        device=None,
        dtype=torch.bfloat16,
    ):
        from safetensors import safe_open

        if not isinstance(prefix, str) or not prefix or prefix.endswith("."):
            raise ValueError("Expected an explicit HF module prefix")
        if dtype != torch.bfloat16 or in_features <= 0 or out_features <= 0:
            raise ValueError(
                "Expected positive projection geometry and BF16 activations"
            )

        def tensor_info(suffix):
            name = f"{prefix}.{suffix}"
            if name not in self.index:
                raise ValueError(f"Missing checkpoint projection tensor: {name}")
            with safe_open(
                self.root / self.index[name], framework="pt", device="cpu"
            ) as handle:
                view = handle.get_slice(name)
                return tuple(view.get_shape()), view.get_dtype()

        shape, stored_dtype = tensor_info("weight")
        recipe = self.recipes.get(prefix)
        has_bias = f"{prefix}.bias" in self.index
        if has_bias != bias:
            raise ValueError(f"Checkpoint bias contract disagrees for {prefix}")
        if recipe is None:
            if shape != (out_features, in_features) or stored_dtype != "BF16":
                raise ValueError(
                    f"Unquantized projection requires matching BF16 weight: {prefix}"
                )
            if any(
                f"{prefix}.{suffix}" in self.index
                for suffix in (
                    "weight_scale",
                    "weight_scale_2",
                    "input_scale",
                )
            ):
                raise ValueError(
                    f"Checkpoint scales require an explicit recipe: {prefix}"
                )
            if bias and tensor_info("bias") != ((out_features,), "BF16"):
                raise ValueError(
                    f"Checkpoint bias geometry/dtype disagrees for {prefix}"
                )
            return torch.nn.Linear(
                in_features, out_features, bias=bias, device=device, dtype=dtype
            )
        if bias:
            raise ValueError("Quantized projection adapters do not support bias")
        algorithm = recipe.get("quant_algo")
        if algorithm == "FP8":
            expected = (out_features, in_features), "F8_E4M3"
        elif algorithm == "W4A16_NVFP4":
            if in_features % 16 or recipe.get("group_size") != 16:
                raise ValueError(
                    "NVFP4 projections require group16 and K divisible by 16"
                )
            expected = (out_features, in_features // 2), "U8"
        else:
            raise ValueError(f"Unsupported projection recipe: {algorithm}")
        if (shape, stored_dtype) != expected:
            raise ValueError(
                f"Quantized projection geometry/dtype disagrees for {prefix}"
            )
        checkpoint = load_quantized_weight(self.root, prefix, recipe)
        return build_quantized_projection(
            checkpoint, prefix, self.quant_config, device=device
        )


_E4M3_MAX = 448.0


def _e4m3_ceil(value):
    """Smallest E4M3 value >= ``value`` (``value`` finite, positive, <= 448)."""
    rounded = value.to(torch.float8_e4m3fn)
    short = rounded.float() < value
    bumped = (rounded.view(torch.uint8) + short.to(torch.uint8)).view(torch.float8_e4m3fn)
    return bumped


def grow_scales(algorithm, master, current):
    """Encode a master, enlarging the current scales only where it overflows.

    ``current`` holds the scales in use (checkpoint-domain ``weight_scale`` and,
    for NVFP4, ``weight_scale_2``). A block keeps its scale while its largest
    magnitude still rounds to the top code without exceeding the top grid's
    rounding error (|x| <= 7 * factor for NVFP4, whose top codes are 4 and 6;
    <= 464 * scale for FP8, whose top values are 448 and the 480 it lacks).
    Otherwise it gets the smallest E4M3 scale that holds it, and the NVFP4
    global only grows when a block exceeds the top E4M3 scale by the same
    tolerance. An unchanged
    master therefore re-encodes to identical bytes.
    """
    if master.dtype != torch.float32 or master.ndim != 2:
        raise ValueError("Expected an FP32 master matrix")
    if not torch.isfinite(master).all():
        raise ValueError("Expected finite FP32 master weights")
    if algorithm == "FP8":
        scale = current["weight_scale"].float().reshape(())
        amax = master.abs().max()
        if amax > 464.0 * scale:
            scale = amax / _E4M3_MAX
        weight = QuantizedWeight("FP8", {"weight": master.to(torch.float8_e4m3fn),
                                         "weight_scale": scale}).encode_master(master)
        return {"weight": weight, "weight_scale": scale}
    if algorithm != "W4A16_NVFP4" or master.shape[-1] % 16:
        raise ValueError(f"Unsupported scale update: {algorithm}")
    rows, cols = master.shape
    block_amax = master.abs().reshape(rows, cols // 16, 16).amax(-1)
    global_scale = current["weight_scale_2"].float().reshape(())
    scale = current["weight_scale"].float().reshape(rows, cols // 16)
    needed = block_amax / (6.0 * global_scale)
    # Same tolerance as a block: the top scale still rounds |x| <= 7/6 of it.
    if needed.max() > _E4M3_MAX * (7.0 / 6.0):
        old = global_scale
        global_scale = block_amax.max() / (6.0 * _E4M3_MAX)
        # Rescaled scales leave the E4M3 grid; round up so no block clips.
        scale = _e4m3_ceil((scale * (old / global_scale)).clamp_max(_E4M3_MAX)).float()
        needed = block_amax / (6.0 * global_scale)
    grow = needed > scale * (7.0 / 6.0)
    scale = torch.where(grow, _e4m3_ceil(needed.clamp_max(_E4M3_MAX)).float(), scale)
    encoded = {
        "weight_scale": scale.to(torch.float8_e4m3fn),
        "weight_scale_2": global_scale.to(torch.float32),
    }
    packed_shape = (rows, cols // 2)
    weight = QuantizedWeight(
        "W4A16_NVFP4",
        {"weight": torch.zeros(packed_shape, dtype=torch.uint8, device=master.device), **encoded},
    ).encode_master(master)
    return {"weight": weight, **encoded}


@dataclass(frozen=True)
class QuantizedWeight:
    """Keep serialized scales separate from runtime-specific packed layouts."""

    algorithm: str
    tensors: dict[str, torch.Tensor]

    def initial_master(self):
        """Recover an FP32 representative, not the original prequantized weight."""
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
            codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2)
            levels = weight.new_tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32
            )
            values = levels[(codes & 7).long()]
            values = torch.where((codes & 8) != 0, -values, values)
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

    def encode_master(self, master):
        """Fixed-scale reference serializer; scales are not recalibrated."""
        if master.dtype != torch.float32 or not torch.isfinite(master).all():
            raise ValueError("Expected finite FP32 master weights")
        stored = self.tensors["weight"]
        expected = (
            (stored.shape[0], stored.shape[1] * 2)
            if self.algorithm == "W4A16_NVFP4"
            else stored.shape
        )
        if master.shape != expected:
            raise ValueError("Master shape does not match checkpoint geometry")
        scale = self.tensors["weight_scale"].float()
        if self.algorithm == "FP8":
            maximum = torch.finfo(torch.float8_e4m3fn).max
            return (
                (master * scale.reciprocal())
                .clamp(-maximum, maximum)
                .to(torch.float8_e4m3fn)
            )
        factors = scale.repeat_interleave(16, -1) * self.tensors["weight_scale_2"]
        # A zero group scale encodes an all-zero block; never divide by it.
        normalized = torch.where(factors > 0, master / factors, 0.0)
        midpoints = master.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        magnitude = normalized.abs().contiguous()
        index = torch.bucketize(magnitude, midpoints, right=False)
        tie = (index < 7) & (magnitude == midpoints[index.clamp_max(6)])
        index = index + (tie & ((index & 1) != 0)).long()
        codes = index.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)
        return codes[:, 0::2] | (codes[:, 1::2] << 4)


class _Nvfp4LinearVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master, dequantized, deployment):
        # Saving master also catches an optimizer update before backward.
        ctx.save_for_backward(x, master, dequantized)
        return deployment(x)

    @staticmethod
    def backward(ctx, dy):
        x, master, dequantized = ctx.saved_tensors
        dx = (dy.float() @ dequantized).to(x.dtype)
        dw = dy.flatten(0, -2).float().T @ x.flatten(0, -2).float()
        return dx, dw, None, None


class Nvfp4TrainingLinear(torch.nn.Module):
    """Inference-visible W4A16 linear with fixed-scale identity weight STE.

    The factory consumes checkpoint-domain tensors and constructs a fresh frozen
    inference layer. Refresh explicitly after optimizer updates and outside Graph
    capture; this initial adapter does not promise stable deployment pointers.
    """

    def __init__(self, checkpoint, deployment_factory, *, device):
        super().__init__()
        if checkpoint.algorithm != "W4A16_NVFP4":
            raise ValueError("This VJP supports W4A16 NVFP4 only")
        self.weight = torch.nn.Parameter(checkpoint.initial_master().to(device))
        self.register_buffer(
            "weight_scale", checkpoint.tensors["weight_scale"].to(device)
        )
        self.register_buffer(
            "weight_scale_2", checkpoint.tensors["weight_scale_2"].to(device)
        )
        self.register_buffer(
            "_packed", checkpoint.tensors["weight"].to(device), persistent=False
        )
        self.register_buffer(
            "_dequantized", self.weight.detach().clone(), persistent=False
        )
        self._factory = deployment_factory
        self._install(self._packed)

    def _checkpoint(self, packed):
        return QuantizedWeight(
            "W4A16_NVFP4",
            {
                "weight": packed,
                "weight_scale": self.weight_scale,
                "weight_scale_2": self.weight_scale_2,
            },
        )

    def _install(self, packed):
        # Clone loader inputs: runtime transforms must not mutate export scales.
        tensors = {
            k: v.detach().clone() for k, v in self._checkpoint(packed).tensors.items()
        }
        deployment = self._factory(tensors)
        deployment.requires_grad_(False)
        self._inference = deployment
        self._packed = packed.detach().clone()
        self._dequantized = self._checkpoint(self._packed).initial_master()
        self._deployed_versions = self._versions()

    def _versions(self):
        return (
            self.weight._version,
            self.weight_scale._version,
            self.weight_scale_2._version,
        )

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        """Re-encode the master; after the first update, grow overflowing scales.

        Scales stay the checkpoint's until ``recompute_scales`` is first set
        (after the first optimizer update); from then on a block's scale only
        grows where the master overflows it (``grow_scales``).
        """
        self._recompute_scales = getattr(self, "_recompute_scales", False)
        self._recompute_scales |= recompute_scales
        if not self._recompute_scales:
            packed = self._checkpoint(self._packed).encode_master(self.weight)
        else:
            tensors = grow_scales(
                "W4A16_NVFP4",
                self.weight,
                {"weight_scale": self.weight_scale, "weight_scale_2": self.weight_scale_2},
            )
            self.weight_scale.copy_(tensors["weight_scale"])
            self.weight_scale_2.copy_(
                tensors["weight_scale_2"].reshape(self.weight_scale_2.shape)
            )
            packed = tensors["weight"]
        self._install(packed)

    def export_quantized(self):
        if self._versions() != self._deployed_versions:
            raise RuntimeError(
                "Refresh deployment after updating master weights or scales"
            )
        return {
            k: v.detach().clone()
            for k, v in self._checkpoint(self._packed).tensors.items()
        }

    def forward(self, x):
        if self._versions() != self._deployed_versions:
            raise RuntimeError(
                "Refresh deployment after updating master weights or scales"
            )
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.weight.shape[-1]:
            raise ValueError("Expected BF16 activations with checkpoint K")
        return _Nvfp4LinearVJP.apply(x, self.weight, self._dequantized, self._inference)


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
    if algorithm == "FP8":
        scale = tensors["input_scale"]
        if (
            scale.dtype != torch.float32
            or scale.numel() != 1
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("Expected a positive FP32 static activation scale")
    return QuantizedWeight(algorithm, tensors)


def build_quantized_projection(checkpoint, prefix, quant_config, *, device):
    """Construct a training projection before binding optimizer parameters.

    vLLM config/parallel initialization belongs to the caller's runtime. NVFP4
    deployments use the same ModelOpt loader and selected kernel as inference.
    """
    from .fp8_training import Fp8TrainingLinear

    if checkpoint.algorithm == "FP8":
        return Fp8TrainingLinear(checkpoint, device=device)
    if checkpoint.algorithm != "W4A16_NVFP4":
        raise ValueError(f"Unsupported projection recipe: {checkpoint.algorithm}")

    def deployment_factory(tensors):
        import vllm.envs as envs
        from vllm.model_executor.layers.linear import ReplicatedLinear

        if not envs.VLLM_BATCH_INVARIANT:
            raise RuntimeError("Aligned NVFP4 projection requires batch invariance")
        n, packed_k = tensors["weight"].shape
        with torch.device(device):
            layer = ReplicatedLinear(
                packed_k * 2,
                n,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                prefix=prefix,
                return_bias=False,
                disable_tp=True,
            )
        shared_projection = prefix.rsplit(".", 2)[-2:]
        if shared_projection in (
            ["shared_experts", "up_proj"],
            ["shared_experts", "down_proj"],
        ):
            from vllm.distributed import get_tensor_model_parallel_world_size
            from vllm.model_executor.kernels.linear.nvfp4.base import (
                NvFp4LinearLayerConfig,
            )
            from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
                NemotronSharedNvFp4LinearKernel,
            )

            if get_tensor_model_parallel_world_size() != 1:
                raise ValueError("Aligned Nemotron shared W4A16 requires TP=1")
            supported, reason = NemotronSharedNvFp4LinearKernel.is_supported()
            if not supported:
                raise ValueError(reason)
            layer.quant_method.kernel = NemotronSharedNvFp4LinearKernel(
                NvFp4LinearLayerConfig()
            )
        elif type(layer.quant_method.kernel).__name__ != "HummingNvFp4LinearKernel":
            raise RuntimeError("Expected the validated BI Humming W4A16 kernel")
        with torch.no_grad():
            for name, tensor in tensors.items():
                parameter = getattr(layer, name)
                parameter.weight_loader(parameter, tensor.to(device))
            layer.quant_method.process_weights_after_loading(layer)
        return layer

    return Nvfp4TrainingLinear(checkpoint, deployment_factory, device=device)
