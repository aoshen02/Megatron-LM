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


def requantize(algorithm, weight):
    """Quantize a BF16 master with the rule that produced the checkpoint.

    NVFP4 uses Transformer Engine's 4over6 quantizer with the 256 E4M3 bound,
    which reproduces the Lightning checkpoint from its dequantized weights. FP8
    uses vLLM's dynamic per-tensor quantization (scale = amax / 448).
    """
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("Expected a BF16 master matrix")
    if algorithm == "FP8":
        from vllm import _custom_ops as ops

        packed, scale = ops.scaled_fp8_quant(weight.contiguous())
        return {"weight": packed, "weight_scale": scale.reshape(())}
    if algorithm != "W4A16_NVFP4" or weight.shape[-1] % 16:
        raise ValueError(f"Unsupported requantization: {algorithm}")
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    quantizer = NVFP4Quantizer(
        rowwise=True, columnwise=False, nvfp4_use_4over6=True, nvfp4_e4m3_max=256
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
        "weight_scale_2": amax / (6.0 * 256.0),
    }


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


class Nvfp4TrainingLinear(torch.nn.Module):
    """Inference-visible W4A16 linear over a BF16 master weight.

    The deployment starts from the checkpoint bytes. After an optimizer update,
    ``refresh_deployment(recompute_scales=True)`` requantizes the master with
    the checkpoint's own rule. Backward is the BF16 master-weight VJP.
    Refresh outside Graph capture; deployment pointers are not stable.
    """

    def __init__(self, checkpoint, deployment_factory, *, device):
        super().__init__()
        if checkpoint.algorithm != "W4A16_NVFP4":
            raise ValueError("This adapter supports W4A16 NVFP4 only")
        self.weight = torch.nn.Parameter(
            checkpoint.initial_master().to(device, torch.bfloat16)
        )
        for name in ("weight_scale", "weight_scale_2"):
            self.register_buffer(name, checkpoint.tensors[name].to(device))
        self.register_buffer(
            "_packed", checkpoint.tensors["weight"].to(device), persistent=False
        )
        self._factory = deployment_factory
        self._requantized = False
        self._install()

    def _tensors(self):
        return {
            "weight": self._packed,
            "weight_scale": self.weight_scale,
            "weight_scale_2": self.weight_scale_2,
        }

    def _install(self):
        # Clone loader inputs: runtime transforms must not mutate export bytes.
        deployment = self._factory(
            {k: v.detach().clone() for k, v in self._tensors().items()}
        )
        deployment.requires_grad_(False)
        self._inference = deployment
        self._deployed_version = self.weight._version

    def _check_fresh(self):
        if self.weight._version != self._deployed_version:
            raise RuntimeError("Refresh deployment after updating master weights")

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        """Reinstall; once the master has been updated, requantize it first."""
        self._requantized |= recompute_scales
        if self._requantized:
            tensors = requantize("W4A16_NVFP4", self.weight)
            self._packed = tensors["weight"]
            self.weight_scale.copy_(tensors["weight_scale"])
            self.weight_scale_2.copy_(
                tensors["weight_scale_2"].reshape(self.weight_scale_2.shape)
            )
        self._install()

    def export_quantized(self):
        self._check_fresh()
        return {k: v.detach().clone() for k, v in self._tensors().items()}

    def forward(self, x):
        from .functional import visible_linear

        self._check_fresh()
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.weight.shape[-1]:
            raise ValueError("Expected BF16 activations with checkpoint K")
        return visible_linear(self._inference, x, self.weight)


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
