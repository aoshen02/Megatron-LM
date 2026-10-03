"""Checkpoint-domain routed-expert weights, NOT a training MoE operator."""

import json
from pathlib import Path

import torch

from .quantization import QuantizedWeight, load_quantized_weight, requantize


class Nvfp4ExpertWeights(torch.nn.Module):
    """TP1/EP1 BF16 masters and their NVFP4 deployment bytes, with explicit refresh.

    The deployment bytes are the checkpoint's until the first
    ``refresh_quantized`` with ``recompute_scales`` (after the first optimizer
    update); from then on every refresh requantizes the masters with the
    checkpoint's rule (``quantization.requantize``).

    No forward, backward, routing, Humming packing, or deployment is implemented.
    Ordinary optimizer mutations are version-checked. Runtime updates through
    ``.data`` must call refresh_quantized explicitly (or mark_dirty before export)
    because they can bypass PyTorch's version counter.
    """

    def __init__(
        self,
        root,
        prefix,
        quantized_layers,
        *,
        num_experts,
        hidden_size,
        intermediate_size,
        tp_size=1,
        ep_size=1,
        device="cpu",
    ):
        super().__init__()
        if tp_size != 1 or ep_size != 1:
            raise ValueError("NVFP4 expert checkpoint container requires TP1/EP1")
        if (
            not isinstance(prefix, str)
            or not prefix.startswith("backbone.layers.")
            or not prefix.endswith(".mixer.experts")
        ):
            raise ValueError("Expected explicit HF routed-experts prefix")
        if (
            min(num_experts, hidden_size, intermediate_size) <= 0
            or hidden_size % 16
            or intermediate_size % 16
        ):
            raise ValueError(
                "Both expert contraction dimensions must be positive multiples of 16"
            )
        self.prefix, self.num_experts = prefix, num_experts
        self._geometry = {
            "up_proj": (intermediate_size, hidden_size),
            "down_proj": (hidden_size, intermediate_size),
        }
        names = {
            f"{prefix}.{expert}.{projection}"
            for expert in range(num_experts)
            for projection in ("up_proj", "down_proj")
        }
        actual = {key for key in quantized_layers if key.startswith(prefix + ".")}
        if actual != names:
            raise ValueError("Recipes must cover exactly all requested up/down experts")
        for name in names:
            recipe = quantized_layers[name]
            if (
                recipe.get("quant_algo") != "W4A16_NVFP4"
                or recipe.get("group_size") != 16
            ):
                raise ValueError("Every routed projection must use W4A16_NVFP4 group16")
        root = Path(root)
        mapping = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        suffixes = ("weight", "weight_scale", "weight_scale_2")
        expected_keys = {f"{name}.{suffix}" for name in names for suffix in suffixes}
        if {key for key in mapping if key.startswith(prefix + ".")} != expected_keys:
            raise ValueError(
                "Checkpoint must contain exactly packed weights and fixed scales "
                "for all experts"
            )
        self._global_shapes = {}
        for projection, rows, columns in (
            ("up_proj", intermediate_size, hidden_size),
            ("down_proj", hidden_size, intermediate_size),
        ):
            master = torch.empty(
                num_experts, rows, columns, dtype=torch.bfloat16, device=device
            )
            packed = torch.empty(
                num_experts, rows, columns // 2, dtype=torch.uint8, device=device
            )
            scales = torch.empty(
                num_experts,
                rows,
                columns // 16,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            global_scales = torch.empty(num_experts, dtype=torch.float32, device=device)
            shapes = []
            for expert in range(num_experts):
                name = f"{prefix}.{expert}.{projection}"
                checkpoint = load_quantized_weight(root, name, quantized_layers[name])
                if checkpoint.tensors["weight"].shape != (rows, columns // 2):
                    raise ValueError(f"Wrong packed geometry for {name}")
                master[expert].copy_(checkpoint.initial_master())
                packed[expert].copy_(checkpoint.tensors["weight"])
                scales[expert].copy_(checkpoint.tensors["weight_scale"])
                global_scales[expert].copy_(
                    checkpoint.tensors["weight_scale_2"].reshape(())
                )
                shapes.append(checkpoint.tensors["weight_scale_2"].shape)
            setattr(self, projection, torch.nn.Parameter(master))
            self.register_buffer(f"_{projection}_packed", packed)
            self.register_buffer(f"_{projection}_scale", scales)
            self.register_buffer(f"_{projection}_global", global_scales)
            self._global_shapes[projection] = shapes
        self._synced_versions = self._versions()
        self._dirty = False

    def _versions(self):
        tensors = []
        for projection in self._geometry:
            tensors.append(getattr(self, projection))
            tensors.extend(
                getattr(self, f"_{projection}_{suffix}")
                for suffix in ("packed", "scale", "global")
            )
        return tuple(
            (id(t), t.data_ptr(), t._version, t.dtype, t.device, t.shape, t.stride())
            for t in tensors
        )

    def _validate_storage(self):
        device = self.up_proj.device
        for projection, (rows, columns) in self._geometry.items():
            master = getattr(self, projection)
            specs = [
                (master, (self.num_experts, rows, columns), torch.bfloat16),
                (
                    getattr(self, f"_{projection}_packed"),
                    (self.num_experts, rows, columns // 2),
                    torch.uint8,
                ),
            ]
            for suffix in ("scale",):
                specs.append(
                    (
                        getattr(self, f"_{projection}_{suffix}"),
                        (self.num_experts, rows, columns // 16),
                        torch.float8_e4m3fn,
                    )
                )
            for suffix in ("global",):
                specs.append(
                    (
                        getattr(self, f"_{projection}_{suffix}"),
                        (self.num_experts,),
                        torch.float32,
                    )
                )
            if not isinstance(master, torch.nn.Parameter):
                raise RuntimeError("Expert master must remain a Parameter")
            for tensor, shape, dtype in specs:
                if (
                    tensor.dtype != dtype
                    or tensor.shape != shape
                    or tensor.device != device
                    or tensor.is_meta
                    or tensor.layout != torch.strided
                    or not tensor.is_contiguous()
                ):
                    raise RuntimeError(
                        "Expert storage must preserve checkpoint dtype, geometry, "
                        "contiguous layout and common real device"
                    )

    def _checkpoint(self, projection, expert):
        return QuantizedWeight(
            "W4A16_NVFP4",
            {
                "weight": getattr(self, f"_{projection}_packed")[expert],
                "weight_scale": getattr(self, f"_{projection}_scale")[expert],
                "weight_scale_2": getattr(self, f"_{projection}_global")[
                    expert
                ].reshape(self._global_shapes[projection][expert]),
            },
        )

    def mark_dirty(self):
        """Invalidate before non-versioned runtime writes; refresh before export."""
        self._dirty = True

    @torch.no_grad()
    def refresh_quantized(self, recompute_scales=False):
        """Once the masters have been updated, requantize them in place."""
        self._dirty = True
        self._validate_storage()
        self._requantized = getattr(self, "_requantized", False) | recompute_scales
        if self._requantized:
            for projection in ("up_proj", "down_proj"):
                parameter = getattr(self, projection)
                for expert in range(self.num_experts):
                    tensors = requantize("W4A16_NVFP4", parameter[expert])
                    getattr(self, f"_{projection}_packed")[expert].copy_(tensors["weight"])
                    getattr(self, f"_{projection}_scale")[expert].copy_(
                        tensors["weight_scale"]
                    )
                    getattr(self, f"_{projection}_global")[expert].copy_(
                        tensors["weight_scale_2"]
                    )
        self._synced_versions = self._versions()
        self._dirty = False

    def export_quantized(self):
        """Return independent tensors under complete original HF expert keys."""
        self._validate_storage()
        if self._dirty or self._versions() != self._synced_versions:
            raise RuntimeError("Refresh quantized expert weights after master updates")
        return {
            f"{self.prefix}.{expert}.{projection}.{suffix}": tensor.detach().clone()
            for expert in range(self.num_experts)
            for projection in ("up_proj", "down_proj")
            for suffix, tensor in self._checkpoint(projection, expert).tensors.items()
        }
