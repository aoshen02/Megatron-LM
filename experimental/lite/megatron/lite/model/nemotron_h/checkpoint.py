"""Explicit HF tensor mapping for locally owned native Nemotron parameters."""

import json
from pathlib import Path

import torch


def _routed_checkpoint_owners(model):
    from .nvfp4_experts import Nvfp4ExpertWeights
    from .nvfp4_moe import Nvfp4RoutedDeployment

    owners = {}
    for name, module in model.named_modules():
        if any(name.startswith(parent + ".") for parent in owners):
            continue
        if isinstance(module, Nvfp4RoutedDeployment | Nvfp4ExpertWeights):
            if (
                getattr(model.ps, "tp_size", 1) != 1
                or model.ps.ep_size != 1
                or model.ps.ep_rank != 0
            ):
                raise ValueError("Quantized routed checkpoint IO requires TP1/EP1")
            weights = (
                module.weights
                if isinstance(module, Nvfp4RoutedDeployment)
                else module
            )
            if (
                weights.prefix != f"backbone.{name}"
                or weights.num_experts != model.config.n_routed_experts
            ):
                raise ValueError("Routed checkpoint prefix/ownership mismatch")
            owners[name] = weights
    return owners


def load_fp8_kv_scales(path, layer_ids, *, device):
    """Read fixed KV scales for explicitly assigned attention layers."""
    from safetensors import safe_open

    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = {}
    for layer in layer_ids:
        if layer in result:
            raise ValueError(f"Duplicate attention layer: {layer}")
        values = []
        for kind in ("k", "v"):
            name = f"backbone.layers.{layer}.mixer.{kind}_proj.{kind}_scale"
            if name not in index:
                raise ValueError(f"Missing required FP8 KV scale: {name}")
            with safe_open(root / index[name], framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
            if (
                value.dtype != torch.float32
                or value.numel() != 1
                or not torch.isfinite(value).all()
                or not (value > 0).all()
            ):
                raise ValueError(f"Expected positive fixed FP32 KV scale: {name}")
            values.append(value.to(device))
        result[layer] = tuple(values)
    return result


class NemotronExport:
    """Expose stacked expert views to the framework's EP/PP exporter."""

    def __init__(self, config):
        self.num_experts = config.n_routed_experts

    @staticmethod
    def is_expert(name):
        return ".mixer.experts." in name

    @staticmethod
    def export_expert_local_id(name):
        return int(name.split(".experts.", 1)[1].split(".", 1)[0])

    @staticmethod
    def export_expert_name(name, index):
        prefix, suffix = name.split(".experts.", 1)
        return f"{prefix}.experts.{index}.{suffix.split('.', 1)[1]}"

    @staticmethod
    def tp_spec(name):
        return None

    @staticmethod
    def native_to_hf(name, tensor):
        return [(name, tensor)]

    def iter_export_tensors(self, model):
        from .fp8_training import Fp8TrainingLinear
        from .quantization import Nvfp4TrainingLinear

        quantized = {}
        for name, module in model.named_modules():
            if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
                prefix = name if name.startswith("lm_head") else f"backbone.{name}"
                quantized[prefix] = module.export_quantized()
        first = model.ps.ep_rank * (self.num_experts // model.ps.ep_size)
        for name, tensor in hf_tensor_views(model):
            if any(name.startswith(prefix + ".") for prefix in quantized) and not (
                name.endswith(".k_proj.k_scale") or name.endswith(".v_proj.v_scale")
            ):
                continue
            if self.is_expert(name):
                name = self.export_expert_name(
                    name, self.export_expert_local_id(name) - first
                )
            yield name, tensor.detach()
        for prefix, tensors in quantized.items():
            for suffix, tensor in tensors.items():
                yield f"{prefix}.{suffix}", tensor
        for weights in _routed_checkpoint_owners(model).values():
            for name, tensor in weights.export_quantized().items():
                yield self.export_expert_name(
                    name, self.export_expert_local_id(name) - first
                ), tensor


@torch.no_grad()
def refresh_quantized_projections(chunks, *, recompute_scales=False):
    """Refresh explicitly after updates, including updates bypassing _version.

    ``recompute_scales`` (set by the post-optimizer hook) switches every module
    from checkpoint scales to scales recomputed from its master, for good.
    """
    from .fp8_training import Fp8TrainingLinear
    from .nvfp4_experts import Nvfp4ExpertWeights
    from .nvfp4_moe import Nvfp4RoutedDeployment
    from .quantization import Nvfp4TrainingLinear

    modules = dict.fromkeys(
        module
        for chunk in chunks
        for module in chunk.modules()
        if isinstance(
            module,
            Nvfp4TrainingLinear
            | Fp8TrainingLinear
            | Nvfp4RoutedDeployment
            | Nvfp4ExpertWeights,
        )
    )
    owned_weights = {
        module.weights
        for module in modules
        if isinstance(module, Nvfp4RoutedDeployment)
    }
    for module in modules:
        if isinstance(module, Nvfp4ExpertWeights):
            if module not in owned_weights:
                module.refresh_quantized(recompute_scales=recompute_scales)
        else:
            module.refresh_deployment(recompute_scales=recompute_scales)


def hf_tensor_views(model):
    """Yield HF names and destination views, without gathering remote experts."""
    local_experts = model.config.n_routed_experts // model.ps.ep_size
    first_expert = model.ps.ep_rank * local_experts
    routed = _routed_checkpoint_owners(model)
    for name, tensor in model.state_dict(keep_vars=True).items():
        if any(name.startswith(prefix + ".") for prefix in routed):
            continue
        for scale in ("k", "v"):
            suffix = f".mixer.kv_attention.{scale}_scale"
            if name.endswith(suffix):
                name = name.removesuffix(suffix) + f".mixer.{scale}_proj.{scale}_scale"
                break
        if ".mixer.experts." in name:
            prefix, projection = name.rsplit(".", 1)
            if projection not in ("up_proj", "down_proj"):
                raise ValueError(f"Unrecognized expert weight: {name}")
            if tensor.shape[0] != local_experts:
                raise ValueError(f"Incorrect local expert ownership: {name}")
            for local in range(local_experts):
                yield (
                    f"backbone.{prefix}.{first_expert + local}.{projection}.weight",
                    tensor[local],
                )
        else:
            yield (name if name.startswith("lm_head.") else f"backbone.{name}"), tensor


@torch.no_grad()
def load_hf_weights(model, path):
    """Load ordinary tensors and verify preconstructed quantized projections.

    Quantized projections must already own the checkpoint's deployment values
    before optimizer binding. This initial loader does not restore different
    quantized weights or optimizer state into an existing training model.
    """
    from safetensors import safe_open

    from .fp8_training import Fp8TrainingLinear
    from .quantization import Nvfp4TrainingLinear

    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    quantized = {}
    for name, module in model.named_modules():
        if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
            prefix = (
                name
                if name.startswith("lm_head.") or name == "lm_head"
                else f"backbone.{name}"
            )
            quantized[prefix] = module.export_quantized()
    for weights in _routed_checkpoint_owners(model).values():
        quantized[weights.prefix] = {
            name.removeprefix(weights.prefix + "."): value
            for name, value in weights.export_quantized().items()
        }
    for prefix, tensors in quantized.items():
        for suffix, deployed in tensors.items():
            name = f"{prefix}.{suffix}"
            if name not in index:
                raise ValueError(
                    f"Missing required quantized checkpoint tensor: {name}"
                )
            with safe_open(root / index[name], framework="pt", device="cpu") as handle:
                stored = handle.get_tensor(name)
            if (
                stored.shape != deployed.shape
                or stored.dtype != deployed.dtype
                or not torch.equal(
                    stored.reshape(-1).view(torch.uint8),
                    deployed.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                )
            ):
                raise ValueError(
                    f"Quantized checkpoint differs from constructed adapter: {name}"
                )
    targets = {
        name: tensor
        for name, tensor in hf_tensor_views(model)
        if not any(name.startswith(prefix + ".") for prefix in quantized)
        or name.endswith(".k_proj.k_scale")
        or name.endswith(".v_proj.v_scale")
    }
    missing = targets.keys() - index.keys()
    if missing:
        raise ValueError(f"Missing required HF weights: {sorted(missing)}")
    for filename in sorted({index[name] for name in targets}):
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for name, target in targets.items():
                if index[name] != filename:
                    continue
                value = handle.get_tensor(name)
                if value.dtype in (torch.uint8, torch.float8_e4m3fn):
                    raise ValueError(
                        f"Quantized weight {name} requires a quantized training adapter"
                    )
                if value.shape != target.shape:
                    raise ValueError(
                        f"HF shape mismatch for {name}: {value.shape} != {target.shape}"
                    )
                if target.is_meta:
                    raise ValueError(
                        "Materialize model parameters before loading weights"
                    )
                target.copy_(value)
