"""Explicit HF tensor mapping for locally owned native Nemotron parameters."""

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import torch

# Trusted digests of the BF16 release (bf16_release.json): the HF LFS sha256 of
# every shard at the pinned revision, and per layer (or top-level tensor) a
# sha256 over its tensors, which a proxy cut of release layers reproduces.
BF16_RELEASE_MANIFEST = Path(__file__).with_name("bf16_release.json")


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
    """Expose the deployment tensors to the framework's PP exporter."""

    def __init__(self, config):
        self.num_experts = config.n_routed_experts

    @staticmethod
    def is_expert(name):
        return ".mixer.experts." in name

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
        for name, tensor in hf_tensor_views(model):
            if any(name.startswith(prefix + ".") for prefix in quantized) and not (
                name.endswith(".k_proj.k_scale") or name.endswith(".v_proj.v_scale")
            ):
                continue
            yield name, tensor.detach()
        for prefix, tensors in quantized.items():
            for suffix, tensor in tensors.items():
                yield f"{prefix}.{suffix}", tensor
        for weights in _routed_checkpoint_owners(model).values():
            yield from weights.export_quantized().items()


@torch.no_grad()
def refresh_quantized_projections(chunks, *, recompute_scales=False, restore=False):
    """Refresh explicitly after updates, including updates bypassing _version.

    ``recompute_scales`` (set by the post-optimizer hook) switches every module
    from checkpoint scales to scales recomputed from its master, for good.
    ``restore`` reinstalls the deployment bytes a training checkpoint restored
    (the bytes last deployed when it was saved) without requantizing.
    """
    if recompute_scales and restore:
        raise ValueError("A restore reinstalls the saved bytes; it does not requantize")
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
                module.refresh_quantized(
                    recompute_scales=recompute_scales, restore=restore
                )
        else:
            module.refresh_deployment(
                recompute_scales=recompute_scales, restore=restore
            )


def hf_tensor_views(model):
    """Yield HF names and destination views of the non-routed tensors."""
    routed = _routed_checkpoint_owners(model)
    for name, tensor in model.state_dict(keep_vars=True).items():
        if any(name.startswith(prefix + ".") for prefix in routed):
            continue
        for scale in ("k", "v"):
            suffix = f".mixer.kv_attention.{scale}_scale"
            if name.endswith(suffix):
                name = name.removesuffix(suffix) + f".mixer.{scale}_proj.{scale}_scale"
                break
        yield (name if name.startswith("lm_head.") else f"backbone.{name}"), tensor


def _read_tensors(root, index, names, device="cpu"):
    from safetensors import safe_open

    result = {}
    for filename in sorted({index[name] for name in names}):
        with safe_open(root / filename, framework="pt", device=str(device)) as handle:
            for name in names:
                if index[name] == filename:
                    result[name] = handle.get_tensor(name)
    return result


def _quantized_masters(model):
    """(HF prefix, algorithm, BF16 master view, deployed tensors) per projection.

    The last element is a callable returning the current deployment tensors.
    """
    from .fp8_training import Fp8TrainingLinear
    from .quantization import Nvfp4TrainingLinear

    for name, module in model.named_modules():
        if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
            prefix = name if name.split(".")[0] == "lm_head" else f"backbone.{name}"
            algorithm = (
                "FP8" if isinstance(module, Fp8TrainingLinear) else "W4A16_NVFP4"
            )
            yield prefix, algorithm, module.weight, module._tensors
    for weights in _routed_checkpoint_owners(model).values():
        for projection in ("up_proj", "down_proj"):
            for expert in range(weights.num_experts):
                yield (
                    f"{weights.prefix}.{expert}.{projection}",
                    "W4A16_NVFP4",
                    getattr(weights, projection)[expert],
                    lambda w=weights, p=projection, e=expert: (
                        w._checkpoint(p, e).tensors
                    ),
                )


def release_group(name):
    """``backbone.layers.N`` / ``mtp.layers.N`` for layer tensors, else the name."""
    parts = name.split(".")
    if len(parts) > 3 and parts[1] == "layers" and parts[0] in ("backbone", "mtp"):
        return ".".join(parts[:3])
    return name


def release_group_digests(root, index, groups):
    """sha256 per group over its tensors' group-relative names, dtypes, shapes, bytes."""
    from safetensors import safe_open

    members = {}
    for name in index:
        group = release_group(name)
        if group in groups:
            members.setdefault(group, []).append(name)
    missing = set(groups) - members.keys()
    if missing:
        raise ValueError(f"BF16 master source lacks {sorted(missing)[:4]}")
    digests = {}
    with ExitStack() as stack:
        handles = {}
        for group, names in sorted(members.items()):
            digest = hashlib.sha256()
            for name in sorted(names):
                if index[name] not in handles:
                    handles[index[name]] = stack.enter_context(
                        safe_open(Path(root) / index[name], framework="pt", device="cpu")
                    )
                tensor = handles[index[name]].get_tensor(name)
                header = f"{name[len(group):]}|{tensor.dtype}|{tuple(tensor.shape)}\n"
                digest.update(header.encode())
                digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy())
            digests[group] = digest.hexdigest()
    return digests


def _verify_bf16_release(master_root, source, names, layers, manifest=None):
    """Fail unless every group read is the trusted release's (proxy layers mapped)."""
    manifest = manifest or json.loads(BF16_RELEASE_MANIFEST.read_text())

    def released(group):
        parts = group.split(".")
        if layers is not None and parts[:2] == ["backbone", "layers"] and len(parts) == 3:
            return f"backbone.layers.{layers[int(parts[2])]}"
        return group

    groups = {release_group(name) for name in names}
    for group, digest in release_group_digests(master_root, source, groups).items():
        if manifest["groups"].get(released(group)) != digest:
            raise ValueError(
                f"BF16 master {group} is not {released(group)} of "
                f"{manifest['repo']}@{manifest['revision']}"
            )


@torch.no_grad()
def _load_bf16_masters(model, root, master_root, layers=None):
    """Initialize every BF16 master from the BF16 release the checkpoint was cut from.

    Identity: every tensor group read must match the trusted release digests
    (``layers`` names the release layer of each proxy layer). Compatibility
    validation: every non-quantized tensor is bitwise equal to the checkpoint
    (FP32 router weights equal the BF16 upcast) and every quantized tensor is
    its BF16 weight encoded on the checkpoint scales. Then every deployment is
    requantized from its master, so the theta0 deployment is requant(BF16
    master), not the checkpoint bytes.
    """
    from .quantization import check_reversible

    master_root = Path(master_root)
    source = json.loads((master_root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    masters = list(_quantized_masters(model))
    verify = getattr(model, "_bf16_master_verified", None) != str(master_root.resolve())
    if verify and any(hasattr(master, "main_grad") for _, _, master, _ in masters):
        # The optimizer's FP32 mains are copied from the masters when it is
        # built; a later first load would leave them on the checkpoint grid.
        raise RuntimeError("Load the BF16 masters before building the optimizer")
    if verify:
        quantized = {prefix for prefix, *_ in masters}
        plain = [
            name
            for name, _ in hf_tensor_views(model)
            if not any(name.startswith(prefix + ".") for prefix in quantized)
            and not name.endswith((".k_proj.k_scale", ".v_proj.v_scale"))
        ]
        _verify_bf16_release(
            master_root,
            source,
            [f"{prefix}.weight" for prefix, *_ in masters] + plain,
            layers,
        )
    suffixes = {
        "FP8": ("weight", "weight_scale"),
        "W4A16_NVFP4": ("weight", "weight_scale", "weight_scale_2"),
    }
    for start in range(0, len(masters), 64):
        group = masters[start : start + 64]
        names = [f"{prefix}.weight" for prefix, *_ in group]
        missing = [name for name in names if name not in source]
        if missing:
            raise ValueError(f"BF16 master source lacks {missing[:4]}")
        # Read straight to the masters' device: host RAM is the optimizer's.
        device = group[0][2].device
        values = _read_tensors(master_root, source, names, device)
        stored = (
            _read_tensors(
                root,
                index,
                [f"{p}.{s}" for p, a, *_ in group for s in suffixes[a]],
                device,
            )
            if verify
            else {}
        )
        for prefix, algorithm, master, _ in group:
            value = values[f"{prefix}.weight"]
            if value.dtype != torch.bfloat16 or value.shape != master.shape:
                raise ValueError(f"BF16 master source disagrees for {prefix}")
            master.copy_(value)
            if verify:
                check_reversible(
                    algorithm,
                    master,
                    {s: stored[f"{prefix}.{s}"] for s in suffixes[algorithm]},
                    prefix,
                    exact_global=True,
                )
    if verify:
        missing = [name for name in plain if name not in source]
        if missing:
            raise ValueError(f"BF16 master source lacks {missing[:4]}")
        device = masters[0][2].device if masters else "cpu"
        theirs = _read_tensors(master_root, source, plain, device)
        ours = _read_tensors(root, index, plain, device)
        for name in plain:
            a, b = ours.pop(name), theirs.pop(name)
            same = (
                a.dtype == b.dtype
                and a.shape == b.shape
                and torch.equal(
                    a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
                )
            ) or (
                a.dtype == torch.float32
                and b.dtype == torch.bfloat16
                and a.shape == b.shape
                and torch.equal(a, b.float())
            )
            if not same:
                raise ValueError(
                    f"BF16 master source is incompatible with this checkpoint: {name}"
                )
    refresh_quantized_projections([model], recompute_scales=True)
    if verify:
        _check_theta0_agreement(masters, root, index)
        model._bf16_master_verified = str(master_root.resolve())


# Fail-closed theta0 floors for requant(BF16 master) against the checkpoint:
# per rank over all NVFP4 tensors, and per tensor. Effective weights (dequantized values) are
# the contract; codes and block scales are compared on blocks that are not all
# zero (ModelOpt floors an all-zero block's scale at 2^-9, TE writes 0). Full
# Lightning model with TE 4over6 MSE, measured on all 5935 NVFP4 tensors:
# 99.93% values overall, >= 99.88% per layer; per tensor >= 97.87% values,
# >= 98.07% codes and >= 97.69% block scales. Global scales and FP8 tensors
# whose checkpoint scale is amax/448 must match exactly.
THETA0_NVFP4_MIN_VALUES = 0.998
THETA0_NVFP4_MIN_NONZERO_BLOCKS = 0.997
THETA0_NVFP4_TENSOR_MIN_VALUES = 0.95
THETA0_NVFP4_TENSOR_MIN_NONZERO_BLOCKS = 0.95


@torch.no_grad()
def _check_theta0_agreement(masters, root, index):
    """Compare the requantized theta0 deployment with the checkpoint and fail closed.

    Per tensor class: packed codes, block scales, global scales and the
    dequantized (effective) weights. NVFP4 aggregates must reach the floors
    above and every global scale must be equal. FP8
    tensors whose checkpoint scale is amax/448 must match exactly; the others
    carry a ModelOpt-calibrated scale the requantization replaces by amax/448.
    """
    import logging
    import re

    from .quantization import QuantizedWeight

    suffixes = {
        "FP8": ("weight", "weight_scale"),
        "W4A16_NVFP4": ("weight", "weight_scale", "weight_scale_2"),
    }
    stats = {}

    def add(key, same):
        entry = stats.setdefault(key, [0, 0])
        entry[0] += int(same.sum())
        entry[1] += same.numel()

    for prefix, algorithm, master, deployed_fn in masters:
        names = [f"{prefix}.{suffix}" for suffix in suffixes[algorithm]]
        stored = _read_tensors(root, index, names, master.device)
        stored = {s: stored[f"{prefix}.{s}"] for s in suffixes[algorithm]}
        deployed = {k: deployed_fn()[k] for k in suffixes[algorithm]}
        kind = re.sub(r"\.\d+\.", ".N.", prefix)
        kind = f"{'fp8' if algorithm == 'FP8' else 'nvfp4'}:{kind}"
        ours = QuantizedWeight(algorithm, stored).initial_master()
        theirs = QuantizedWeight(algorithm, deployed).initial_master()
        values_same = ours == theirs
        add((kind, "values"), values_same)
        if algorithm == "W4A16_NVFP4":
            agreement = float(values_same.float().mean())
            if agreement < THETA0_NVFP4_TENSOR_MIN_VALUES:
                raise RuntimeError(
                    f"theta0 {prefix} values agreement {agreement:.4%} is below "
                    f"{THETA0_NVFP4_TENSOR_MIN_VALUES:.2%}"
                )
        nonzero = None
        if algorithm == "W4A16_NVFP4":
            add(("nvfp4", "values"), values_same)
            rows = ours.shape[0]
            nonzero = (ours.reshape(rows, -1, 16) != 0).any(-1) | (
                theirs.reshape(rows, -1, 16) != 0
            ).any(-1)
        for suffix in suffixes[algorithm]:
            a = stored[suffix]
            b = deployed[suffix].reshape(a.shape).to(a.dtype)
            width = torch.uint8 if a.element_size() == 1 else torch.int32
            same = a.view(width) == b.view(width)
            add((kind, suffix), same)
            if nonzero is not None and suffix != "weight_scale_2":
                # packed codes: 8 bytes per 16-value block
                mask = nonzero if suffix == "weight_scale" else (
                    nonzero.repeat_interleave(8, -1)
                )
                add((kind, f"{suffix}[nonzero blocks]"), same[mask])
                add(("nvfp4", f"{suffix}[nonzero blocks]"), same[mask])
                if same[mask].numel():
                    agreement = float(same[mask].float().mean())
                    if agreement < THETA0_NVFP4_TENSOR_MIN_NONZERO_BLOCKS:
                        raise RuntimeError(
                            f"theta0 {prefix}.{suffix} agreement on nonzero blocks "
                            f"{agreement:.4%} is below "
                            f"{THETA0_NVFP4_TENSOR_MIN_NONZERO_BLOCKS:.2%}"
                        )
            if algorithm == "W4A16_NVFP4" and suffix == "weight_scale_2":
                add(("nvfp4", suffix), same)
        if algorithm == "FP8":
            amax = master.float().abs().amax()
            standard = torch.equal(
                (amax / torch.tensor(448.0, device=amax.device)).reshape(()),
                stored["weight_scale"].float().reshape(()),
            )
            add(("fp8", "standard_scale_tensors"), torch.tensor([standard]))
            if standard and not bool(values_same.all()):
                raise RuntimeError(f"{prefix}: amax/448 checkpoint not reproduced")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    log = logging.getLogger(__name__)
    for (kind, suffix), (equal, total) in sorted(stats.items()):
        log.warning(
            "theta0 vs checkpoint rank%d %s.%s: %d/%d equal (%.4f%%)",
            rank, kind, suffix, equal, total, 100 * equal / max(total, 1),
        )
    floors = {
        "values": THETA0_NVFP4_MIN_VALUES,
        "weight[nonzero blocks]": THETA0_NVFP4_MIN_NONZERO_BLOCKS,
        "weight_scale[nonzero blocks]": THETA0_NVFP4_MIN_NONZERO_BLOCKS,
    }
    for suffix, floor in floors.items():
        equal, total = stats.get(("nvfp4", suffix), (0, 0))
        if total and equal < floor * total:
            raise RuntimeError(
                f"theta0 NVFP4 {suffix} agreement {equal / total:.4%} is below "
                f"{floor:.2%}"
            )
    equal, total = stats.get(("nvfp4", "weight_scale_2"), (0, 0))
    if equal != total:
        raise RuntimeError(f"theta0 NVFP4 global scales differ: {total - equal}")


@torch.no_grad()
def load_hf_weights(model, path):
    """Load ordinary tensors and verify preconstructed quantized projections.

    Quantized projections must already own the checkpoint's deployment values
    before optimizer binding. This initial loader does not restore different
    quantized weights or optimizer state into an existing training model.
    With a BF16 master source (``model._bf16_master_root``), the masters come
    from that release and the deployments are their requantization instead.
    """
    from safetensors import safe_open

    from .fp8_training import Fp8TrainingLinear
    from .quantization import Nvfp4TrainingLinear

    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    master_root = getattr(model, "_bf16_master_root", None)
    if master_root is not None:
        _load_bf16_masters(
            model, root, master_root, getattr(model, "_bf16_master_layers", None)
        )
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
    from .quantization import check_reversible

    if master_root is None:
        for name, module in model.named_modules():
            if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
                algorithm = (
                    "FP8" if isinstance(module, Fp8TrainingLinear) else "W4A16_NVFP4"
                )
                check_reversible(algorithm, module.weight, module._tensors(), name)
        for weights in _routed_checkpoint_owners(model).values():
            for projection in ("up_proj", "down_proj"):
                for expert in range(weights.num_experts):
                    check_reversible(
                        "W4A16_NVFP4",
                        getattr(weights, projection)[expert],
                        weights._checkpoint(projection, expert).tensors,
                        f"{weights.prefix}.{expert}.{projection}",
                    )
    for prefix, tensors in quantized.items():
        for suffix, deployed in tensors.items():
            name = f"{prefix}.{suffix}"
            if name not in index:
                raise ValueError(
                    f"Missing required quantized checkpoint tensor: {name}"
                )
            with safe_open(root / index[name], framework="pt", device="cpu") as handle:
                stored = handle.get_tensor(name)
            # A requantized deployment keeps only the checkpoint's static
            # activation scale; the weight bytes are requant(master).
            if master_root is not None and suffix != "input_scale":
                if stored.shape != deployed.shape or stored.dtype != deployed.dtype:
                    raise ValueError(f"Requantized deployment geometry differs: {name}")
                continue
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
