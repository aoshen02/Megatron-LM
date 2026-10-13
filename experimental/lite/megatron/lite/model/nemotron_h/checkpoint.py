"""Explicit HF tensor mapping for locally owned native Nemotron parameters."""

import json
from pathlib import Path

import torch


def is_expert(name):
    return ".mixer.experts." in name


def hf_name(name):
    """HF name of a native module or tensor name."""
    return name if name.split(".")[0] in ("lm_head", "mtp") else f"backbone.{name}"


def _mtp_experts(model):
    """The trained MTP block's BF16 routed experts (native module name, module)."""
    from megatron.lite.model.nemotron_h.vllm.mtp import Bf16RoutedExperts

    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Bf16RoutedExperts)
    ]


def _full_bf16(tensor):
    """The forward's value of a (possibly FSDP2-sharded FP32) parameter."""
    model_dtype = getattr(tensor, "_fsdp2_model_param_dtype", None)
    if model_dtype is not None:
        return tensor.full_tensor().to(model_dtype)
    full_tensor = getattr(tensor, "full_tensor", None)
    return full_tensor() if callable(full_tensor) else tensor


def _routed_checkpoint_owners(model):
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import (
        Nvfp4ExpertWeights,
        Nvfp4RoutedDeployment,
    )

    owners = {}
    for name, module in model.named_modules():
        if any(name.startswith(parent + ".") for parent in owners):
            continue
        if isinstance(module, Nvfp4RoutedDeployment | Nvfp4ExpertWeights):
            if getattr(model.ps, "tp_size", 1) != 1:
                raise ValueError("Quantized routed checkpoint IO requires TP1")
            weights = (
                module.weights
                if isinstance(module, Nvfp4RoutedDeployment)
                else module
            )
            local = model.config.n_routed_experts // model.ps.ep_size
            if (
                weights.prefix != f"backbone.{name}"
                or weights.num_experts != model.config.n_routed_experts
                or (weights.num_local, weights.offset) != (local, model.ps.ep_rank * local)
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

    is_expert = staticmethod(is_expert)

    @staticmethod
    def tp_spec(name):
        return None

    @staticmethod
    def native_to_hf(name, tensor):
        return [(name, tensor)]

    def iter_export_tensors(self, model):
        from megatron.lite.model.nemotron_h.vllm.primitive.dense import Fp8TrainingLinear, Nvfp4TrainingLinear

        quantized = {}
        for name, module in model.named_modules():
            if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
                quantized[hf_name(name)] = module.export_quantized()
        for name, tensor in hf_tensor_views(model):
            kv_scale = name.endswith((".k_proj.k_scale", ".v_proj.v_scale"))
            if any(name.startswith(prefix + ".") for prefix in quantized) and not kv_scale:
                continue
            if kv_scale:
                # vLLM's first load keeps KV scales in BF16 parameters, a reload
                # in FP32 ones; send the BF16-rounded value so both agree.
                tensor = tensor.to(torch.bfloat16).to(torch.float32)
            model_dtype = getattr(tensor, "_fsdp2_model_param_dtype", None)
            if model_dtype is not None:
                # An FSDP2 FP32 shard of a BF16 parameter: the forward's value.
                tensor = tensor.full_tensor().to(model_dtype)
            yield name, tensor.detach()
        for prefix, tensors in quantized.items():
            for suffix, tensor in tensors.items():
                yield f"{prefix}.{suffix}", tensor
        # EP shards use local expert numbers; the exporter gathers them.
        for weights in _routed_checkpoint_owners(model).values():
            yield from weights.export_quantized(local_names=model.ps.ep_size > 1).items()
        for _, experts in _mtp_experts(model):
            up, down = (_full_bf16(p).detach() for p in (experts.up_proj, experts.down_proj))
            for expert in range(experts.num_local):
                names = experts.hf_names(expert, local=model.ps.ep_size > 1)
                yield names["up_proj"], up[expert]
                yield names["down_proj"], down[expert]


@torch.no_grad()
def refresh_quantized_projections(chunks, *, recompute_scales=False):
    """Refresh explicitly after updates, including updates bypassing _version.

    ``recompute_scales`` (set by the post-optimizer hook) switches every module
    from checkpoint scales to scales recomputed from its master, for good.
    """
    from megatron.lite.model.nemotron_h.vllm.primitive.dense import Fp8TrainingLinear, Nvfp4TrainingLinear
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import (
        Nvfp4ExpertWeights,
        Nvfp4RoutedDeployment,
    )

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
    """Yield HF names and destination views of the non-routed tensors."""
    routed = _routed_checkpoint_owners(model)
    routed.update(dict(_mtp_experts(model)))
    for name, tensor in model.state_dict(keep_vars=True).items():
        if any(name.startswith(prefix + ".") for prefix in routed):
            continue
        for scale in ("k", "v"):
            suffix = f".mixer.kv_attention.{scale}_scale"
            if name.endswith(suffix):
                name = name.removesuffix(suffix) + f".mixer.{scale}_proj.{scale}_scale"
                break
        yield hf_name(name), tensor


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
    from megatron.lite.model.nemotron_h.vllm.primitive.dense import Fp8TrainingLinear, Nvfp4TrainingLinear

    for name, module in model.named_modules():
        if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
            prefix = hf_name(name)
            algorithm = (
                "FP8" if isinstance(module, Fp8TrainingLinear) else "W4A16_NVFP4"
            )
            yield prefix, algorithm, module.weight, module._tensors
    for weights in _routed_checkpoint_owners(model).values():
        for projection in ("up_proj", "down_proj"):
            for expert in range(weights.num_local):
                yield (
                    f"{weights.prefix}.{weights.offset + expert}.{projection}",
                    "W4A16_NVFP4",
                    getattr(weights, projection)[expert],
                    lambda w=weights, p=projection, e=expert: (
                        w._checkpoint(p, e).tensors
                    ),
                )


@torch.no_grad()
def _load_bf16_masters(model, root, master_root):
    """Initialize every BF16 master from the BF16 release the checkpoint was cut from.

    Compatibility validation: every non-quantized tensor is bitwise equal to the checkpoint
    (FP32 router weights equal the BF16 upcast) and every quantized tensor is
    its BF16 weight encoded on the checkpoint scales. Then the deployment
    scales are frozen at ModelOpt's static calibration of these masters and
    every deployment is quantized from its master on them, so the theta0
    deployment is the ModelOpt rule on the BF16 master, not the checkpoint bytes.
    """
    from megatron.lite.model.nemotron_h.quantization import check_reversible

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
    if verify:
        freeze_static_scales(masters)
    refresh_quantized_projections([model], recompute_scales=True)
    if verify:
        _check_theta0_agreement(masters, root, index)
        _write_deployment_digests(masters)
        model._bf16_master_verified = str(master_root.resolve())


@torch.no_grad()
def freeze_static_scales(masters):
    """Set every deployment's weight scales to ModelOpt's static calibration of its master.

    Called once, on the BF16 release masters (theta0). The scales live in the
    modules' scale buffers, the tensors the forward uses and the export sends;
    every later refresh quantizes the updated master on them. Checkpoints hold
    parameters only: a resumed run rebuilds the model from the same BF16
    release and recomputes the same scales here before its checkpoint loads.
    """
    import gc

    from megatron.lite.model.nemotron_h.quantization import STATIC_SCALES, calibrate_static_scales

    before = torch.cuda.memory_allocated()
    scales = calibrate_static_scales(
        {prefix: (algorithm, master) for prefix, algorithm, master, _ in masters}
    )
    for prefix, algorithm, _, deployed in masters:
        tensors = deployed()
        for suffix in STATIC_SCALES[algorithm]:
            target, value = tensors[suffix], scales[prefix][suffix]
            if value.dtype != target.dtype or value.numel() != target.numel():
                raise RuntimeError(f"ModelOpt scale geometry differs for {prefix}.{suffix}")
            target.copy_(value.reshape(target.shape))
    # The calibrated ModelOpt tree sits in reference cycles; the colocated
    # rollout needs its memory back before it starts.
    del scales
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[mlite static scales] {len(masters)} layers calibrated by ModelOpt; allocated "
        f"{before / 2**30:.2f} -> {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
        f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB, "
        f"device free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB",
        flush=True,
    )


def _write_deployment_digests(masters):
    """With MLITE_DEPLOY_DIGEST_DIR set, write this rank's theta0 deployment sha256 per tensor."""
    import hashlib
    import os

    directory = os.environ.get("MLITE_DEPLOY_DIGEST_DIR")
    if not directory:
        return
    digests = {}
    for prefix, _, _, deployed in masters:
        for suffix, tensor in deployed().items():
            if suffix == "input_scale":
                continue
            data = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
            digests[f"{prefix}.{suffix}"] = [
                str(tensor.dtype), tensor.numel(), hashlib.sha256(data.tobytes()).hexdigest()
            ]
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    os.makedirs(directory, exist_ok=True)
    path = Path(directory) / f"theta0-rank{rank}-pid{os.getpid()}.json"
    path.write_text(json.dumps(digests))
    total = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()[:16]
    print(
        f"[mlite theta0 deployment] rank {rank}: {len(digests)} tensors, digest {total}",
        flush=True,
    )


# Fail-closed theta0 floors for requant(BF16 master) against the checkpoint,
# per rank over all NVFP4 tensors and per tensor. Codes and block scales are
# compared on blocks that are not all zero (ModelOpt floors an all-zero
# block's scale at 2^-9, TE writes 0).
THETA0_NVFP4_MIN_VALUES = 0.998
THETA0_NVFP4_MIN_NONZERO_BLOCKS = 0.997
THETA0_NVFP4_TENSOR_MIN_VALUES = 0.95
THETA0_NVFP4_TENSOR_MIN_NONZERO_BLOCKS = 0.95
THETA0_FP8_MIN_CODES = 0.9


@torch.no_grad()
def _check_theta0_agreement(masters, root, index):
    """Compare the requantized theta0 deployment with the checkpoint.

    NVFP4 values (FP4 value times block scale), codes and block scales must
    reach the floors above and every global scale must agree within one FP32
    ulp (ModelOpt 0.47 differs from the checkpoint in the last place on some).
    FP8 tensors deployed on the checkpoint's scale (within one ulp) must reach
    the FP8 floor: on the GPU ``to_quantized_weight`` divides by the scale
    rounded to BF16, the checkpoint by the FP32 scale. On the others the
    calibration chose another scale.
    """
    from megatron.lite.model.nemotron_h.quantization import nvfp4_decode_values

    suffixes = {
        "FP8": ("weight", "weight_scale"),
        "W4A16_NVFP4": ("weight", "weight_scale", "weight_scale_2"),
    }
    stats = {}

    def block_values(tensors):
        return nvfp4_decode_values(tensors["weight"]) * tensors[
            "weight_scale"
        ].float().repeat_interleave(16, -1)

    def add(key, same):
        entry = stats.setdefault(key, [0, 0])
        entry[0] += int(same.sum())
        entry[1] += same.numel()

    for prefix, algorithm, master, deployed_fn in masters:
        names = [f"{prefix}.{suffix}" for suffix in suffixes[algorithm]]
        stored = _read_tensors(root, index, names, master.device)
        stored = {s: stored[f"{prefix}.{s}"] for s in suffixes[algorithm]}
        deployed = {k: deployed_fn()[k] for k in suffixes[algorithm]}
        if algorithm == "FP8":
            a = stored["weight_scale"].double().reshape(())
            b = deployed["weight_scale"].double().reshape(())
            if (a - b).abs() <= a * 2.0**-23:
                agreement = float(
                    (deployed["weight"].view(torch.uint8) == stored["weight"].view(torch.uint8))
                    .float()
                    .mean()
                )
                if agreement < THETA0_FP8_MIN_CODES:
                    raise RuntimeError(
                        f"theta0 {prefix} codes agreement {agreement:.4%} on the "
                        f"checkpoint scale is below {THETA0_FP8_MIN_CODES:.2%}"
                    )
            continue
        ours, theirs = block_values(stored), block_values(deployed)
        values_same = ours == theirs
        agreement = float(values_same.float().mean())
        if agreement < THETA0_NVFP4_TENSOR_MIN_VALUES:
            raise RuntimeError(
                f"theta0 {prefix} values agreement {agreement:.4%} is below "
                f"{THETA0_NVFP4_TENSOR_MIN_VALUES:.2%}"
            )
        add("values", values_same)
        rows = ours.shape[0]
        nonzero = (ours.reshape(rows, -1, 16) != 0).any(-1) | (
            theirs.reshape(rows, -1, 16) != 0
        ).any(-1)
        for suffix in suffixes[algorithm]:
            a = stored[suffix]
            b = deployed[suffix].reshape(a.shape).to(a.dtype)
            width = torch.uint8 if a.element_size() == 1 else torch.int32
            same = a.view(width) == b.view(width)
            if suffix == "weight_scale_2":
                add(suffix, (a.double() - b.double()).abs() <= a.double().abs() * 2.0**-23)
                continue
            # packed codes: 8 bytes per 16-value block
            mask = nonzero if suffix == "weight_scale" else nonzero.repeat_interleave(8, -1)
            add(suffix, same[mask])
            if same[mask].numel():
                agreement = float(same[mask].float().mean())
                if agreement < THETA0_NVFP4_TENSOR_MIN_NONZERO_BLOCKS:
                    raise RuntimeError(
                        f"theta0 {prefix}.{suffix} agreement on nonzero blocks "
                        f"{agreement:.4%} is below "
                        f"{THETA0_NVFP4_TENSOR_MIN_NONZERO_BLOCKS:.2%}"
                    )
    floors = {
        "values": THETA0_NVFP4_MIN_VALUES,
        "weight": THETA0_NVFP4_MIN_NONZERO_BLOCKS,
        "weight_scale": THETA0_NVFP4_MIN_NONZERO_BLOCKS,
    }
    for suffix, floor in floors.items():
        equal, total = stats.get(suffix, (0, 0))
        if total and equal < floor * total:
            raise RuntimeError(
                f"theta0 NVFP4 {suffix} agreement {equal / total:.4%} is below "
                f"{floor:.2%}"
            )
    equal, total = stats.get("weight_scale_2", (0, 0))
    if equal != total:
        raise RuntimeError(
            f"theta0 NVFP4 global scales differ by more than one ulp: {total - equal}"
        )


@torch.no_grad()
def load_hf_weights(model, path):
    """Load ordinary tensors and verify preconstructed quantized projections.

    The masters come from the BF16 release (``model._bf16_master_root``) and
    the deployments are their requantization; the checkpoint provides the
    static activation scales. This initial loader does not restore optimizer
    state into an existing training model.
    """
    from safetensors import safe_open

    from megatron.lite.model.nemotron_h.vllm.primitive.dense import Fp8TrainingLinear, Nvfp4TrainingLinear

    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    _load_bf16_masters(model, root, model._bf16_master_root)
    quantized = {}
    for name, module in model.named_modules():
        if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
            quantized[hf_name(name)] = module.export_quantized()
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
            # A requantized deployment keeps only the checkpoint's static
            # activation scale; the weight bytes are requant(master).
            if suffix != "input_scale":
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
    _load_mtp_experts(model, root, index)
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
                target.copy_(value)


@torch.no_grad()
def _load_mtp_experts(model, root, index):
    """This EP rank's BF16 MTP experts, from the checkpoint (BF16 there; the
    BF16 release holds the same bytes, which is checked)."""
    experts_list = _mtp_experts(model)
    if not experts_list:
        return
    master_root = Path(model._bf16_master_root)
    source = json.loads((master_root / "model.safetensors.index.json").read_text())["weight_map"]
    for _, experts in experts_list:
        names = [
            name
            for expert in range(experts.num_local)
            for name in experts.hf_names(expert).values()
        ]
        missing = [name for name in names if name not in index or name not in source]
        if missing:
            raise ValueError(f"Missing MTP expert tensors: {missing[:4]}")
        device = experts.up_proj.device
        ours = _read_tensors(root, index, names, device)
        theirs = _read_tensors(master_root, source, names, device)
        for expert in range(experts.num_local):
            for projection, name in experts.hf_names(expert).items():
                value = ours[name]
                if value.dtype != torch.bfloat16 or not torch.equal(value, theirs[name]):
                    raise ValueError(f"MTP expert differs from the BF16 release: {name}")
                target = getattr(experts, projection)[expert]
                if value.shape != target.shape:
                    raise ValueError(f"MTP expert shape mismatch for {name}")
                target.copy_(value)
