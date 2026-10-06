"""Autograd bridges from vLLM-visible dense kernels to BF16-master VJPs.

The quantized linears (W4A16 NVFP4, FP8) call the serving kernels directly
with tensors, after vLLM's own weight preparation in serving order; no
vLLM layer, config, process group or forward context is created.
"""

import json

import torch

from megatron.lite.model.nemotron_h.quantization import (
    full_master,
    load_quantized_weight,
    master_version,
    requantize,
)


def native_linear_vjp(grad_output, value, weight):
    """BF16 dgrad/wgrad on the master weight (TE ``high_precision`` semantics)."""
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    x2d = value.reshape(-1, value.shape[-1]).contiguous()
    dy2d = grad_output.reshape(-1, grad_output.shape[-1]).to(value.dtype).contiguous()
    grad_value = general_gemm(
        weight.to(dy2d.dtype), dy2d, out_dtype=value.dtype, layout="NN", grad=True
    )[0]
    grad_weight = general_gemm(
        x2d, dy2d, out_dtype=weight.dtype, layout="NT", grad=True
    )[0]
    return grad_value.reshape(value.shape), grad_weight


def parameter_versions(parameters):
    return tuple(parameter._version for parameter in parameters)


def check_parameter_versions(parameters, expected):
    if parameter_versions(parameters) != expected:
        raise RuntimeError("Master weight changed between forward and backward")


class _VisibleLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, value, weight):
        ctx.save_for_backward(value, weight)
        ctx.versions = parameter_versions((weight,))
        return visible(value)

    @staticmethod
    def backward(ctx, grad_output):
        value, weight = ctx.saved_tensors
        check_parameter_versions((weight,), ctx.versions)
        return None, *native_linear_vjp(grad_output, value, weight)


def visible_linear(visible, value, weight):
    """Inference-visible forward; BF16 master-weight VJP for value and weight."""
    if not torch.is_grad_enabled() or not (value.requires_grad or weight.requires_grad):
        return visible(value)
    return _VisibleLinear.apply(visible, value, weight)


def linear(x, weight, bias=None):
    from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant

    if bias is not None:
        raise NotImplementedError("Nemotron BF16 projections are bias-free")
    return visible_linear(lambda x: linear_batch_invariant(x, weight), x, weight)


def projection(x, module):
    """Dispatch quantized modules without bypassing their deployment and VJP."""
    if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
        return module(x)
    if isinstance(module, torch.nn.Linear):
        return linear(x, module.weight, module.bias)
    raise TypeError(f"Unsupported Nemotron projection: {type(module).__name__}")


_COMPILED_SIGNATURES = {}
_RECOMPILE_LIMIT_HIT = set()


def _signature(args):
    return tuple(
        (arg.shape, arg.dtype) for arg in args if isinstance(arg, torch.Tensor)
    )


def compiled_vjp_or_eager(compiled, eager, *args):
    """``compiled`` for every input it compiled for, ``eager`` otherwise.

    With ``use_dynamic_bsz`` most backward shapes are new once the recompile
    limit is hit. Each such call would still enter dynamo, fail every guard and
    leave ~2 KiB in its global tables, so after the limit an input whose
    signature never ran compiled goes straight to ``eager``. The signature
    (shape, dtype) is a subset of the static-shape guards, so no input that
    matches a compiled entry is sent to eager.
    """
    if not any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args):
        return eager(*args)
    signature = _signature(args)
    compiled_signatures = _COMPILED_SIGNATURES.setdefault(compiled, set())
    if compiled in _RECOMPILE_LIMIT_HIT and signature not in compiled_signatures:
        return eager(*args)
    try:
        result = compiled(*args)
    except torch._dynamo.exc.FailOnRecompileLimitHit:
        _RECOMPILE_LIMIT_HIT.add(compiled)
        return eager(*args)
    compiled_signatures.add(signature)
    return result


def _rms_norm_vjp(grad_output, value, weight, eps):
    x = value.float()
    w = weight.float()
    grad = grad_output.float()
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    scaled_grad = grad * w
    correction = (scaled_grad * x).mean(dim=-1, keepdim=True)
    grad_value = (scaled_grad * rstd - x * rstd.pow(3) * correction).to(value.dtype)
    reduce_dims = tuple(range(grad.ndim - 1))
    grad_weight = (grad * x * rstd).sum(dim=reduce_dims).to(weight.dtype)
    return grad_value, grad_weight


def _residual_rms_norm_vjp(grad_output, grad_residual, x, residual, weight, eps):
    grad_sum, grad_weight = _rms_norm_vjp(
        grad_output, x.float() + residual.float(), weight, eps
    )
    grad_sum = grad_sum + grad_residual.float()
    return grad_sum.to(x.dtype), grad_sum.to(residual.dtype), grad_weight


def _gated_rms_norm_vjp(grad_output, x, gate, weight, group_size, eps):
    g = gate.float()
    sigmoid = torch.sigmoid(g)
    groups = (x.float() * g * sigmoid).unflatten(-1, (-1, group_size))
    rstd = torch.rsqrt(groups.square().mean(dim=-1, keepdim=True) + eps)
    normalized = groups * rstd
    grad = grad_output.float()
    scaled_grad = (grad * weight.float()).unflatten(-1, (-1, group_size))
    correction = (scaled_grad * normalized).mean(dim=-1, keepdim=True)
    grad_y = ((scaled_grad - normalized * correction) * rstd).flatten(-2)
    grad_x = (grad_y * g * sigmoid).to(x.dtype)
    grad_gate = (grad_y * x.float() * sigmoid * (1 + g * (1 - sigmoid))).to(gate.dtype)
    reduce_dims = tuple(range(grad.ndim - 1))
    grad_weight = (grad * normalized.flatten(-2)).sum(dim=reduce_dims).to(weight.dtype)
    return grad_x, grad_gate, grad_weight


# These compile on their first call inside backward, on the autograd engine's
# device thread. Inductor config overrides are thread-local, so that thread
# does not see the deterministic mode torch.use_deterministic_algorithms set
# on the main thread; its first compile would autotune reductions by timing
# (dynamic RBLOCK scaling) and pick the reduction order nondeterministically.
_VJP_COMPILE = dict(fullgraph=True, dynamic=False, options={"deterministic": True})
_compiled_rms_norm_vjp = torch.compile(_rms_norm_vjp, **_VJP_COMPILE)
_compiled_residual_rms_norm_vjp = torch.compile(_residual_rms_norm_vjp, **_VJP_COMPILE)
_compiled_gated_rms_norm_vjp = torch.compile(_gated_rms_norm_vjp, **_VJP_COMPILE)


class _RMSNormVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, value, weight, eps):
        ctx.save_for_backward(value, weight)
        ctx.eps, ctx.versions = eps, parameter_versions((weight,))
        return visible(value, weight)

    @staticmethod
    def backward(ctx, grad_output):
        value, weight = ctx.saved_tensors
        check_parameter_versions((weight,), ctx.versions)
        grad_value, grad_weight = compiled_vjp_or_eager(
            _compiled_rms_norm_vjp, _rms_norm_vjp, grad_output, value, weight, ctx.eps
        )
        return None, grad_value, grad_weight, None


class _ResidualRMSNormVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, x, residual, weight, eps):
        ctx.save_for_backward(x, residual, weight)
        ctx.eps, ctx.versions = eps, parameter_versions((weight,))
        return visible(x, residual, weight)

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        x, residual, weight = ctx.saved_tensors
        check_parameter_versions((weight,), ctx.versions)
        grads = compiled_vjp_or_eager(
            _compiled_residual_rms_norm_vjp,
            _residual_rms_norm_vjp,
            grad_output,
            grad_residual,
            x,
            residual,
            weight,
            ctx.eps,
        )
        return None, *grads, None


class _GatedRMSNormVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, x, gate, weight, group_size, eps):
        ctx.save_for_backward(x, gate, weight)
        ctx.group_size, ctx.eps = group_size, eps
        ctx.versions = parameter_versions((weight,))
        return visible(x, gate, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, gate, weight = ctx.saved_tensors
        check_parameter_versions((weight,), ctx.versions)
        grads = compiled_vjp_or_eager(
            _compiled_gated_rms_norm_vjp,
            _gated_rms_norm_vjp,
            grad_output,
            x,
            gate,
            weight,
            ctx.group_size,
            ctx.eps,
        )
        return None, *grads, None, None


class GatedRMSNorm(torch.nn.Module):
    def __init__(self, width, group_size, eps, *, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.group_size, self.eps = group_size, eps

    def forward(self, x, gate):
        from vllm.model_executor.layers.mamba.mamba_mixer2 import (
            grouped_gated_rms_norm,
        )

        def visible(x, gate, weight):
            return grouped_gated_rms_norm(x, gate, weight, self.group_size, self.eps)

        if not torch.is_grad_enabled():
            return visible(x, gate, self.weight)
        return _GatedRMSNormVJP.apply(
            visible, x, gate, self.weight, self.group_size, self.eps
        )


class RMSNorm(torch.nn.Module):
    """Inference rounding, including FP32 residual addition before normalization."""

    def __init__(self, width, eps, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x, residual=None):
        from vllm.model_executor.layers.layernorm import cuda_rms_norm

        if residual is None:

            def visible(x, weight):
                return cuda_rms_norm(x, weight, self.eps)

            if not torch.is_grad_enabled():
                return visible(x, self.weight)
            return _RMSNormVJP.apply(visible, x, self.weight, self.eps)

        def visible(x, residual, weight):
            return cuda_rms_norm(x, weight, self.eps, residual)

        if not torch.is_grad_enabled():
            return visible(x, residual, self.weight)
        return _ResidualRMSNormVJP.apply(visible, x, residual, self.weight, self.eps)


_CT_NVFP4 = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "type": "float",
    "num_bits": 4,
    "strategy": "group",
    "group_size": 16,
}


def require_batch_invariance():
    import vllm.envs as envs

    if not envs.VLLM_BATCH_INVARIANT or envs.VLLM_HUMMING_USE_F16_ACCUM:
        raise RuntimeError("Aligned Nemotron kernels require BI=1 and FP32 accumulation")


class _Holder(torch.nn.Module):
    """Layer-shaped carrier for vLLM's weight preparation helpers."""

    def __init__(self, tensors, **attributes):
        super().__init__()
        for name, tensor in tensors.items():
            setattr(self, name, torch.nn.Parameter(tensor, requires_grad=False))
        for name, value in attributes.items():
            setattr(self, name, value)


def _modelopt_global_scale(weight_scale, weight_scale_2):
    """ModelOpt KNvfp4Static.process: one FP32 global scale per matrix."""
    if weight_scale_2.dtype != torch.float32 or weight_scale_2.numel() != 1:
        raise ValueError("Expected one FP32 NVFP4 global scale")
    return weight_scale_2.max().to(torch.float32)


class HummingNvfp4Linear:
    """BI Humming dense W4A16 GEMM (ModelOpt W4A16 + HummingNvFp4LinearKernel)."""

    def __init__(self, weight, weight_scale, weight_scale_2):
        from vllm.model_executor.layers.quantization.utils.humming import (
            prepare_humming_linear_layer_config,
            quant_key_to_input_schema,
        )

        require_batch_invariance()
        n, packed_k = weight.shape
        holder = _Holder(
            {
                "weight_packed": weight.detach().clone(),
                "weight_scale": weight_scale.detach().clone(),
                "weight_global_scale": 1.0
                / _modelopt_global_scale(weight_scale, weight_scale_2),
            },
            input_size=packed_k * 2,
            output_partition_sizes=[n],
            params_dtype=torch.bfloat16,
            has_bias=False,
        )
        self.config = prepare_humming_linear_layer_config(
            holder, _CT_NVFP4, input_schema=quant_key_to_input_schema(None)
        )
        if self.config.input_quant_mode.should_quantize:
            raise RuntimeError("Unexpected activation quantization in W4A16")
        self.weight = holder.weight.data
        self.weight_scale = holder.weight_scale.data
        self.weight_scale_2 = getattr(holder, "weight_scale_2", None)
        if self.weight_scale_2 is not None:
            self.weight_scale_2 = self.weight_scale_2.data
        self.hadamard_block_size = holder.weight_schema.hadamard_block_size
        self.compute_config = json.dumps(
            {"use_batch_invariant": True, "use_f16_accum": False, "gemm_type": "dense"}
        )
        self.locks = torch.zeros(1024, dtype=torch.int32, device=weight.device)

    def __call__(self, x):
        from vllm.utils.humming import humming_forward

        output = humming_forward(
            self.config,
            inputs=x.reshape(-1, x.shape[-1]),
            weight=self.weight,
            weight_scale=self.weight_scale,
            zero_point=None,
            bias=None,
            weight_scale_2=self.weight_scale_2,
            input_scale=None,
            input_scale_2=None,
            hadamard_block_size=self.hadamard_block_size,
            locks=self.locks,
            compute_config=self.compute_config,
        )
        return output.view(*x.shape[:-1], output.shape[-1])


class CuteDslNvfp4Linear:
    """Shared-expert W4A16 GEMM (NemotronSharedNvFp4LinearKernel)."""

    def __init__(self, weight, weight_scale, weight_scale_2):
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            pad_nvfp4_weight_for_cutlass,
            swizzle_blockscale,
        )
        from vllm.utils.flashinfer import flashinfer_prepare_bf16_fp4_weights

        require_batch_invariance()
        scale = _modelopt_global_scale(weight_scale, weight_scale_2)
        # swizzle_blockscale allocates on the current device.
        with torch.cuda.device(weight.device):
            padded, self.padding = pad_nvfp4_weight_for_cutlass(
                weight.detach().clone(), alignment=64
            )
            self.weight, self.weight_scale, self.alpha = (
                flashinfer_prepare_bf16_fp4_weights(
                    padded,
                    swizzle_blockscale(weight_scale.detach().clone()),
                    scale.reshape(1),
                    backend="cute-dsl",
                )
            )
        # Humming stores the inverse scale and inverts it again during packing.
        self.alpha.copy_((1.0 / (1.0 / scale)).reshape_as(self.alpha))
        self.out_features = weight.shape[0]

    def __call__(self, x):
        import vllm.utils.flashinfer  # noqa: F401  (registers the op)
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            slice_nvfp4_output,
        )

        rows = x.reshape(-1, x.shape[-1])
        if self.padding:
            rows = torch.nn.functional.pad(rows, (0, self.padding * 2))
        out = torch.ops.vllm.flashinfer_mm_bf16_fp4(
            rows.contiguous(), self.weight, self.weight_scale, self.alpha
        )
        out = slice_nvfp4_output(out, self.out_features)
        return out.view(*x.shape[:-1], self.out_features)


class CheckpointProjectionFactory:
    """Strict checkpoint-aware projection construction, never module replacement.

    Plain BF16 tensors are validated here and loaded by the normal HF loader.
    Quantized adapters load their own checkpoint-domain tensors at construction.
    """

    def __init__(self, root, quantized_layers):
        from pathlib import Path

        self.root = Path(root)
        self.recipes = dict(quantized_layers)
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
        return build_quantized_projection(checkpoint, prefix, device=device)


class Nvfp4TrainingLinear(torch.nn.Module):
    """Inference-visible W4A16 linear over a BF16 master weight.

    The deployment starts from the checkpoint bytes and is requantized from the
    master after ``refresh_deployment(recompute_scales=True)``.
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
        self.register_buffer("_packed", checkpoint.tensors["weight"].to(device))
        self._factory = deployment_factory
        self._requantized = False
        self.bind_master()
        self._install()

    def bind_master(self):
        """Track the parameter the optimizer updates (its FSDP2 shard once wrapped);
        during an FSDP2 forward ``self.weight`` is the unsharded BF16 copy."""
        self.__dict__["_master"] = self._parameters["weight"]

    def _tensors(self):
        return {
            "weight": self._packed,
            "weight_scale": self.weight_scale,
            "weight_scale_2": self.weight_scale_2,
        }

    def _install(self):
        self._inference = self._factory(**self._tensors())
        self._deployed_version = master_version(self._master)

    def _check_fresh(self):
        if master_version(self._master) != self._deployed_version:
            raise RuntimeError("Refresh deployment after updating master weights")

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        """Reinstall; once the master has been updated, requantize it first."""
        self._requantized |= recompute_scales
        if self._requantized:
            tensors = requantize("W4A16_NVFP4", full_master(self._master))
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
        self._check_fresh()
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.weight.shape[-1]:
            raise ValueError("Expected BF16 activations with checkpoint K")
        return visible_linear(self._inference, x, self.weight)


class Fp8TrainingLinear(torch.nn.Module):
    """Static FP8 W/A forward over a BF16 master weight.

    Requantization replaces the weight and its scale (amax / 448) and keeps
    the calibrated input scale.
    """

    def __init__(self, checkpoint, *, device):
        super().__init__()
        if checkpoint.algorithm != "FP8":
            raise ValueError("Expected a tensor-scaled FP8 checkpoint")
        for name in ("input_scale", "weight_scale"):
            scale = checkpoint.tensors[name]
            if (
                scale.dtype != torch.float32
                or scale.numel() != 1
                or not torch.isfinite(scale).all()
                or not (scale > 0).all()
            ):
                raise ValueError(f"Expected positive scalar FP32 {name}")
        master = checkpoint.initial_master()
        if master.ndim != 2:
            raise ValueError("Expected matrix FP8 weights")
        self.weight = torch.nn.Parameter(master.to(device, torch.bfloat16))
        self._scale_shapes = {
            name: checkpoint.tensors[name].shape
            for name in ("input_scale", "weight_scale")
        }
        for name in ("input_scale", "weight_scale"):
            self.register_buffer(
                name, checkpoint.tensors[name].to(device).reshape(1).clone()
            )
        self.register_buffer("_packed", checkpoint.tensors["weight"].to(device).clone())
        self._requantized = False
        self.bind_master()
        self._deployed_version = master_version(self._master)

    def bind_master(self):
        self.__dict__["_master"] = self._parameters["weight"]

    def _tensors(self):
        return {
            "weight": self._packed,
            "weight_scale": self.weight_scale,
            "input_scale": self.input_scale,
        }

    def _check_fresh(self):
        if master_version(self._master) != self._deployed_version:
            raise RuntimeError("Refresh deployment after updating master weights")

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        """Once the master has been updated, requantize it."""
        self._requantized |= recompute_scales
        if self._requantized:
            tensors = requantize("FP8", full_master(self._master))
            self._packed = tensors["weight"]
            self.weight_scale.copy_(tensors["weight_scale"].reshape(1))
        self._deployed_version = master_version(self._master)

    def export_quantized(self):
        self._check_fresh()
        return {
            name: tensor.detach()
            .reshape(self._scale_shapes.get(name, tensor.shape))
            .clone()
            for name, tensor in self._tensors().items()
        }

    def _visible(self, x):
        from vllm import _custom_ops as ops
        from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm

        rows = x.reshape(-1, x.shape[-1])
        quantized, _ = ops.scaled_fp8_quant(rows.contiguous(), self.input_scale)
        output = flashinfer_scaled_fp8_mm(
            quantized, self._packed.T, self.input_scale, self.weight_scale, torch.bfloat16
        )
        return output.reshape(*x.shape[:-1], self._packed.shape[0])

    def forward(self, x):
        self._check_fresh()
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.weight.shape[-1]:
            raise ValueError("Expected BF16 activations with checkpoint K")
        return visible_linear(self._visible, x, self.weight)


def build_quantized_projection(checkpoint, prefix, *, device):
    """NVFP4 uses the serving kernels: CuTe-DSL for the shared expert, Humming
    elsewhere."""
    if checkpoint.algorithm == "FP8":
        return Fp8TrainingLinear(checkpoint, device=device)
    if checkpoint.algorithm != "W4A16_NVFP4":
        raise ValueError(f"Unsupported projection recipe: {checkpoint.algorithm}")
    shared = prefix.rsplit(".", 2)[-2:] in (
        ["shared_experts", "up_proj"],
        ["shared_experts", "down_proj"],
    )
    factory = CuteDslNvfp4Linear if shared else HummingNvfp4Linear
    return Nvfp4TrainingLinear(checkpoint, factory, device=device)
