"""Inference-visible forwards with dedicated training VJPs for Nemotron-H."""

import torch


def native_linear_vjp(grad_output, value, weight):
    """BF16 dgrad/wgrad on the master weight (TE ``high_precision`` semantics).

    Same arithmetic as the DeepSeek-V4 aligned actor (NVIDIA/Megatron-LM#7050).
    """
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
    from .fp8_training import Fp8TrainingLinear
    from .quantization import Nvfp4TrainingLinear

    if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
        return module(x)
    if isinstance(module, torch.nn.Linear):
        return linear(x, module.weight, module.bias)
    raise TypeError(f"Unsupported Nemotron projection: {type(module).__name__}")


def compiled_vjp_or_eager(compiled, eager, *args):
    if not any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args):
        return eager(*args)
    try:
        return compiled(*args)
    except torch._dynamo.exc.FailOnRecompileLimitHit:
        return eager(*args)


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


_compiled_rms_norm_vjp = torch.compile(_rms_norm_vjp, fullgraph=True, dynamic=False)
_compiled_residual_rms_norm_vjp = torch.compile(
    _residual_rms_norm_vjp, fullgraph=True, dynamic=False
)
_compiled_gated_rms_norm_vjp = torch.compile(
    _gated_rms_norm_vjp, fullgraph=True, dynamic=False
)


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
        from vllm.model_executor.models.nemotron_h_alignment import gated_forward

        def visible(x, gate, weight):
            return gated_forward(x, gate, weight, self.group_size, self.eps)

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
        from vllm.model_executor.models.nemotron_h_alignment import rms_forward

        if residual is None:

            def visible(x, weight):
                return rms_forward(x, weight, self.eps)

            if not torch.is_grad_enabled():
                return visible(x, self.weight)
            return _RMSNormVJP.apply(visible, x, self.weight, self.eps)

        def visible(x, residual, weight):
            return rms_forward(x, weight, self.eps, residual)

        if not torch.is_grad_enabled():
            return visible(x, residual, self.weight)
        return _ResidualRMSNormVJP.apply(visible, x, residual, self.weight, self.eps)
