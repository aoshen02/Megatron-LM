"""Inference-visible arithmetic with native training VJPs for Nemotron-H."""

import torch


class _VisibleForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, native, *inputs):
        ctx.native = native
        ctx.save_for_backward(*inputs)
        return visible(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        with torch.enable_grad():
            inputs = tuple(
                x.detach().requires_grad_(required)
                for x, required in zip(
                    ctx.saved_tensors, ctx.needs_input_grad[2:], strict=True
                )
            )
            active = tuple(x for x in inputs if x.requires_grad)
            output = ctx.native(*inputs)
            gradients = iter(torch.autograd.grad(output, active, grad_outputs))
        return (
            None,
            None,
            *(next(gradients) if x.requires_grad else None for x in inputs),
        )


def visible_forward(visible, native, *inputs):
    """Keep inference rounding in forward; recompute the native VJP only in backward."""
    if not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs):
        return visible(*inputs)
    return _VisibleForward.apply(visible, native, *inputs)


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


class _VisibleLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, value, weight):
        ctx.save_for_backward(value)
        ctx.weight, ctx.version = weight, weight._version
        return visible(value)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weight._version != ctx.version:
            raise RuntimeError("Master weight changed between forward and backward")
        (value,) = ctx.saved_tensors
        return None, *native_linear_vjp(grad_output, value, ctx.weight)


def visible_linear(visible, value, weight):
    """Inference-visible forward; BF16 master-weight VJP for value and weight."""
    if not torch.is_grad_enabled() or not (value.requires_grad or weight.requires_grad):
        return visible(value)
    return _VisibleLinear.apply(visible, value, weight)


def linear(x, weight, bias=None):
    from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant

    inputs = (x, weight) if bias is None else (x, weight, bias)
    return visible_forward(linear_batch_invariant, torch.nn.functional.linear, *inputs)


def projection(x, module):
    """Dispatch quantized modules without bypassing their deployment and VJP."""
    from .fp8_training import Fp8TrainingLinear
    from .quantization import Nvfp4TrainingLinear

    if isinstance(module, Nvfp4TrainingLinear | Fp8TrainingLinear):
        return module(x)
    if isinstance(module, torch.nn.Linear):
        return linear(x, module.weight, module.bias)
    raise TypeError(f"Unsupported Nemotron projection: {type(module).__name__}")


class GatedRMSNorm(torch.nn.Module):
    def __init__(self, width, group_size, eps, *, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.group_size, self.eps = group_size, eps

    def forward(self, x, gate):
        from vllm.model_executor.models.nemotron_h_alignment import gated_forward

        def visible(x, gate, weight):
            return gated_forward(x, gate, weight, self.group_size, self.eps)

        def native(x, gate, weight):
            y = x.float() * torch.nn.functional.silu(gate.float())
            groups = y.unflatten(-1, (-1, self.group_size))
            groups = groups * torch.rsqrt(
                groups.square().mean(-1, keepdim=True) + self.eps
            )
            return weight * groups.flatten(-2).to(x.dtype)

        return visible_forward(visible, native, x, gate, self.weight)


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

            def native(x, weight):
                y = x.float()
                y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
                return weight * y.to(x.dtype)

            return visible_forward(visible, native, x, self.weight)

        def visible(x, residual, weight):
            return rms_forward(x, weight, self.eps, residual)

        def native(x, residual, weight):
            y = x.float() + residual.float()
            residual_out = y.to(weight.dtype)
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
            return y.to(weight.dtype) * weight, residual_out

        return visible_forward(visible, native, x, residual, self.weight)
