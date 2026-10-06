"""mamba_ssm / causal_conv1d VJPs for the serving-visible Mamba2 kernels."""

import torch

from megatron.lite.model.nemotron_h.vllm.primitive.dense import (
    check_parameter_versions,
    parameter_versions,
)


def _shift(x, steps, seq_idx):
    """``x[t - steps]`` within each request, zero before its start."""
    if steps == 0:
        return x
    out = torch.zeros_like(x)
    same = seq_idx[steps:] == seq_idx[:-steps]
    out[steps:] = torch.where(same[:, None], x[:-steps], 0)
    return out


def _conv_weight_grads(x, weight, bias, grad, seq_idx):
    """FP32 weight/bias gradients of the causal SiLU convolution, reduced in a
    fixed order (causal_conv1d's backward accumulates them with atomics)."""
    width = weight.shape[1]
    x = x.float()
    pre = torch.zeros_like(x) if bias is None else bias.float().expand_as(x).clone()
    for k in range(width):
        pre += weight[:, k].float() * _shift(x, width - 1 - k, seq_idx)
    sigmoid = pre.sigmoid()
    g = grad.float() * sigmoid * (1 + pre * (1 - sigmoid))
    dweight = torch.stack(
        [(g * _shift(x, width - 1 - k, seq_idx)).sum(0) for k in range(width)], dim=1
    )
    return dweight, None if bias is None else g.sum(0)


class _PackedConvVJP(torch.autograd.Function):
    """Visible vLLM convolution; causal_conv1d backward on the packed batch."""

    @staticmethod
    def forward(ctx, visible, x, weight, bias, seq_idx):
        ctx.save_for_backward(x, weight, bias, seq_idx)
        ctx.versions = parameter_versions((weight,) if bias is None else (weight, bias))
        return visible(x, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        from causal_conv1d.cpp_functions import causal_conv1d_bwd_function

        x, weight, bias, seq_idx = ctx.saved_tensors
        check_parameter_versions(
            (weight,) if bias is None else (weight, bias), ctx.versions
        )
        dx, *_ = causal_conv1d_bwd_function(
            x.T[None],
            weight,
            bias,
            grad.contiguous().T[None],
            seq_idx,
            None,
            None,
            None,
            False,
            True,
        )
        dweight, dbias = _conv_weight_grads(x, weight, bias, grad, seq_idx[0])
        return None, dx[0].T, dweight, dbias, None


class _PackedScanVJP(torch.autograd.Function):
    """Visible vLLM SSD; mamba_ssm SSD backward on the packed batch."""

    @staticmethod
    def forward(ctx, visible, x, dt, A, B, C, D, dt_bias, seq_idx, chunk_size):
        output = visible(x, dt, A, B, C, D, dt_bias)
        ctx.save_for_backward(x, dt, A, B, C, D, dt_bias, seq_idx, output)
        ctx.chunk_size, ctx.versions = chunk_size, parameter_versions((D, dt_bias))
        return output

    @staticmethod
    def backward(ctx, grad):
        from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_bwd

        x, dt, A, B, C, D, dt_bias, seq_idx, output = ctx.saved_tensors
        check_parameter_versions((D, dt_bias), ctx.versions)
        # The output is only shape-checked without a gate z.
        dx, ddt, dA, dB, dC, dD, _, ddt_bias, _ = _mamba_chunk_scan_combined_bwd(
            grad[None],
            x[None],
            dt[None],
            A,
            B[None],
            C[None],
            output[None],
            ctx.chunk_size,
            D=D,
            dt_bias=dt_bias,
            seq_idx=seq_idx,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            state_dtype=torch.float32,
        )
        grads = (dx[0], ddt[0], dA, dB[0], dC[0], dD, ddt_bias)
        inputs = (x, dt, A, B, C, D, dt_bias)
        return None, *(g.to(t.dtype) for g, t in zip(grads, inputs)), None, None
