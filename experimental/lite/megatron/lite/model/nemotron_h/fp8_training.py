"""Candidate fixed-scale FP8 linear with inference-visible forward and STE VJP."""

import torch

from .quantization import QuantizedWeight, grow_scales


class _Fp8LinearVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master, packed, dequantized, input_scale, weight_scale):
        from vllm import _custom_ops as ops
        from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm

        rows = x.reshape(-1, x.shape[-1])
        quantized, scale = ops.scaled_fp8_quant(rows.contiguous(), input_scale)
        input_dequantized = quantized.float() * scale
        # Save master/scales as well: normal optimizer/copy_ mutation before
        # backward must fail instead of using a mixed-policy VJP.
        ctx.save_for_backward(
            x,
            master,
            packed,
            dequantized,
            input_dequantized,
            input_scale,
            weight_scale,
        )
        output = flashinfer_scaled_fp8_mm(
            quantized, packed.T, input_scale, weight_scale, torch.bfloat16
        )
        return output.reshape(*x.shape[:-1], packed.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, master, packed, weight_ref, input_ref, sx, sw = ctx.saved_tensors
        grad = dy.reshape(-1, dy.shape[-1]).float()
        dx = (grad @ weight_ref).to(x.dtype).reshape_as(x)
        dw = grad.T @ input_ref
        return dx, dw, None, None, None, None


class Fp8TrainingLinear(torch.nn.Module):
    """Static FP8 W/A forward with candidate identity STE on both quantizers.

    Master weights and VJP arithmetic are FP32; activations, output and dX are
    BF16. dX uses dequantized deployment weights and dW uses the actual quantized
    then dequantized input. Scales are non-trainable buffers: the
    input scale stays calibrated; the weight scale stays the checkpoint's until
    the first optimizer update and afterwards only grows where the master overflows
    it (``grow_scales``). This module
    does not establish training quality or select the final gradient recipe.

    Refresh explicitly after updates, outside capture. Graph pointer stability,
    mutation through .data, and higher-order derivatives are unsupported.
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
        self.weight = torch.nn.Parameter(master.to(device).clone())
        self._scale_shapes = {
            name: checkpoint.tensors[name].shape
            for name in ("input_scale", "weight_scale")
        }
        for name in ("input_scale", "weight_scale"):
            self.register_buffer(
                name, checkpoint.tensors[name].to(device).reshape(1).clone()
            )
        self.register_buffer(
            "_packed",
            checkpoint.tensors["weight"].to(device).clone(),
            persistent=False,
        )
        self.register_buffer(
            "_dequantized", self.weight.detach().clone(), persistent=False
        )
        self._deployed_versions = self._versions()

    def _checkpoint(self):
        return QuantizedWeight(
            "FP8",
            {
                "weight": self._packed,
                "weight_scale": self.weight_scale,
                "input_scale": self.input_scale,
            },
        )

    def _versions(self):
        return tuple(
            (
                id(tensor),
                tensor.data_ptr(),
                tensor._version,
                tensor.dtype,
                tensor.device,
            )
            for tensor in (
                self.weight,
                self.input_scale,
                self.weight_scale,
                self._packed,
                self._dequantized,
            )
        )

    def _validate_types(self):
        if self.weight.dtype != torch.float32:
            raise ValueError("FP8 training master must remain FP32")
        if (
            self.input_scale.dtype != torch.float32
            or self.weight_scale.dtype != torch.float32
        ):
            raise ValueError("FP8 scales must remain FP32")

    def _check_fresh(self):
        self._validate_types()
        if self._versions() != self._deployed_versions:
            raise RuntimeError(
                "Refresh deployment after master, scale or device changes"
            )

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        self._validate_types()
        if self.weight.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Refresh deployment outside CUDA Graph capture")
        self._recompute_scales = getattr(self, "_recompute_scales", False)
        self._recompute_scales |= recompute_scales
        if self._recompute_scales:
            tensors = grow_scales("FP8", self.weight, {"weight_scale": self.weight_scale})
            self.weight_scale.copy_(tensors["weight_scale"].reshape(1))
        for scale in (self.input_scale, self.weight_scale):
            if not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError("FP8 scales must remain finite and positive")
        if self._recompute_scales:
            packed = tensors["weight"]
        else:
            packed = self._checkpoint().encode_master(self.weight)
        self._packed = packed.detach().clone()
        self._dequantized = self._packed.float() * self.weight_scale
        self._deployed_versions = self._versions()

    def export_quantized(self):
        self._check_fresh()
        return {
            name: tensor.detach()
            .reshape(self._scale_shapes.get(name, tensor.shape))
            .clone()
            for name, tensor in self._checkpoint().tensors.items()
        }

    def forward(self, x):
        import vllm.envs as envs

        if not envs.VLLM_BATCH_INVARIANT:
            raise RuntimeError("FP8 aligned training requires batch invariance")
        self._check_fresh()
        if (
            x.dtype != torch.bfloat16
            or x.ndim < 2
            or x.shape[-1] != self.weight.shape[-1]
            or not x.is_cuda
            or x.device != self.weight.device
        ):
            raise ValueError("Expected CUDA BF16 activations with checkpoint K/device")
        return _Fp8LinearVJP.apply(
            x,
            self.weight,
            self._packed,
            self._dequantized,
            self.input_scale,
            self.weight_scale,
        )
