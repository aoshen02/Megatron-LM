"""Static FP8 linear with inference-visible forward and BF16 master-weight VJP."""

import torch

from .quantization import requantize


class Fp8TrainingLinear(torch.nn.Module):
    """Static FP8 W/A forward over a BF16 master weight.

    The deployment starts from the checkpoint bytes (or, with a BF16 master
    source, from the master's requantization); after an optimizer update the
    weight is requantized per tensor (scale = amax / 448) and the calibrated
    input scale is kept. Backward is the BF16 master-weight VJP on the BF16
    input. Refresh outside Graph capture.
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
        self.register_buffer(
            "_packed", checkpoint.tensors["weight"].to(device).clone(), persistent=False
        )
        self._requantized = False
        self._deployed_version = self.weight._version

    def _tensors(self):
        return {
            "weight": self._packed,
            "weight_scale": self.weight_scale,
            "input_scale": self.input_scale,
        }

    def _check_fresh(self):
        if self.weight._version != self._deployed_version:
            raise RuntimeError("Refresh deployment after updating master weights")

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        """Once the master has been updated, requantize it."""
        if self.weight.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Refresh deployment outside CUDA Graph capture")
        self._requantized |= recompute_scales
        if self._requantized:
            tensors = requantize("FP8", self.weight)
            self._packed = tensors["weight"]
            self.weight_scale.copy_(tensors["weight_scale"].reshape(1))
        self._deployed_version = self.weight._version

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
        import vllm.envs as envs

        from .functional import visible_linear

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
        return visible_linear(self._visible, x, self.weight)
