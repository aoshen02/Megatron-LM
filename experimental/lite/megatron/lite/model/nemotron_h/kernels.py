"""Serving kernels called directly with tensors (DeepSeek-V4 aligned style).

Weights pass once through vLLM's own preparation helpers (Humming repack,
scale inversion and padding, FlashInfer swizzle), in the order the serving
ModelOpt methods apply them. Forwards call the kernels with tensors: no vLLM
layer, config, process group, forward context or workspace is involved.
"""

import json
import math
from types import SimpleNamespace

import torch

_CT_NVFP4 = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "type": "float",
    "num_bits": 4,
    "strategy": "group",
    "group_size": 16,
}
_MODELOPT_NVFP4 = {"quant_method": "modelopt", "quant_algo": "nvfp4"}


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


def _modelopt_global_scale(weight_scale_2):
    # ModelOpt KNvfp4Static.process: one FP32 global scale per matrix.
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
                "weight_global_scale": 1.0 / _modelopt_global_scale(weight_scale_2),
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
        scale = _modelopt_global_scale(weight_scale_2)
        padded, self.padding = pad_nvfp4_weight_for_cutlass(
            weight.detach().clone(), alignment=64
        )
        self.weight, self.weight_scale, self.alpha = flashinfer_prepare_bf16_fp4_weights(
            padded,
            swizzle_blockscale(weight_scale.detach().clone()),
            scale.reshape(1),
            backend="cute-dsl",
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


class HummingRoutedExperts:
    """BI Humming indexed W4A16 ReLU2 experts (ModelOpt NVFP4 MoE, Humming).

    Holds ``num_local`` experts of ``num_experts``; ids outside
    ``[offset, offset + num_local)`` are ignored through ``expert_map``,
    exactly as an EP rank of the serving deployment does.
    """

    def __init__(self, up, down, *, num_experts, offset, layer_name):
        """``up``/``down`` are ``(packed, scale, global)`` checkpoint stacks."""
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.utils.humming import (
            convert_to_humming_moe_kernel_format,
            get_humming_moe_quant_config,
        )
        from vllm.utils.humming import GemmType, get_heuristics_config

        require_batch_invariance()
        num_local, intermediate, packed_hidden = up[0].shape
        hidden = packed_hidden * 2
        device = up[0].device
        tensors = {}
        for stem, (packed, scale, global_scale) in (("w13", up), ("w2", down)):
            tensors[f"{stem}_weight"] = packed.detach().clone()
            tensors[f"{stem}_weight_scale"] = scale.detach().clone()
            tensors[f"{stem}_weight_scale_2"] = global_scale.detach().clone()
        self.activation = MoEActivation.RELU2_NO_MUL
        holder = _Holder(
            tensors,
            moe_config=SimpleNamespace(
                has_bias=False,
                num_local_experts=num_local,
                activation=self.activation,
                intermediate_size_per_partition=intermediate,
                hidden_dim=hidden,
            ),
            params_dtype=torch.bfloat16,
            layer_name=layer_name,
        )
        self.humming_configs = convert_to_humming_moe_kernel_format(
            holder, quant_config=_MODELOPT_NVFP4
        )
        for name in ("w13", "w2"):
            dtype = holder.input_schemas[name].a_dtype
            if dtype is not None and dtype.num_bits != 16:
                raise RuntimeError("Unexpected activation quantization in W4A16")
        self.quant_config = get_humming_moe_quant_config(holder)
        self.w13_weight = holder.w13_weight.data
        self.w2_weight = holder.w2_weight.data
        self.num_experts = num_local
        self.global_num_experts = num_experts
        self.offset = offset
        self.hidden, self.intermediate = hidden, intermediate
        self.locks = torch.zeros(1024, dtype=torch.int32, device=device)
        gemm_type = GemmType.INDEXED
        self.compute_config = {
            "use_batch_invariant": True,
            "use_f16_accum": False,
            "gemm_type": gemm_type.value,
        }
        for name in ("w13", "w2"):
            setattr(
                self,
                f"{name}_tuning_config",
                get_heuristics_config(
                    layer_config=self.humming_configs[name],
                    use_f16_accum=False,
                    use_batch_invariant=True,
                    gemm_type=gemm_type,
                ),
            )
        if num_local != num_experts:
            from vllm.model_executor.models.nemotron_h_moe import (
                configure_nemotron_humming,
            )

            # The serving EP4 rank's launch schedule.
            configure_nemotron_humming(self)
        self.expert_map = None
        if num_local != num_experts:
            self.expert_map = torch.full(
                (num_experts,), -1, dtype=torch.int32, device=device
            )
            self.expert_map[offset : offset + num_local] = torch.arange(
                num_local, dtype=torch.int32, device=device
            )

    def _block(self, name, valid_shape_m):
        for lower, upper, config in getattr(self, f"{name}_tuning_config"):
            if lower < valid_shape_m <= upper:
                return config["block_shape"][0]
        return 64

    def _process_input(self, name, inputs, outputs, *, activation=None, scatter_idx=None):
        from vllm.model_executor.layers.quantization.utils.humming.activation import (
            get_humming_activation,
        )
        from vllm.utils.humming import may_process_input

        prefix = "w1" if name == "w13" else "w2"
        config = self.humming_configs[name]
        mode = config.input_quant_mode
        scale = getattr(self.quant_config, f"{prefix}_input_scale")
        scale_2 = getattr(self.quant_config, f"{prefix}_input_scale_2")
        inputs, group_scales, token_scales = may_process_input(
            config,
            inputs=inputs,
            outputs=outputs,
            token_scales=scale_2 if mode.has_secondary_scale else scale,
            hadamard_block_size=getattr(
                self.quant_config, f"{prefix}_hadamard_block_size"
            ),
            layout="normal" if scatter_idx is None else "scatter",
            expert_tokens=None,
            scatter_idx=scatter_idx,
            num_valid_tokens=None,
            **(get_humming_activation(activation) if activation is not None else {}),
        )
        return (
            inputs,
            group_scales if mode.has_group_scale else token_scales,
            token_scales if mode.has_secondary_scale else None,
        )

    def _gemm(self, name, inputs, weight, scales, outputs, **kwargs):
        from vllm.utils.humming import humming_forward

        index = 1 if name == "w13" else 2
        q = self.quant_config
        return humming_forward(
            self.humming_configs[name],
            inputs=inputs,
            weight=weight,
            weight_scale=getattr(q, f"w{index}_scale"),
            zero_point=getattr(q, f"w{index}_zp"),
            bias=getattr(q, f"w{index}_bias"),
            weight_scale_2=getattr(q, f"g{index}_alphas"),
            input_scale=scales[0],
            input_scale_2=scales[1],
            outputs=outputs,
            locks=self.locks,
            **kwargs,
        )

    def routes(self, x, ids, *, global_tokens=None):
        """Per-route FC1 ``[M*topk, I]`` and down output ``[M, topk, H]``.

        ``global_tokens`` is the source token count summed over the EP group
        (the serving DP metadata); it only selects the launch schedule.
        """
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.utils.platform_utils import num_compute_units

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Routed experts are eager only")
        rows, topk = ids.shape
        tokens = rows if global_tokens is None else global_tokens
        valid_shape_m = math.ceil(
            tokens * topk * self.num_experts / self.global_num_experts
        )
        block = self._block("w13", valid_shape_m)
        scatter = (
            self.expert_map is not None
            and ids.numel() > 2 * num_compute_units(ids.get_device())
        )
        alignment = moe_align_block_size(
            topk_ids=ids,
            block_size=block,
            num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            ignore_invalid_experts=True,
            return_scatter_idx=scatter,
        )
        common = {
            "sorted_ids": alignment[0],
            "expert_ids": alignment[1],
            "num_tokens_padded": alignment[2],
            "compute_config": json.dumps(self.compute_config),
            "valid_shape_m": valid_shape_m,
        }
        scatter_idx = alignment[3] if len(alignment) == 4 else None
        kwargs1 = dict(common, top_k=topk, tuning_config=json.dumps(self.w13_tuning_config))
        kwargs2 = dict(common, top_k=1, tuning_config=json.dumps(self.w2_tuning_config))
        block2 = self._block("w2", valid_shape_m)
        if block2 != block:
            sorted_ids, expert_ids, padded = moe_align_block_size(
                topk_ids=ids,
                block_size=block2,
                num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                ignore_invalid_experts=True,
            )
            kwargs2.update(
                sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded
            )
        empty = torch.empty
        device = x.device
        fc1 = empty(rows * topk, self.intermediate, dtype=torch.bfloat16, device=device)
        down = empty(rows * topk, self.hidden, dtype=torch.bfloat16, device=device)
        inputs, *scales = self._process_input(
            "w13", x, empty(rows, self.hidden, dtype=torch.bfloat16, device=device)
        )
        self._gemm("w13", inputs, self.w13_weight, scales, fc1, **kwargs1)
        inputs, *scales = self._process_input(
            "w2",
            fc1,
            empty(rows * topk, self.intermediate, dtype=torch.bfloat16, device=device),
            activation=self.activation,
            scatter_idx=scatter_idx,
        )
        self._gemm("w2", inputs, self.w2_weight, scales, down, **kwargs2)
        return fc1, down.view(rows, topk, self.hidden)

    def rank_partial(self, down, weights, ids, expert_map):
        """The BF16 sum over the slots ``expert_map`` keeps, in slot order."""
        from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
            moe_fused_mul_sum,
        )

        return moe_fused_mul_sum(down, weights, topk_ids=ids, expert_map=expert_map)


def scaled_fp8_quant(x, scale):
    """Static per-tensor FP8 quantization (vLLM ``QuantFP8`` on CUDA)."""
    from vllm import _custom_ops as ops

    return ops.scaled_fp8_quant(x, scale)
