"""Serving kernels called directly with tensors (DeepSeek-V4 aligned style).

Weights pass once through vLLM's own preparation helpers (Humming repack,
scale inversion and padding, FlashInfer swizzle), in the order the serving
ModelOpt methods apply them. Forwards call the kernels with tensors: no vLLM
layer, config, process group, forward context or workspace is involved.
"""

import json
import logging
import math
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

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


def _modelopt_global_scale(weight_scale, weight_scale_2):
    """ModelOpt KNvfp4Static.process: one FP32 global scale per matrix.

    Fails like serving on unloaded (NaN) group scales; Nemotron's checkpoint
    has one FP32 global scale per matrix.
    """
    if torch.isnan(weight_scale.float()).any():
        raise RuntimeError("NVFP4 weight_scale was never loaded (NaN)")
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


def torch_visible_devices():
    """CUDA_VISIBLE_DEVICES naming, by UUID, the devices torch enumerates."""
    return ",".join(
        f"GPU-{torch.cuda.get_device_properties(i).uuid}"
        for i in range(torch.cuda.device_count())
    )


def seed_device_capability(device):
    """Cache vLLM's capability lookup for ``device`` once, at actor build.

    Consumer: vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py
    ``moe_fused_mul_sum`` -> ``_heuristic_config(..., inputs.device.index)``
    -> ``current_platform.get_device_capability(device_index)`` (cached per
    ordinal), which maps the torch ordinal to a physical GPU through
    CUDA_VISIBLE_DEVICES. A verl Ray actor initializes CUDA before Ray
    narrows CUDA_VISIBLE_DEVICES to its GPU, so the variable no longer lists
    the devices torch enumerates. Only then, the lookup is resolved once with
    the variable naming torch's devices by UUID, which is right for any
    original (restricted or reordered) list. Call from the single-threaded
    model build only.
    """
    import os

    from vllm.platforms import current_platform

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or torch.cuda.device_count() == len(visible.split(",")):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = torch_visible_devices()
    try:
        current_platform.get_device_capability(device.index)
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible


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
            if torch.isnan(scale.float()).any():
                raise RuntimeError(f"NVFP4 {stem}_weight_scale was never loaded (NaN)")
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
                nemotron_humming_schedule,
            )

            # The launch schedule serving selects for a Lightning EP4 rank
            # (nemotron_h_moe.schedule_unsupported_reason).
            if (num_local, num_experts, hidden, intermediate) != (32, 128, 2688, 1856):
                raise ValueError("The Nemotron EP4 schedule is measured for Lightning")
            for name in ("w13", "w2"):
                table = getattr(self, f"{name}_tuning_config")
                setattr(self, f"{name}_tuning_config", nemotron_humming_schedule(table))
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
        # Serving's fallback (fused_humming_moe.prepare_humming_moe_kwargs).
        if not getattr(self, "_warned_block", False):
            logger.warning("No Humming tuning interval for M=%s; block 64", valid_shape_m)
            self._warned_block = True
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
        if rows == 0:
            return x.new_empty(0, self.intermediate), x.new_empty(0, topk, self.hidden)
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


# FlashInfer private W4A16 helpers the actor calls, and the launcher whose
# stage order it mirrors (flashinfer-python 0.7.0.post1,
# flashinfer/fused_moe/cute_dsl/blackwell/moe_w4a16.py :49/:59/:195/:358).
FLASHINFER_W4A16_SIGNATURES = {
    "_get_workspace": (
        "x", "top_k", "num_experts", "num_local_experts", "intermediate_size",
        "route_tile",
    ),
    "_run_grouped_gemm": (
        "weight", "weight_sf", "activations", "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit", "num_non_exiting_tiles", "alpha", "output",
        "num_local_experts", "activation_type", "swiglu_alpha", "swiglu_beta",
        "swiglu_limit", "situ_beta", "situ_linear_beta", "use_fused_finalize",
        "permuted_idx_to_expanded_idx", "token_final_scales", "enable_pdl", "tactic",
    ),
}
FLASHINFER_W4A16_SOURCE_SHA256 = {
    "_W4A16Workspace": "9a4534c8fd89a1093c3007874c77f39cfb200bdc74036f0f6f6a05d9e73aed87",
    "_get_workspace": "c5acedc5bb61dffbb894aba3d015ade3c301886d9e671cb08ac7cd09235b1fa4",
    "_run_grouped_gemm": "430265499f90cf3552e2dfbf2bc4d58649d6d814423563843579bd55df771fc1",
    "launch_w4a16_moe": "ae7abf3e50ea6ef411240f602a9b90980f8994faaefe10e47c5726c3c9abbc55",
}
_FLASHINFER_W4A16_CHECKED = False


def check_flashinfer_w4a16():
    """Fail unless FlashInfer's W4A16 helpers are the reviewed ones."""
    global _FLASHINFER_W4A16_CHECKED
    if _FLASHINFER_W4A16_CHECKED:
        return
    import hashlib
    import inspect

    from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16

    for name, parameters in FLASHINFER_W4A16_SIGNATURES.items():
        actual = tuple(inspect.signature(getattr(moe_w4a16, name)).parameters)
        if actual != parameters:
            raise RuntimeError(f"FlashInfer {name} signature changed: {actual}")
    for name, digest in FLASHINFER_W4A16_SOURCE_SHA256.items():
        source = inspect.getsource(getattr(moe_w4a16, name))
        if hashlib.sha256(source.encode()).hexdigest() != digest:
            raise RuntimeError(
                f"FlashInfer {name} changed; re-review CuteDslRoutedExperts "
                "before training with it"
            )
    _FLASHINFER_W4A16_CHECKED = True


class CuteDslRoutedExperts:
    """FlashInfer CuTe-DSL W4A16 ReLU2 experts (vLLM ``flashinfer_cutedsl``, BI).

    Serving runs ``launch_w4a16_moe`` (``CuteDslFusedMoEW4A16Runner``) with
    the batch-invariant tactic on each EP rank's 32 experts. This class runs
    the same stages in the same order over the ``num_local`` experts it holds
    (``offset`` onwards of ``num_experts``): ``moe_sort`` -> ``moe_permute``
    -> GEMM1 with the fused ReLU2 epilogue -> GEMM2 -> ``moe_unpermute``.
    Slots of experts held elsewhere get no permuted row, as on a serving
    rank. Every routed row is one expert tile's full-K FP32 accumulation, so
    its output does not depend on which other rows or experts share the
    launch.

    With ``return_fc1`` GEMM1 also runs with the identity epilogue (the
    variant GEMM2 uses): same kernel, tactic, tiles and K order, writing
    ``bf16(alpha * acc)``, the visible FC1 pre-activation for the VJP.

    Uses FlashInfer's ``_get_workspace`` and ``_run_grouped_gemm``
    (``flashinfer/fused_moe/cute_dsl/blackwell/moe_w4a16.py``); construction
    checks their signatures and source (``check_flashinfer_w4a16``).
    """

    TOP_K = 6

    def __init__(self, up, down, *, num_experts, offset=0):
        """``up``/``down`` are ``(packed, scale, global)`` checkpoint stacks."""
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
            BATCH_INVARIANT_TACTIC,
            prepare_w4a16_scales,
        )

        require_batch_invariance()
        check_flashinfer_w4a16()
        tensors = {}
        for stem, (packed, scale, global_scale) in (("w1", up), ("w2", down)):
            if torch.isnan(scale.float()).any():
                raise RuntimeError(f"NVFP4 {stem} weight_scale was never loaded (NaN)")
            if global_scale.dtype != torch.float32 or global_scale.numel() != len(packed):
                raise ValueError("Expected one FP32 NVFP4 global scale per expert")
            tensors[stem] = packed.detach().clone()
            with torch.cuda.device(packed.device):
                tensors[f"{stem}_sf"] = prepare_w4a16_scales(scale.detach())
            # Serving passes weight_scale_2 as the GEMM alpha.
            tensors[f"{stem}_alpha"] = global_scale.detach().reshape(-1).clone()
        self.tensors = tensors
        self.num_experts = len(up[0])
        self.global_num_experts = num_experts
        self.offset = offset
        if not 0 <= offset <= num_experts - self.num_experts:
            raise ValueError("Local experts out of range")
        self.intermediate = up[0].shape[1]
        self.hidden = down[0].shape[1]
        self.tactic = BATCH_INVARIANT_TACTIC

    def _gemm(self, stem, activations, output, meta, activation_type):
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import _run_grouped_gemm
        from flashinfer.tllm_enums import (
            DEFAULT_SWIGLU_ALPHA,
            DEFAULT_SWIGLU_BETA,
            DEFAULT_SWIGLU_LIMIT,
        )

        t = self.tensors
        _run_grouped_gemm(
            weight=t[stem],
            weight_sf=t[f"{stem}_sf"],
            activations=activations,
            tile_idx_to_expert_idx=meta["tile_idx_to_expert_idx"],
            tile_idx_to_mn_limit=meta["tile_idx_to_mn_limit"],
            num_non_exiting_tiles=meta["num_non_exiting_tiles"],
            alpha=t[f"{stem}_alpha"],
            output=output,
            num_local_experts=self.num_experts,
            activation_type=activation_type,
            swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
            swiglu_beta=DEFAULT_SWIGLU_BETA,
            swiglu_limit=DEFAULT_SWIGLU_LIMIT,
            situ_beta=None,
            situ_linear_beta=None,
            use_fused_finalize=False,
            permuted_idx_to_expanded_idx=None,
            token_final_scales=None,
            enable_pdl=True,
            tactic=self.tactic,
        )

    def _launch(self, x, routes, ids, *, return_fc1):
        """Run the launcher's stages up to GEMM2.

        Returns the permuted GEMM2 output, ``expanded_idx_to_permuted_idx``
        ``[M, topk]`` (-1 for slots of other experts), the permuted visible
        FC1 (or None) and the permuted fused activation.
        """
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import _get_workspace
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            get_max_num_permuted_tokens,
            moe_permute,
            moe_sort,
            normalize_cute_dsl_moe_activation_type,
        )
        from flashinfer.tllm_enums import ActivationType

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Routed experts are eager only")
        rows, topk = ids.shape
        if (
            topk != self.TOP_K
            or x.dtype != torch.bfloat16
            or x.shape != (rows, self.hidden)
            or ids.dtype != torch.int32
            or routes.dtype != torch.float32
            or routes.shape != ids.shape
        ):
            raise ValueError("Expected BF16 tokens, int32 ids, FP32 routes, top-6")
        relu2, _ = normalize_cute_dsl_moe_activation_type(ActivationType.Relu2)
        tile = self.tactic[0][1]
        local = self.num_experts
        workspace = _get_workspace(
            x, topk, self.global_num_experts, local, self.intermediate, tile
        )
        (t2e, t2lim, e2p, p2e, _, live) = moe_sort(
            token_selected_experts=ids,
            token_final_scales=routes,
            num_experts=self.global_num_experts,
            top_k=topk,
            local_expert_offset=self.offset,
            num_local_experts=local,
            tile_tokens_dim=tile,
            enable_pdl=True,
            **workspace.moe_sort_buffers,
        )
        slots = get_max_num_permuted_tokens(rows, topk, local, tile)
        meta = {
            "tile_idx_to_expert_idx": t2e[: slots // tile],
            "tile_idx_to_mn_limit": t2lim[: slots // tile],
            "num_non_exiting_tiles": live,
        }
        hidden = workspace.hidden_workspace[:slots]
        moe_permute(
            input=x,
            permuted_output=hidden,
            tile_idx_to_mn_limit=meta["tile_idx_to_mn_limit"],
            permuted_idx_to_expanded_idx=p2e[:slots],
            num_non_exiting_tiles=live,
            max_num_permuted_tokens=slots,
            top_k=topk,
            tile_size=tile,
            enable_pdl=True,
        )
        activated = workspace.intermediate[:slots]
        fc1 = None
        if return_fc1:
            fc1 = torch.empty_like(activated)
            self._gemm("w1", hidden, fc1, meta, None)
        self._gemm("w1", hidden, activated, meta, relu2)
        self._gemm("w2", activated, hidden, meta, None)
        return hidden, e2p[:rows], fc1, activated

    def _unpermute(self, hidden, e2p, routes):
        from flashinfer.fused_moe.cute_dsl.moe_utils import moe_unpermute

        rows, topk = e2p.shape
        part = hidden.new_empty(rows, self.hidden)
        moe_unpermute(
            permuted_input=hidden,
            output=part,
            expanded_idx_to_permuted_idx=e2p,
            topk_scales=routes,
            num_tokens=rows,
            top_k=topk,
            enable_pdl=True,
        )
        return part

    def rank_partial(self, x, routes, ids, *, save=False):
        """This EP rank's BF16 partial over the slots of its own experts.

        With ``save`` also returns, for those slots only (token-major,
        slot-minor), the visible FC1 pre-activation ``[S, I]``, expert output
        ``[S, H]`` and fused GEMM1 activation GEMM2 consumed ``[S, I]``.
        """
        rows = ids.shape[0]
        if rows == 0:
            part = x.new_empty(0, self.hidden)
            if not save:
                return part
            return part, x.new_empty(0, self.intermediate), part, x.new_empty(0, self.intermediate)
        hidden, e2p, fc1, activated = self._launch(x, routes, ids, return_fc1=save)
        part = self._unpermute(hidden, e2p, routes)
        if not save:
            return part
        index = e2p.reshape(-1)
        index = index.index_select(0, (index >= 0).nonzero().squeeze(1)).long()
        return (
            part,
            fc1.index_select(0, index),
            hidden.index_select(0, index),
            activated.index_select(0, index),
        )

    def ep_partials(self, x, routes, ids, *, ranks=4, return_fc1=False):
        """BF16 EP-rank partials of a deployment holding every expert; with
        ``return_fc1`` also, per route (token-major, slot-minor), the visible
        FC1 pre-activation ``[M*topk, I]``, expert output ``[M*topk, H]`` and
        the fused GEMM1 activation GEMM2 consumed ``[M*topk, I]``."""
        if self.num_experts != self.global_num_experts or self.num_experts % ranks:
            raise ValueError("EP partials need every expert on this deployment")
        hidden, e2p, fc1, activated = self._launch(x, routes, ids, return_fc1=return_fc1)
        owner = ids // (self.num_experts // ranks)
        parts = [
            self._unpermute(hidden, torch.where(owner == rank, e2p, -1), routes)
            for rank in range(ranks)
        ]
        if not return_fc1:
            return parts
        index = e2p.reshape(-1).long()
        return (
            parts,
            fc1.index_select(0, index),
            hidden.index_select(0, index),
            activated.index_select(0, index),
        )


def scaled_fp8_quant(x, scale):
    """Static per-tensor FP8 quantization (vLLM ``QuantFP8`` on CUDA)."""
    from vllm import _custom_ops as ops

    return ops.scaled_fp8_quant(x, scale)
