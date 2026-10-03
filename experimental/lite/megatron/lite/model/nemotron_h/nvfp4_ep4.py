"""Pinned EP4 BI reductions for the unsharded Lightning trainer.

Explicit measured AG/RS and FlashInfer one-sided recipes, not arbitrary
collective-version parity. Reuses existing expert and ATen primitives.
"""

import torch

EP4_REDUCTION = "ep4-agrs-bf16-r0-r1r2r3-v1"
EP4_ONESIDED_REDUCTION = "ep4-fi-onesided-fp32-top6-first-rank-v1"


def validate_reduction(recipe):
    if recipe not in (None, EP4_REDUCTION, EP4_ONESIDED_REDUCTION):
        raise ValueError("Unknown routed forward reduction recipe")


def reduce_ep4_parts(parts, ids, recipe):
    """Combine rounded rank partials using the selected inference arithmetic."""
    validate_reduction(recipe)
    if (
        recipe is None
        or len(parts) != 4
        or ids.ndim != 2
        or ids.shape[1] != 6
        or any(
            p.dtype != torch.bfloat16 or p.ndim != 2 or p.shape != parts[0].shape
            for p in parts
        )
        or parts[0].shape[0] != ids.shape[0]
    ):
        raise ValueError("Require four BF16 rank partials and six routes per row")
    if recipe == EP4_REDUCTION:
        return parts[0] + ((parts[1] + parts[2]) + parts[3])
    owners = ids // 32
    stacked = torch.stack(parts)
    rows = torch.arange(ids.shape[0], device=ids.device)
    slots = []
    for slot in range(6):
        rank = owners[:, slot]
        duplicate = (owners[:, :slot] == rank[:, None]).any(dim=1)
        value = stacked[rank, rows]
        slots.append(torch.where(duplicate[:, None], 0, value).float())
    # FlashInfer TOP_K=6 combines first-occurrence rank slots in FP32.
    total = ((slots[0] + slots[1]) + (slots[2] + slots[3])) + (slots[4] + slots[5])
    return total.to(torch.bfloat16)


def ep4_routed_experts(layer, x, topk_weights, topk_ids, recipe, *, return_fc1=False):
    """Run unsharded Humming indexed experts with an EP4 serving combine.

    Calls the experts' own stages in the order ``HummingIndexedExperts.apply``
    does (no prepare quantization applies to W4A16), then replaces only its
    final ``moe_fused_mul_sum`` with the four rank partials an EP4 deployment
    computes and combines them in the selected serving order. ``return_fc1``
    also returns the visible FC1 output per route (token-major, slot-minor).
    """
    validate_reduction(recipe)
    if recipe is None:
        raise ValueError("An explicit EP4 reduction recipe is required")
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum

    experts = layer.quant_method.moe_kernel.fused_experts
    if (
        type(experts).__name__ != "HummingIndexedExperts"
        or experts.num_experts != 128
        or experts.global_num_experts != 128
        or not envs.VLLM_BATCH_INVARIANT
    ):
        raise RuntimeError("EP4 reduction requires BI1 full128 indexed experts")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("EP4 reduction is eager only")
    rows, topk = topk_ids.shape
    if x.dtype != torch.bfloat16 or x.shape != (rows, 2688) or topk != 6:
        raise ValueError("Expected BF16 Lightning top6 geometry")
    activation = layer.activation
    metas, required = experts.get_buffer_metas(rows, topk, activation)
    buffers = {
        name: torch.empty(metas[name]["shape"], dtype=metas[name]["dtype"], device=x.device)
        for name in required
        if name != "output"
    }
    w13_kwargs, w2_kwargs, scatter_idx = experts.prepare_humming_moe_kwargs(
        topk_ids=topk_ids, expert_map=None, expert_tokens_meta=None
    )
    inputs, scale, scale_2 = experts.process_input(
        "w13", inputs=x, input_scale=None,
        quanted_input=buffers["quanted_gate_up_input"],
    )
    experts.humming_forward(
        "w13", inputs=inputs, weight=layer.w13_weight, input_scale=scale,
        input_scale_2=scale_2, outputs=buffers["gate_up_output"], **w13_kwargs,
    )
    inputs, scale, scale_2 = experts.process_input(
        "w2", inputs=buffers["gate_up_output"],
        quanted_input=buffers["quanted_down_input"], activation=activation,
        scatter_idx=scatter_idx,
    )
    experts.humming_forward(
        "w2", inputs=inputs, weight=layer.w2_weight, input_scale=scale,
        input_scale_2=scale_2, outputs=buffers["down_output"].view(-1, x.shape[1]),
        **w2_kwargs,
    )
    per_route = buffers["down_output"].view(rows, topk, x.shape[1])
    parts = []
    for rank in range(4):
        mapping = torch.full((128,), -1, dtype=torch.int32, device=x.device)
        mapping[rank * 32 : (rank + 1) * 32] = torch.arange(32, device=x.device)
        parts.append(
            moe_fused_mul_sum(per_route, topk_weights, topk_ids=topk_ids, expert_map=mapping)
        )
    out = reduce_ep4_parts(parts, topk_ids, recipe)
    return (out, buffers["gate_up_output"]) if return_fc1 else out
