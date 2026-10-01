"""Pinned EP4 BI reductions for the unsharded Lightning trainer.

Explicit measured AG/RS and FlashInfer one-sided recipes, not arbitrary
collective-version parity. Reuses existing expert and ATen primitives.
"""

import inspect

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


def install_ep4_reduction(expert, recipe=EP4_REDUCTION):
    validate_reduction(recipe)
    if recipe is None:
        raise ValueError("An explicit EP4 reduction recipe is required")
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum

    if (
        type(expert).__name__ != "HummingIndexedExperts"
        or expert.num_experts != 128
        or expert.global_num_experts != 128
        or not envs.VLLM_BATCH_INVARIANT
    ):
        raise RuntimeError("EP4 reduction requires BI1 full128 indexed experts")
    if hasattr(expert, "_ep4_reduction_state"):
        raise RuntimeError("EP4 reduction already installed")
    original_apply = expert.apply
    signature = inspect.signature(original_apply)
    state = {"busy": False}
    maps = []
    for rank in range(4):
        mapping = torch.full((128,), -1, dtype=torch.int32, device=expert.locks.device)
        mapping[rank * 32 : (rank + 1) * 32] = torch.arange(32, device=mapping.device)
        maps.append(mapping)

    def aligned_apply(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        output, ids, weights = bound["output"], bound["topk_ids"], bound["topk_weights"]
        if state["busy"] or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EP4 reduction is eager and non-reentrant")
        if (
            output.dtype != torch.bfloat16
            or output.shape != (ids.shape[0], 2688)
            or ids.shape[1] != 6
        ):
            raise ValueError("Expected BF16 Lightning top6 geometry")
        if bound["expert_map"] is not None or bound["apply_router_weight_on_input"]:
            raise ValueError("Require unsharded trainer with output routing weights")
        original_forward = expert.humming_forward
        captured = {}

        def observe_forward(sublayer, *values, **options):
            result = original_forward(sublayer, *values, **options)
            if sublayer == "w2":
                if captured:
                    raise RuntimeError("Ambiguous repeated down projection")
                captured["outputs"] = options["outputs"]
            return result

        state["busy"] = True
        expert.humming_forward = observe_forward
        try:
            result = original_apply(*args, **kwargs)
            if "outputs" not in captured:
                raise RuntimeError("Down projection capture did not execute")
            per_route = captured["outputs"].view(ids.shape[0], 6, 2688)
            parts = [
                moe_fused_mul_sum(per_route, weights, topk_ids=ids, expert_map=mapping)
                for mapping in maps
            ]
            output.copy_(reduce_ep4_parts(parts, ids, recipe))
            return result
        finally:
            expert.humming_forward = original_forward
            state["busy"] = False

    expert.apply = aligned_apply
    expert._ep4_reduction_state = state
    return state
