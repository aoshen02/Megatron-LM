"""Pinned EP4 BI reductions for the unsharded Lightning trainer.

EP4 serving reduction orders (AG/RS and FlashInfer one-sided), not arbitrary
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
    # FlashInfer 0.7.0 one-sided combine (csrc/nv_internal/tensorrt_llm/kernels/
    # communicationKernels/moeAlltoAllKernels.cu): dispatch sends each token once
    # per target rank, from the first top-k slot naming it (:495); combine loads
    # that rank's partial for the first slot and FP32 zero for duplicates
    # (:971); TOP_K=6 sums ((s0+s1)+(s2+s3))+(s4+s5) in FP32 (:1104-1115).
    total = ((slots[0] + slots[1]) + (slots[2] + slots[3])) + (slots[4] + slots[5])
    return total.to(torch.bfloat16)


def ep4_routed_experts(experts, x, topk_weights, topk_ids, recipe, *, return_fc1=False):
    """Run all 128 Humming experts locally with an EP4 serving combine.

    Forms the four BF16 rank partials an EP4 deployment computes (each rank's
    ``moe_fused_mul_sum`` over its 32 experts) and combines them in the
    selected serving order. ``return_fc1`` also returns the visible FC1 and
    expert outputs per route (token-major, slot-minor).
    """
    validate_reduction(recipe)
    if recipe is None:
        raise ValueError("An explicit EP4 reduction recipe is required")
    if experts.num_experts != 128 or experts.global_num_experts != 128:
        raise RuntimeError("EP4 reduction requires full128 indexed experts")
    rows, topk = topk_ids.shape
    if x.dtype != torch.bfloat16 or x.shape != (rows, 2688) or topk != 6:
        raise ValueError("Expected BF16 Lightning top6 geometry")
    fc1, per_route = experts.routes(x, topk_ids)
    parts = []
    for rank in range(4):
        mapping = torch.full((128,), -1, dtype=torch.int32, device=x.device)
        mapping[rank * 32 : (rank + 1) * 32] = torch.arange(32, device=x.device)
        parts.append(experts.rank_partial(per_route, topk_weights, topk_ids, mapping))
    out = reduce_ep4_parts(parts, topk_ids, recipe)
    return (out, fc1, per_route.view(-1, x.shape[1])) if return_fc1 else out
