"""The rollout's EP4 combine (FlashInfer one-sided) for the Lightning trainer."""

import torch


def reduce_ep4_parts(parts, ids):
    """Combine the four BF16 rank partials as the FlashInfer one-sided combine."""
    if (
        len(parts) != 4
        or ids.ndim != 2
        or ids.shape[1] != 6
        or any(
            p.dtype != torch.bfloat16 or p.ndim != 2 or p.shape != parts[0].shape
            for p in parts
        )
        or parts[0].shape[0] != ids.shape[0]
    ):
        raise ValueError("Require four BF16 rank partials and six routes per row")
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


def ep4_routed_experts(experts, x, topk_weights, topk_ids, *, return_fc1=False):
    """Run all 128 experts locally with the rollout's EP4 combine.

    The four BF16 rank partials are the CuTe-DSL kernel's own per-rank top-k
    combine (``moe_unpermute``) over each rank's 32 experts. ``return_fc1``
    also returns, per route (token-major, slot-minor), the visible FC1, the
    expert output and the activation GEMM2 consumed.
    """
    if experts.num_experts != 128 or experts.global_num_experts != 128:
        raise RuntimeError("EP4 reduction requires full128 indexed experts")
    rows, topk = topk_ids.shape
    if x.dtype != torch.bfloat16 or x.shape != (rows, 2688) or topk != 6:
        raise ValueError("Expected BF16 Lightning top6 geometry")
    result = experts.ep_partials(x, topk_weights, topk_ids, return_fc1=return_fc1)
    if not return_fc1:
        return reduce_ep4_parts(result, topk_ids)
    parts, fc1, per_route, activated = result
    return reduce_ep4_parts(parts, topk_ids), fc1, per_route, activated
