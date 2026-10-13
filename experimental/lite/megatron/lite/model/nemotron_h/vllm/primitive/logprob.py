"""Selected log-probabilities: rollout value, chunked differentiable VJP (DS4)."""

import torch
from torch.utils.checkpoint import checkpoint

from megatron.lite.model.nemotron_h.vllm.primitive.dense import projection


def _chunk_logits(hidden, lm_head, temperature):
    logits = projection(hidden, lm_head)
    if temperature != 1.0:
        logits = logits.float() / temperature
    return logits


def _chunk_log_probs(hidden, lm_head, ids, temperature):
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    logits = _chunk_logits(hidden, lm_head, temperature)
    with torch.no_grad():
        visible = compute_token_logprobs(logits, ids)
    if torch.is_grad_enabled():
        differentiable = logits.float().log_softmax(-1).gather(-1, ids.long())
        visible = visible + (differentiable - differentiable.detach())
    return visible.reshape(-1)


def _chunk_entropy(hidden, lm_head, temperature, tp_group):
    from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy

    return vocab_parallel_entropy(_chunk_logits(hidden, lm_head, temperature), tp_group)


def aligned_selected_log_probs(
    hidden, lm_head, labels, temperature, chunk_size, *, calculate_entropy, tp_group
):
    """Rollout-kernel value with an FP32 log-softmax VJP on the same BF16 logits,
    one ``chunk_size`` token chunk at a time (DS4's ``aligned_selected_log_probs``).

    With gradients, each chunk's log-probabilities and entropy are separate
    checkpoints: their logits are recomputed in backward (twice in forward
    when both are needed) instead of kept, so memory does not grow with the
    sequence length.

    Returns ``(log_probs[T], entropy[T] or None)``.
    """
    from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy

    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    if chunk_size <= 0:
        raise ValueError("logprob chunk size must be positive")
    selected, entropy = [], []
    grad = torch.is_grad_enabled()
    for start in range(0, hidden.shape[0], chunk_size):
        chunk = hidden[start : start + chunk_size]
        ids = labels[start : start + chunk_size, None]
        if grad:
            selected.append(
                checkpoint(_chunk_log_probs, chunk, lm_head, ids, temperature,
                           use_reentrant=False)
            )
        else:
            logits = _chunk_logits(chunk, lm_head, temperature)
            selected.append(compute_token_logprobs(logits, ids).reshape(-1))
        if calculate_entropy:
            if grad:
                # A separate checkpoint: when the loss ignores the entropy its
                # recomputation never runs, so nothing of it stays alive.
                entropy.append(
                    checkpoint(_chunk_entropy, chunk, lm_head, temperature, tp_group,
                               use_reentrant=False)
                )
            else:
                entropy.append(vocab_parallel_entropy(logits, tp_group))
    return torch.cat(selected), torch.cat(entropy) if calculate_entropy else None
