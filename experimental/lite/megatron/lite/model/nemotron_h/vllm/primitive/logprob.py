"""Selected log-probabilities: rollout value, chunked differentiable VJP (DS4)."""

import torch

from megatron.lite.model.nemotron_h.vllm.primitive.dense import projection


def aligned_selected_log_probs(
    hidden, lm_head, labels, temperature, chunk_size, *, calculate_entropy, tp_group
):
    """Rollout-kernel value with an FP32 log-softmax VJP on the same BF16 logits,
    one ``chunk_size`` token chunk at a time (DS4's ``aligned_selected_log_probs``).

    Returns ``(log_probs[T], entropy[T] or None)``.
    """
    from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy

    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    if chunk_size <= 0:
        raise ValueError("logprob chunk size must be positive")
    selected, entropy = [], []
    for start in range(0, hidden.shape[0], chunk_size):
        ids = labels[start : start + chunk_size, None]
        logits = projection(hidden[start : start + chunk_size], lm_head)
        if temperature != 1.0:
            logits = logits.float() / temperature
        with torch.no_grad():
            visible = compute_token_logprobs(logits, ids)
        if torch.is_grad_enabled():
            differentiable = logits.float().log_softmax(-1).gather(-1, ids.long())
            visible = visible + (differentiable - differentiable.detach())
        selected.append(visible.reshape(-1))
        if calculate_entropy:
            entropy.append(vocab_parallel_entropy(logits, tp_group))
    return torch.cat(selected), torch.cat(entropy) if calculate_entropy else None
