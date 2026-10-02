# Nemotron-H (vLLM-aligned implementation)

Nemotron-H Lightning NVFP4 training whose forward is bitwise equal to vLLM
batch-invariant serving. It is registered as `impl="vllm"`, as DeepSeek-V4's
aligned implementation is; there is no `lite` implementation. Every compute
module calls vLLM kernels, so this package always requires the companion vLLM
build. With verl, select it with `actor_rollout_ref.actor.engine.impl=vllm`.

## What runs where

- **Quantized layers keep the checkpoint format.**
  - Routed experts (W4A16 NVFP4, group 16) run the vLLM Humming MoE kernel
    on packed FP4 weights with the serving EP4 one-sided reduction order.
  - NVFP4 and FP8 linears run vLLM's ModelOpt linear methods.
  - Attention runs the fixed-schedule FA4 kernel over an FP8 KV cache,
    replaying `vllm.model_executor.models.nemotron_h_fa4`.
- **Unquantized layers** (Mamba2, norms, embeddings, router) are BF16 and use
  vLLM's batch-invariant kernels; Mamba2 uses vLLM's exact-replay SSD.
- **Parameters.**
  - Each quantized weight is an FP32 master `Parameter`. The forward never
    reads it; it reads the FP4/FP8 encoding refreshed by the
    `post_optimizer_step_hook` after every successful optimizer step.
  - The actor forward and the rollout export read the same refreshed bytes,
    so the rollout serves exactly the weights the actor computes with.
- **Backward.**
  - Weight gradients use a straight-through estimator onto the FP32 masters.
  - The routed-expert VJP follows the surrogate contract
    `moe-fixedscale-grouped-bf16edges-v2`.
  - Sequence length is bounded by `routed_vjp_token_limit` (at most 16384).
- **Parallelism.** The validated topology is PP4 with `dist_opt`.

## Limits

Validated on GB200 with the companion vLLM build. Quality, throughput and
full-depth distributed checkpoint resume are not covered here.

CPU suites live in `tests/unit/model/test_nemotron_*_unit.py`. CUDA and
real-weight cases need the companion environment and `NEMOTRON_TEST_MODEL`;
skipped CUDA cases are not evidence of GPU correctness.
