# Nemotron-H (vLLM-aligned implementation)

Nemotron-H Lightning NVFP4 training whose forward is bitwise equal to vLLM
batch-invariant serving. It is registered as `impl="vllm"`, as DeepSeek-V4's
aligned implementation is; there is no `lite` implementation. Every compute
module calls vLLM kernels, so this package always requires the companion vLLM
build. With verl, select it with `actor_rollout_ref.actor.engine.impl=vllm`.

Only the ModelOpt `MIXED_PRECISION` Lightning checkpoint (NVFP4 experts and
linears, FP8 Mamba projections, static FP8 KV) is supported; `build_model`
rejects checkpoints without a `quantization_config`.

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
  - Each quantized weight is a BF16 master `Parameter` with FP32 optimizer
    main parameters, as in the DeepSeek-V4 aligned actor. The forward never
    reads it; it reads the FP4/FP8 deployment bytes.
  - The deployment starts from the checkpoint bytes. After every optimizer
    step the `post_optimizer_step_hook` requantizes the masters with the
    checkpoint's own rule: Transformer Engine NVFP4 4over6 (E4M3 bound 256)
    and FP8 per-tensor amax/448.
  - The actor forward and the rollout export read the same bytes, so the
    rollout serves exactly the weights the actor computes with.
- **Backward.** Transformer Engine `high_precision` semantics: BF16 GEMMs on
  the BF16 masters and the BF16 inputs (`functional.native_linear_vjp`), and
  grouped BF16 GEMMs for the routed experts from the visible FC1 output
  (`nvfp4_moe_vjp.routed_vjp`). Every other visible op has its own autograd
  Function, as in the DeepSeek-V4 actor: closed-form compiled FP32 VJPs for
  the RMSNorms, the mamba_ssm / causal_conv1d backwards for Mamba2, the
  FlashAttention varlen backward for attention and DS4's chunked selected
  log-probabilities.
- **Parallelism.** TP/EP/CP 1 with PP1 or PP4; the validated topology is PP4
  with `dist_opt`.

## Limits

Validated on GB200 with the companion vLLM build. Quality, throughput and
full-depth distributed checkpoint resume are not covered here.

CPU suites live in `tests/unit/model/test_nemotron_*_unit.py`. CUDA cases
need the companion environment; skipped CUDA cases are not evidence of GPU
correctness.
