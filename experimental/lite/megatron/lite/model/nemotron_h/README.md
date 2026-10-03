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

- **Quantized layers keep the checkpoint format.** As in the DeepSeek-V4
  aligned actor, the serving kernels are called directly with tensors
  (`kernels.py`); no vLLM layer, config or process group is created. Weights
  pass once through vLLM's own preparation helpers (Humming repack,
  FlashInfer swizzle, scale inversion), in the order serving applies them.
  - Routed experts (W4A16 NVFP4, group 16) run the Humming indexed MoE kernel
    on packed FP4 weights with the serving EP4 one-sided reduction order.
  - NVFP4 linears run Humming dense GEMMs (shared expert: the FlashInfer
    CuTe-DSL GEMM); FP8 linears run the static FP8 input quant and vLLM's
    `flashinfer_scaled_fp8_mm` (FlashInfer `bmm_fp8`, CUTLASS under BI).
  - Attention runs the fixed-schedule FA4 kernel over an FP8 KV cache,
    replaying `vllm.model_executor.models.nemotron_h_fa4`.
- **Unquantized layers** (Mamba2, norms, embeddings, router) are BF16 and use
  vLLM's batch-invariant kernels; Mamba2 uses vLLM's exact-replay SSD.
- **Parameters.**
  - Each quantized weight is a BF16 master `Parameter` with FP32 optimizer
    main parameters, as in the DeepSeek-V4 aligned actor. The forward never
    reads it; it reads the FP4/FP8 deployment bytes.
  - The deployment starts from the checkpoint bytes. After every optimizer
    step the `post_optimizer_step_hook` gathers dist_opt's parameters and
    requantizes the masters with the
    checkpoint's own rule: Transformer Engine NVFP4 4over6 (E4M3 bound 256,
    squared-error choice, global amax/1536 per HF tensor) and FP8 per-tensor
    amax/448 with the quotient rounded through BF16 (ModelOpt's arithmetic).
  - With `impl_cfg.bf16_master_path` (the BF16 release the checkpoint was
    quantized from: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16@a9904d2` for
    `-NVFP4@bee7596`), the masters are loaded from that release instead of
    dequantized from the checkpoint, and the θ0 deployment is their
    requantization. Loading fails unless the release is the checkpoint's
    source: non-quantized tensors bitwise equal, every NVFP4/FP8 code the
    BF16 weight encoded on the checkpoint's scales, every NVFP4 global scale
    amax/1536. The checkpoint still provides the config and the static FP8
    activation and KV scales. θ0 differs from the checkpoint where ModelOpt's
    choices are not reproduced by Transformer Engine's 4over6 MSE quantizer
    and where FP8 scales were calibrated:
    - NVFP4: 0.228% of the block scales differ. Of those, 70% are exact ties
      (blocks that underflow to zero: TE writes scale 0, ModelOpt 2^-9; same
      values), 19% are E4M3 candidate scales rounding to the other side of a
      midpoint, 10% come from TE's encode-scale arithmetic and under 1% are
      near-ties; in addition some codes flip on TE's internal global scale,
      which is 1 ulp off amax/1536 for 23% of the tensors. The effective
      weights differ in 0.061% of the values.
    - FP8: the 24 of 46 projections whose calibrated scale is not amax/448
      use amax/448 from θ0 on (scheme A), so the first update has no scale
      jump; the per-tensor ratios are in the FP8 table below.
    Release identity: every tensor group the rank reads (a layer, or a
    top-level tensor) must match `bf16_release.json`, sha256 digests made
    after every shard matched its HF LFS sha256 at the pinned revision. A
    proxy cut names the release layer of each of its layers
    (`impl_cfg.bf16_master_layers`). The checks against the checkpoint above
    are compatibility validation only; a master moved within its FP4/FP8
    cells passes them.
  - The actor forward and the rollout export read the same bytes, so the
    rollout serves exactly the weights the actor computes with.
  - A training-checkpoint restore (`post_checkpoint_load_hook`) reinstalls
    the deployment bytes saved with the masters, the bytes last deployed:
    the θ0 deployment before the first update, requant(master) after it. It
    never requantizes.
- **Backward.** Transformer Engine `high_precision` semantics: BF16 GEMMs on
  the BF16 masters and the BF16 inputs (`functional.native_linear_vjp`), and
  grouped BF16 GEMMs for the routed experts from the visible FC1 output
  (`nvfp4_moe_vjp.routed_vjp`). Frozen per-input contract of the routed
  experts, as DeepSeek-V4's grouped MoE:
  - routing weights: `<dy, visible per-route expert output>` (exact);
  - token input, up and down weights: BF16-master VJP through the visible
    FC1 output (identity straight-through for the quantization); a token's
    route input gradients are summed in slot order and rounded to BF16 after
    each add, as DS4's deterministic scatter backward.

  Every other visible op has its own autograd Function, as in the
  DeepSeek-V4 actor: closed-form compiled FP32 VJPs for the RMSNorms, the
  mamba_ssm / causal_conv1d backwards for Mamba2, the deterministic
  FlashAttention varlen backward from the visible output and LSE for
  attention, and DS4's chunked selected log-probabilities.
- **Parallelism.** TP/EP/CP 1 with PP1 or PP4; the validated topology is PP4
  with `dist_opt`. The routed experts emulate the rollout's EP4 reduction on
  one rank.

### FP8 projections with a ModelOpt-calibrated scale

24 of the 46 FP8 Mamba projections carry a calibrated weight scale
that is not amax/448 of the BF16 release. Scheme A replaces it by amax/448 from
θ0 on; the θ0 codes then agree with the checkpoint's only where both grids
coincide.

| Tensor (`backbone.layers.`) | checkpoint scale / (amax/448) | θ0 codes equal |
|---|---|---|
| `0.mixer.in_proj` | 1.5000 | 0.098% |
| `11.mixer.in_proj` | 1.4967 | 0.016% |
| `14.mixer.out_proj` | 1.4962 | 0.032% |
| `16.mixer.out_proj` | 1.5033 | 0.037% |
| `18.mixer.in_proj` | 1.4970 | 0.018% |
| `2.mixer.out_proj` | 1.4969 | 0.038% |
| `21.mixer.out_proj` | 1.5000 | 0.036% |
| `23.mixer.in_proj` | 1.5000 | 0.016% |
| `25.mixer.in_proj` | 1.5000 | 0.015% |
| `28.mixer.in_proj` | 1.4959 | 0.013% |
| `28.mixer.out_proj` | 1.5000 | 0.031% |
| `30.mixer.out_proj` | 1.4979 | 0.032% |
| `35.mixer.in_proj` | 1.4979 | 0.013% |
| `37.mixer.out_proj` | 1.4957 | 0.029% |
| `39.mixer.in_proj` | 1.5000 | 0.021% |
| `39.mixer.out_proj` | 1.5000 | 0.035% |
| `4.mixer.out_proj` | 1.5054 | 0.023% |
| `41.mixer.in_proj` | 1.5000 | 0.017% |
| `41.mixer.out_proj` | 1.5020 | 0.032% |
| `44.mixer.in_proj` | 1.5000 | 0.017% |
| `44.mixer.out_proj` | 1.5000 | 0.037% |
| `48.mixer.out_proj` | 1.5000 | 0.036% |
| `50.mixer.out_proj` | 1.5000 | 0.038% |
| `9.mixer.in_proj` | 1.5000 | 0.015% |

## Limits

Validated on GB200 with the companion vLLM build. Quality, throughput and
full-depth distributed checkpoint resume are not covered here.

CPU suites live in `tests/unit/model/test_nemotron_*_unit.py`. CUDA cases
need the companion environment; skipped CUDA cases are not evidence of GPU
correctness.
