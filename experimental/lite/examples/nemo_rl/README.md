# NeMo-RL Megatron Lite Example

`nemorl_mlite` runs a Megatron Lite policy inside NeMo-RL GRPO. With
`impl=vllm` the training forward uses the rollout's vLLM kernels, so the
recomputed logprobs equal the vLLM rollout logprobs bitwise under
`VLLM_BATCH_INVARIANT=1`.

## Layout

- `nemorl_mlite/policy_worker.py`: `MLitePolicyWorker`, the NeMo-RL policy
  worker. It reads `policy.external_cfg` and registers its actor environment
  on import.
- `nemorl_mlite/vllm_extension.py`: `MLiteVllmWorkerExtension`, the vLLM
  worker extension that refits a ModelOpt mixed-precision rollout (e.g. the
  Nemotron-H NVFP4 release) through vLLM's layerwise reload, keeps the objects
  captured CUDA graphs point to, and refits a co-trained MTP drafter.

## NeMo-RL settings

```yaml
policy:
  external_cfg:
    enabled: true
    worker_cls_fqn: nemorl_mlite.policy_worker.MLitePolicyWorker
    trains_mtp: true            # with impl_cfg.mtp_enable_train
    impl: vllm
    model_name: nemotron_h
    expert_parallel_size: 4
    impl_cfg: {...}             # Megatron Lite impl_cfg
    optimizer: {...}            # Megatron Lite OptimizerConfig
  generation:
    vllm_cfg:
      kv_scales_in_refit_weights: true
      internal_worker_extension_cls: nemorl_mlite.vllm_extension.MLiteVllmWorkerExtension
```

The NeMo-RL branch with these hooks, the Nemotron-H recipe and its image is
listed in [`REQUIRED_NEMO_RL.txt`](REQUIRED_NEMO_RL.txt).

## Install

```bash
pip install --no-deps experimental/lite/examples/nemo_rl
```
