# Nemotron-H NVFP4 alignment review snapshot

This change stacks on the native BF16 adapter in
[PR 226](https://github.com/ISEEKYAN/Megatron-LM/pull/226), at
`3594b1a883ed10e80b03d9c71a4f23ce9db2e825`. It adds the tested diagnostic
quantized actor implementation, not a general Megatron-Core quantization backend.
The existing Lite runtime is unchanged. No model weights, container binaries,
dataset files or machine-specific source paths are shipped.

## Implementation boundary

- Explicit HF mixed-precision metadata selects FP8 projections, NVFP4 W4A16
  shared/routed experts and static FP8 KV attention. BF16 configuration remains
  available without quantization metadata.
- FP32 master parameters retain identity while deployment tensors are refreshed
  after loading and optimizer steps. Visible quantized forward and the declared
  fixed-scale surrogate backward are distinct contracts; gradients need not be
  bitwise equal to an unquantized model's gradients.
- `moe-fixedscale-grouped-bf16edges-v2` declares the routed surrogate policy.
  The padded implementation and packaged compact CuTeDSL implementation are
  separate backends. The compact kernel uses **TF32 MMA** with FP32 storage,
  accumulation and output, not full FP32 multiplication.
- Quantized construction is restricted to the Lightning geometry, TP/ETP/EP/CP/
  VPP 1, four/five-layer operator-covering proxies, or explicitly selected
  52-layer PP4 diagnostics. The caller owns distributed groups, vLLM config,
  workspace and process lifecycle.
- `experimental/lite/pyproject.toml` builds a normal training wheel containing
  the Lite namespace and two Mamba helpers. It does not package Megatron-Core;
  matching Core, Torch/CUDA, vLLM, Transformers, Humming and CuTeDSL dependencies
  must be installed and independently pinned in the unified image.

## Blocking cleanup before production acceptance

**This Draft still contains runtime method replacement. It is not the final
overlay-free delivery.** `nvfp4_ep4.install_ep4_reduction` replaces an expert's
`apply` method and temporarily replaces `humming_forward` to capture down
projection outputs. It is eager-only and non-reentrant. Move this arithmetic
into a normal expert subclass/factory with an explicit per-route output API;
preserve BF16 edges, first-occurrence owner ordering and the FP32 sum tree,
then rerun numerical and lifecycle gates before removing the old implementation.

`mamba_chunk_native` also creates a private callable by transforming the exact
hash-pinned Transformers SSD source. It changes four contraction allocations,
without globally replacing the HF function. This remains a source-version
dependency and fails closed on other source bytes; a maintained explicit native
implementation with attribution and equivalent-gradient tests is preferable.
The packaged NVIDIA grouped kernel retains its original BSD-3-Clause notices
and source hash; formatting it would invalidate the current artifact contract.

## Evidence and remaining gates

Prior unified-image diagnostics reported three SSD VJP cases plus 48 contraction
cases bitwise equal to independent references, eight shared-projection cases,
and six synthetic FP8 attention cases. A five-layer 8192/1024 proxy through the
installed verl engine reported 1024 raw FP32 response logprobs equal to rollout
and completed backward, **without an optimizer update**. These are historical
diagnostic receipts, not a fresh validation of this review branch.

Fresh checks for this branch are listed in its PR description. CPU metadata
and reduction tests cannot establish GPU arithmetic parity. Current compact
VJP independent validation, real variable-length DAPO actor batches, nonzero
optimizer/master deltas, resident online weight sync, two real RL updates,
full-model quality and same-image BI=0/BI=1 performance remain unaccepted.
Do not substitute disk rollout replay for that online RL acceptance gate.
