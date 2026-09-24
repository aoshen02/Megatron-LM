# Experimental DSv4 inference precision

These default-off `TransformerConfig` options reduce numerical differences
between training and inference. They do not guarantee batch invariance or
bitwise agreement, and are not a throughput optimization.

| Option | Precision boundary |
| --- | --- |
| `csa_inference_kv_quantization` | Round the 448 NoPE KV channels to block-64 E4M3 values; preserve the 64 RoPE channels. |
| `csa_inference_projection_quantization` | Round output-projection inputs and weights with block-128 and block-128x128 power-of-two scales. |
| `csa_inference_rope_fp32` | Keep Q normalization and RoPE in FP32 until the attention/projection boundaries. |
| `csa_use_vllm_flashmla` | Use vLLM's bundled sparse FlashMLA forward and the existing cuDNN backward. |
| `mhc_inference_precision` | Keep mHC mappings and stream accumulation in FP32; return the original activation dtype. |

Set the desired options to `True` on the model configuration. The KV and
FlashMLA options require the unfused CSA path, `dsa_indexer_loss_coeff=0`,
`v_head_dim=512`, and `qk_pos_emb_head_dim=64`. FP32 RoPE additionally requires
fused RoPE and inference projection quantization. FP32 mHC aggregation does
not support a caller-owned output buffer.

Transformer Engine supplies straight-through gradients for quantization.
The grouped projection uses BF16 operands rounded to FP8 values, not a
hardware FP8 GEMM. The FlashMLA option needs a compatible installed vLLM
extension exposing `flash_mla_sparse_fwd`; it does not replace Top-K selection.
No additional dependency is imported when these options are disabled.

Validation includes quantization forward/gradient contracts, packed and
unpacked attention, context-parallel parity, and mHC checkpoint recomputation.
One fixed 4-layer rollout replay reduced mean absolute response logprob error
from approximately 0.06930 to 0.05017 with the combined options. This remains
above 0.05; long-training convergence and full-model acceptance are pending.
