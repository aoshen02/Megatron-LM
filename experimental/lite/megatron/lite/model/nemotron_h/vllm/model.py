"""Nemotron-H layer composition over the vLLM-visible primitives."""

from pathlib import Path

import torch
from torch import nn

from megatron.lite.model.nemotron_h.checkpoint import load_fp8_kv_scales
from megatron.lite.model.nemotron_h.vllm.primitive.attention.module import Attention
from megatron.lite.model.nemotron_h.vllm.primitive.dense import (
    CheckpointProjectionFactory,
    RMSNorm,
    projection,
)
from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import (
    MambaMixer,
    SSMMeta,
)
from megatron.lite.model.nemotron_h.vllm.primitive.moe.module import MoE


class NemotronHLayer(nn.Module):
    def __init__(
        self,
        config,
        ps,
        layer,
        *,
        device=None,
        dtype=torch.bfloat16,
        fp8_kv_scales,
        projection_factory,
    ):
        super().__init__()
        self.norm = RMSNorm(
            config.hidden_size, config.layer_norm_epsilon, device=device, dtype=dtype
        )
        self.kind = config.layers_block_type[layer]
        types = {
            "linear_attention": MambaMixer,
            "full_attention": Attention,
            "moe": MoE,
        }
        if self.kind not in types:
            raise ValueError(f"Unsupported Nemotron block: {self.kind}")
        kwargs = {}
        if self.kind == "full_attention":
            kwargs["fp8_kv_scales"] = fp8_kv_scales
        self.mixer = types[self.kind](
            config,
            ps,
            device=device,
            dtype=dtype,
            projection_factory=projection_factory,
            hf_prefix=f"backbone.layers.{layer}.mixer",
            **kwargs,
        )

    def forward(self, hidden, residual, meta):
        if residual is None:
            residual, hidden = hidden, self.norm(hidden)
        else:
            hidden, residual = self.norm(hidden, residual)
        hidden = self.mixer(hidden) if self.kind == "moe" else self.mixer(hidden, meta)
        return hidden, residual


class NemotronHModel(nn.Module):
    """A contiguous pipeline stage; intermediate stages return the hidden and
    residual streams as [tokens, 1, 2*hidden]."""

    def __init__(
        self,
        config,
        ps,
        *,
        layer_range=None,
        device=None,
        dtype=torch.bfloat16,
        fp8_kv_scales,
        projection_factory,
    ):
        super().__init__()
        self.config, self.ps = config, ps
        start, end = layer_range or (0, config.num_hidden_layers)
        if not 0 <= start < end <= config.num_hidden_layers:
            raise ValueError("Invalid contiguous layer range")
        self.pre_process = start == 0
        self.post_process = end == config.num_hidden_layers
        if config.tie_word_embeddings:
            raise ValueError("Tied embedding synchronization is not implemented")
        self.share_embeddings_and_output_weights = False
        self.embeddings = (
            nn.Embedding(
                config.vocab_size, config.hidden_size, device=device, dtype=dtype
            )
            if self.pre_process
            else None
        )
        self.layers = nn.ModuleDict(
            {
                str(i): NemotronHLayer(
                    config,
                    ps,
                    i,
                    device=device,
                    dtype=dtype,
                    projection_factory=projection_factory,
                    fp8_kv_scales=fp8_kv_scales.get(i),
                )
                for i in range(start, end)
            }
        )
        self.norm_f = (
            RMSNorm(
                config.hidden_size,
                config.layer_norm_epsilon,
                device=device,
                dtype=dtype,
            )
            if self.post_process
            else None
        )
        self.lm_head = (
            projection_factory(
                "lm_head",
                config.hidden_size,
                config.vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
            if self.post_process
            else None
        )
        self._input_tensor = None

    def set_input_tensor(self, tensor):
        if isinstance(tensor, list):
            if len(tensor) != 1:
                raise ValueError("Nemotron stage expects one packed pipeline payload")
            tensor = tensor[0]
        self._input_tensor = tensor

    def forward(self, input_ids, *, meta: SSMMeta, return_logits=True):
        if self.pre_process:
            hidden, residual = self.embeddings(input_ids.reshape(-1)), None
        else:
            if self._input_tensor is None:
                raise ValueError("Missing pipeline hidden/residual payload")
            if self._input_tensor.shape[1:] != (1, 2 * self.config.hidden_size):
                raise ValueError("Expected [tokens, 1, 2*hidden] pipeline payload")
            hidden, residual = self._input_tensor[:, 0].chunk(2, dim=-1)
            self._input_tensor = None
        for layer in self.layers.values():
            hidden, residual = layer(hidden, residual, meta=meta)
        if not self.post_process:
            return torch.cat((hidden, residual), dim=-1).unsqueeze(1)
        hidden, _ = self.norm_f(hidden, residual)
        return projection(hidden, self.lm_head) if return_logits else hidden


def build_stage(config, impl, ps, *, layer_range):
    """Construct one pipeline stage from the checkpoint, before optimizer binding."""
    from vllm.utils.torch_utils import set_default_torch_dtype

    factory = CheckpointProjectionFactory(
        impl.hf_path, config.quantization_config["quantized_layers"]
    )
    attention_ids = [
        i
        for i in range(*layer_range)
        if config.layers_block_type[i] == "full_attention"
    ]
    kv = load_fp8_kv_scales(impl.hf_path, attention_ids, device="cuda")
    with set_default_torch_dtype(torch.bfloat16):
        model = NemotronHModel(
            config,
            ps,
            layer_range=layer_range,
            device="cuda",
            dtype=torch.bfloat16,
            projection_factory=factory,
            fp8_kv_scales=kv,
        )
    model._hf_root = str(Path(impl.hf_path).resolve())
    model._bf16_master_root = str(Path(impl.bf16_master_path).resolve())
    return model
