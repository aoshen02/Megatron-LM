"""Nemotron-H layer composition over the vLLM-visible primitives."""

from pathlib import Path

import torch
from torch import nn

from megatron.lite.model.nemotron_h.checkpoint import load_fp8_kv_scales
from megatron.lite.model.nemotron_h.vllm.primitive.attention.module import Attention
from megatron.lite.model.nemotron_h.vllm.primitive.dense import (
    CheckpointProjectionFactory,
    Fp8TrainingLinear,
    Nvfp4TrainingLinear,
    RMSNorm,
    projection,
    projection_layer,
)
from megatron.lite.model.nemotron_h.vllm.primitive.mamba.module import (
    MambaMixer,
    SSMMeta,
)
from megatron.lite.model.nemotron_h.vllm.primitive.moe.grouped import (
    Nvfp4ExpertWeights,
    Nvfp4RoutedDeployment,
)
from megatron.lite.model.nemotron_h.vllm.primitive.moe.module import MoE


class Block(nn.Module):
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
        routed_factory,
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
        elif self.kind == "moe":
            kwargs["routed_factory"] = routed_factory
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


class NemotronModel(nn.Module):
    """A contiguous pipeline stage with explicit hidden/residual boundary state.

    Intermediate stages return both streams as [tokens, 1, 2*hidden]. No broadcast
    or pipeline scheduler is implemented here; the mlite runtime must transport
    this payload and supply it through set_input_tensor.
    """

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
        routed_factory,
    ):
        super().__init__()
        self.config, self.ps = config, ps
        start, end = layer_range or (0, config.num_hidden_layers)
        if not 0 <= start < end <= config.num_hidden_layers:
            raise ValueError("Invalid contiguous layer range")
        if ps.pp_size > 1 and layer_range is None:
            raise ValueError("PP requires an explicit layer assignment from runtime")
        attention_layers = {
            i
            for i in range(start, end)
            if config.layers_block_type[i] == "full_attention"
        }
        if set(fp8_kv_scales) != attention_layers:
            raise ValueError(
                "FP8 KV scales must cover exactly the local attention layers"
            )
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
                str(i): Block(
                    config,
                    ps,
                    i,
                    device=device,
                    dtype=dtype,
                    projection_factory=projection_factory,
                    routed_factory=routed_factory,
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
            projection_layer(
                projection_factory,
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


def stage_quantization_contract(config, layer_range):
    """Select stage-owned recipe entries without discarding unknown prefixes."""
    start, end = layer_range
    if not 0 <= start < end <= config.num_hidden_layers:
        raise ValueError("Invalid quantized stage layer range")
    prefixes = set()
    for prefix in config.quantization_config["quantized_layers"]:
        if prefix == "lm_head":
            if end == config.num_hidden_layers:
                prefixes.add(prefix)
            continue
        parts = prefix.split(".")
        if (
            len(parts) < 5
            or parts[:2] != ["backbone", "layers"]
            or not parts[2].isdigit()
            or parts[3] != "mixer"
            or not 0 <= int(parts[2]) < config.num_hidden_layers
        ):
            raise ValueError(f"Unknown quantized stage prefix: {prefix}")
        if start <= int(parts[2]) < end:
            prefixes.add(prefix)
    attention_ids = [
        i for i in range(start, end) if config.layers_block_type[i] == "full_attention"
    ]
    return prefixes, attention_ids


def build_stage(config, impl, ps, *, layer_range):
    """Construct one pipeline stage from the checkpoint, before optimizer binding."""
    from vllm.utils.torch_utils import set_default_torch_dtype

    recipe = config.quantization_config
    factory = CheckpointProjectionFactory(impl.hf_path, recipe["quantized_layers"])

    def routed_factory(prefix, model_cfg, parallel, *, device, dtype):
        if dtype != torch.bfloat16 or parallel is not ps or model_cfg is not config:
            raise ValueError("Unexpected routed factory configuration")
        weights = Nvfp4ExpertWeights(
            impl.hf_path,
            prefix,
            recipe["quantized_layers"],
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            tp_size=ps.tp_size,
            ep_size=ps.ep_size,
            ep_rank=ps.ep_rank,
            device=device,
        )
        return Nvfp4RoutedDeployment(
            weights,
            config,
            ep_group=ps.ep_group if ps.ep_size > 1 else None,
        )

    expected_prefixes, attention_ids = stage_quantization_contract(config, layer_range)
    kv = load_fp8_kv_scales(impl.hf_path, attention_ids, device="cuda")
    with set_default_torch_dtype(torch.bfloat16):
        model = NemotronModel(
            config,
            ps,
            layer_range=layer_range,
            device="cuda",
            dtype=torch.bfloat16,
            projection_factory=factory,
            routed_factory=routed_factory,
            fp8_kv_scales=kv,
        )
    covered = set()
    for name, module in model.named_modules():
        prefix = name if name == "lm_head" else "backbone." + name
        if isinstance(module, Fp8TrainingLinear | Nvfp4TrainingLinear):
            covered.add(prefix)
        if isinstance(module, Nvfp4RoutedDeployment):
            covered.update(
                f"{prefix}.{e}.{projection}"
                for e in range(config.n_routed_experts)
                for projection in ("up_proj", "down_proj")
            )
    if covered != expected_prefixes:
        raise ValueError(
            "Quantized construction did not cover every stage recipe prefix"
        )
    model._hf_root = str(Path(impl.hf_path).resolve())
    model._bf16_master_root = str(Path(impl.bf16_master_path).resolve())
    return model
