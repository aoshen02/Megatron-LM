"""Humming routed forward with an explicit diagnostic-only V2 surrogate."""

from copy import copy

import torch

from .nvfp4_experts import Nvfp4ExpertWeights
from .nvfp4_moe_vjp import (
    COMPACT_BACKEND,
    PADDED_BACKEND,
    SURROGATE_CONTRACT,
    RoutedSurrogateVJP,
    compact_kernel_path,
    validate_backend,
    validate_matmul_policy,
)


def validate_token_limit(backend, contract, token_limit):
    """Resolve the explicit diagnostic length bound without changing defaults."""
    if backend not in (PADDED_BACKEND, COMPACT_BACKEND):
        raise ValueError("Unknown routed VJP backend")
    if token_limit is None:
        return 512 if backend == COMPACT_BACKEND else 128
    if type(token_limit) is not int or token_limit not in (2048, 8192, 9216, 16384):
        raise ValueError(
            "Explicit routed VJP token limit must be 2048, 8192, 9216 or 16384"
        )
    if backend != COMPACT_BACKEND or contract != SURROGATE_CONTRACT:
        raise ValueError("Extended token limit requires compact and the V2 contract")
    return token_limit


class Nvfp4RoutedDeployment(torch.nn.Module):
    """Own a frozen W4A16 Humming deployment alongside FP32 expert masters.

    Caller initializes vLLM config, TP/EP1 groups and workspace before construction.
    Inputs are fixed expert IDs and continuous routing weights; this adapter does
    not route, normalize scores, apply routed_scaling_factor, or add shared output.
    No Graph/compile/concurrent-host dispatch support is claimed. Default forward
    rejects active autograd. The optional V2 surrogate is a bounded proxy
    experiment, not an approved training-quality policy or V1 parity claim.
    """

    def __init__(
        self, weights, model_config, vllm_config, quant_config,
        *, surrogate_contract=None, routed_vjp_backend=PADDED_BACKEND,
        routed_vjp_kernel_source=None, routed_vjp_token_limit=None,
        routed_forward_reduction=None,
        recompute_surrogate=False,
    ):
        super().__init__()
        from .nvfp4_ep4 import validate_reduction

        validate_reduction(routed_forward_reduction)
        self.routed_forward_reduction = routed_forward_reduction
        validate_token_limit(
            routed_vjp_backend, surrogate_contract, routed_vjp_token_limit
        )
        self.routed_vjp_token_limit = routed_vjp_token_limit
        if surrogate_contract not in (None, SURROGATE_CONTRACT):
            raise ValueError("Unknown routed surrogate contract")
        if surrogate_contract is not None and model_config.num_experts_per_tok != 6:
            raise ValueError("Diagnostic V2 surrogate requires top6")
        self.surrogate_contract = surrogate_contract
        if type(recompute_surrogate) is not bool or (
            recompute_surrogate and surrogate_contract != SURROGATE_CONTRACT
        ):
            raise ValueError("Surrogate recomputation requires explicit V2 training")
        self.recompute_surrogate = recompute_surrogate
        validate_backend(
            routed_vjp_backend, routed_vjp_kernel_source, surrogate_contract
        )
        self.routed_vjp_backend = routed_vjp_backend
        self.routed_vjp_kernel_source = (
            str(compact_kernel_path())
            if routed_vjp_backend == COMPACT_BACKEND
            else None
        )
        self.routed_vjp_execution_revision = (
            "compact-grouped-tf32-r4-f32-tma-base"
            if routed_vjp_backend == COMPACT_BACKEND else "padded-v2"
        )
        self._compact_adapter = None
        self._compact_busy = False
        self._surrogate_generation = 0
        if not isinstance(weights, Nvfp4ExpertWeights):
            raise TypeError("Expected Nvfp4ExpertWeights")
        if (
            model_config.n_routed_experts != weights.num_experts
            or weights._geometry["up_proj"]
            != (model_config.moe_intermediate_size, model_config.hidden_size)
            or model_config.mlp_hidden_act != "relu2"
            or model_config.mlp_bias
        ):
            raise ValueError("Expected matching bias-free ReLU2 expert geometry")
        if not 1 <= model_config.num_experts_per_tok <= weights.num_experts:
            raise ValueError("Invalid top-k expert count")
        if (routed_vjp_backend == COMPACT_BACKEND or routed_forward_reduction is not None) and (
            weights.num_experts, model_config.hidden_size,
            model_config.moe_intermediate_size, model_config.num_experts_per_tok,
        ) != (128, 2688, 1856, 6):
            raise ValueError("Compact diagnostic requires Lightning expert geometry")
        if weights.up_proj.device.type != "cuda":
            raise ValueError("Humming deployment requires CUDA checkpoint storage")
        self.weights = weights
        self.config = model_config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self._ready = False
        self._install()

    def _apply(self, fn, recurse=True):
        raise RuntimeError(
            "Construct deployment on its final device; do not convert its recipe"
        )

    def _validate_checkpoint(self):
        self.weights._validate_storage()
        if (
            self.weights._dirty
            or self.weights._versions() != self.weights._synced_versions
        ):
            raise RuntimeError(
                "Refresh deployment after changing expert masters/storage"
            )
        self.weights._validate_fixed_scales()

    def _validate_runtime(self):
        import vllm.envs as envs
        from vllm.distributed import get_ep_group, get_tp_group

        if not envs.VLLM_BATCH_INVARIANT:
            raise RuntimeError("Humming aligned expert deployment requires BI=1")
        if get_tp_group().world_size != 1 or get_ep_group().world_size != 1:
            raise RuntimeError("Routed deployment currently requires TP1/EP1")
        if self.vllm_config.kernel_config.moe_backend != "humming":
            raise RuntimeError("Explicit Humming MoE backend is required")

    @torch.no_grad()
    def _install(self):
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

        self._ready = False
        self._validate_checkpoint()
        self._validate_runtime()
        c, device = self.config, self.weights.up_proj.device
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Deployment construction cannot run during Graph capture"
            )
        # Each rebuilt deployment owns its registration, not the caller's model.
        deployment_config = copy(self.vllm_config)
        deployment_config.compilation_config = copy(self.vllm_config.compilation_config)
        deployment_config.compilation_config.static_forward_context = {}
        deployment_config.compilation_config.static_all_moe_layers = []
        with set_current_vllm_config(deployment_config), torch.device(device):
            runner = FusedMoEFactory(
                num_experts=c.n_routed_experts,
                top_k=c.num_experts_per_tok,
                hidden_size=c.hidden_size,
                intermediate_size=c.moe_intermediate_size,
                params_dtype=torch.bfloat16,
                quant_config=self.quant_config,
                prefix=self.weights.prefix,
                ckpt_names=("up_proj", "down_proj", ""),
                activation="relu2_no_mul",
                apply_router_weight_on_input=False,
                shared_experts=None,
                enable_eplb=False,
                num_redundant_experts=0,
                use_grouped_topk=True,
                num_expert_group=c.n_group,
                topk_group=c.topk_group,
                renormalize=c.norm_topk_prob,
                scoring_func="sigmoid",
                e_score_correction_bias=torch.zeros(
                    c.n_routed_experts, dtype=torch.float32
                ),
                routed_scaling_factor=c.routed_scaling_factor,
                apply_routed_scale_to_output=True,
                router_logits_dtype=torch.float32,
                skip_padding=True,
            )
            layer = runner.routed_experts
            method = layer.quant_method
            if (
                type(method).__name__ != "ModelOptNvFp4FusedMoE"
                or method.nvfp4_backend.value != "HUMMING"
                or not method.use_a16
                or layer.activation.value != "relu2_no_mul"
            ):
                raise RuntimeError("Expected Humming W4A16 ReLU2 routed implementation")
            # Copy one checkpoint-domain expert at a time. No full cloned export;
            # runtime transforms only own these destination tensors.
            for expert in range(self.weights.num_experts):
                for projection, stem in (("up_proj", "w13"), ("down_proj", "w2")):
                    checkpoint = self.weights._checkpoint(projection, expert)
                    for suffix, value in checkpoint.tensors.items():
                        target = getattr(layer, f"{stem}_{suffix}")[expert]
                        shape_ok = (
                            value.numel() == target.numel()
                            if suffix == "weight_scale_2"
                            else value.shape == target.shape
                        )
                        if not shape_ok or value.dtype != target.dtype:
                            raise ValueError(
                                "Humming checkpoint geometry mismatch: "
                                f"{projection}.{suffix}"
                            )
                        target.copy_(value.reshape(target.shape))
            layer.w13_input_scale.fill_(float("nan"))
            layer.w2_input_scale.fill_(float("nan"))
            method.process_weights_after_loading(layer)
            experts = method.moe_kernel.fused_experts
            if (
                "Humming" not in type(experts).__name__
                or experts.compute_config["use_batch_invariant"] is not True
                or getattr(layer, "w13_input_scale", None) is not None
                or getattr(layer, "w2_input_scale", None) is not None
                or hasattr(layer, "input_global_scale")
            ):
                raise RuntimeError("Humming deployment violated BI/W4A16 contract")
            for name in ("w13", "w2"):
                dtype = layer.input_schemas[name].a_dtype
                if dtype is not None and dtype.num_bits != 16:
                    raise RuntimeError("Unexpected activation quantization in W4A16")
            layer.requires_grad_(False)
        self._deployment = layer
        self._deployment_config = deployment_config
        self._deployed_versions = self.weights._versions()
        if self.surrogate_contract is not None:
            self._snapshot_surrogate()
        self._ready = True

    @torch.no_grad()
    def _snapshot_surrogate(self):
        if getattr(self, "recompute_surrogate", False):
            self._surrogate_generation += 1
            return
        # Allocate new snapshots so outstanding forwards retain their own values.
        for projection, name in (("up_proj", "up"), ("down_proj", "down")):
            snapshot = torch.empty_like(getattr(self.weights, projection))
            for expert in range(self.weights.num_experts):
                snapshot[expert].copy_(
                    self.weights._checkpoint(projection, expert).initial_master()
                )
            self.register_buffer(f"_surrogate_{name}", snapshot, persistent=False)
        self._surrogate_generation += 1

    @torch.no_grad()
    def _recompute_surrogate_weights(self):
        """Rebuild identical checkpoint-valued operands only for this VJP."""
        values = []
        for projection in ("up_proj", "down_proj"):
            value = torch.empty_like(getattr(self.weights, projection))
            for expert in range(self.weights.num_experts):
                value[expert].copy_(
                    self.weights._checkpoint(projection, expert).initial_master()
                )
            values.append(value)
        return tuple(values)

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        """Call after every optimizer/runtime update, including .data writes."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Deployment refresh cannot run during Graph capture")
        self._ready = False
        self.weights.refresh_quantized(recompute_scales=recompute_scales)
        self._install()

    def forward(self, x, ids, routing_weights):
        # Masters normally require grad, so even a detached x must not bypass
        # this gate. This diagnostic API requires caller-owned torch.no_grad().
        active_grad = torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (x, routing_weights, self.weights.up_proj, self.weights.down_proj)
        )
        if active_grad:
            if getattr(self, "surrogate_contract", None) != SURROGATE_CONTRACT:
                raise RuntimeError(
                    "Routed training VJP is not approved; use diagnostic no_grad"
                )
            self._validate_vjp_length(x.shape[0])
            validate_matmul_policy(x.device)
            return RoutedSurrogateVJP.apply(
                x, self.weights.up_proj, self.weights.down_proj,
                routing_weights, ids, self,
            )
        return self._visible(x, ids, routing_weights)

    def _validate_vjp_length(self, tokens):
        backend = getattr(self, "routed_vjp_backend", PADDED_BACKEND)
        limit = validate_token_limit(
            backend, getattr(self, "surrogate_contract", None),
            getattr(self, "routed_vjp_token_limit", None),
        )
        if not 1 <= tokens <= limit:
            raise ValueError(
                f"Diagnostic {backend} surrogate is limited to M1..{limit}"
            )

    def _routed_vjp(self, x, up, down, routes, ids, dy):
        from . import nvfp4_moe_vjp

        self._validate_vjp_length(x.shape[0])
        validate_matmul_policy(x.device)
        if getattr(self, "routed_vjp_backend", PADDED_BACKEND) == PADDED_BACKEND:
            return nvfp4_moe_vjp.grouped_vjp(x, up, down, routes, ids, dy)
        if not x.is_cuda or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Compact diagnostic requires eager CUDA execution")
        if self._compact_busy:
            raise RuntimeError("Compact diagnostic forbids concurrent dispatch")
        from .nvfp4_compact_vjp import GroupedTF32, compact_vjp

        self._compact_busy = True
        try:
            if self._compact_adapter is None:
                self._compact_adapter = GroupedTF32(self.routed_vjp_kernel_source)
            return compact_vjp(self._compact_adapter, x, up, down, routes, ids, dy)
        finally:
            # Keep launch metadata alive until all work completes, including errors.
            try:
                torch.cuda.synchronize(x.device)
                if self._compact_adapter is not None:
                    self._compact_adapter.release_metadata()
            finally:
                self._compact_busy = False

    def _visible(self, x, ids, routing_weights):
        self._validate_checkpoint()
        if not self._ready or self.weights._versions() != self._deployed_versions:
            raise RuntimeError("Refresh deployment before routed forward")
        self._validate_runtime()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Diagnostic routed forward has no Graph capture contract"
            )
        c, device = self.config, self.weights.up_proj.device
        if (
            x.ndim != 2
            or x.shape[1] != c.hidden_size
            or x.dtype != torch.bfloat16
            or ids.shape != (x.shape[0], c.num_experts_per_tok)
            or routing_weights.shape != ids.shape
            or ids.dtype != torch.int32
            or routing_weights.dtype != torch.float32
            or any(
                t.device != device or not t.is_contiguous()
                for t in (x, ids, routing_weights)
            )
        ):
            raise ValueError(
                "Expected contiguous BF16 X, int32 IDs and FP32 route weights"
            )
        if (
            not torch.isfinite(x).all()
            or not torch.isfinite(routing_weights).all()
            or (ids < 0).any()
            or (ids >= c.n_routed_experts).any()
        ):
            raise ValueError("Nonfinite input or out-of-range expert IDs")
        from vllm.forward_context import (
            is_forward_context_available,
            set_forward_context,
        )

        if is_forward_context_available():
            raise RuntimeError(
                "Diagnostic adapter must own an unnested forward context"
            )
        if x.shape[0] == 0:
            return torch.empty_like(x)
        with set_forward_context(None, self._deployment_config, num_tokens=x.shape[0]):
            if self.routed_forward_reduction is None:
                out = self._deployment.quant_method.apply(
                    self._deployment, x, routing_weights, ids, None, None
                )
            else:
                from .nvfp4_ep4 import ep4_routed_experts

                out = ep4_routed_experts(
                    self._deployment, x, routing_weights, ids,
                    self.routed_forward_reduction,
                )
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != x.shape
            or out.dtype != x.dtype
        ):
            raise RuntimeError("Unexpected Humming routed output contract")
        return out
