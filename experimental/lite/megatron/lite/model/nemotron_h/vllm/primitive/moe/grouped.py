"""FlashInfer CuTe-DSL W4A16 routed experts with a BF16-master grouped backward."""

import json
from pathlib import Path

import torch

from megatron.lite.model.nemotron_h.quantization import (
    QuantizedWeight,
    full_master,
    load_quantized_weight,
    master_version,
    requantize,
)
from megatron.lite.model.nemotron_h.vllm.primitive.dense import require_batch_invariance


# FlashInfer private W4A16 helpers the actor calls, and the launcher whose
# stage order it mirrors (flashinfer-python 0.7.0.post1).
FLASHINFER_W4A16_SIGNATURES = {
    "_get_workspace": (
        "x", "top_k", "num_experts", "num_local_experts", "intermediate_size",
        "route_tile",
    ),
    "_run_grouped_gemm": (
        "weight", "weight_sf", "activations", "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit", "num_non_exiting_tiles", "alpha", "output",
        "num_local_experts", "activation_type", "swiglu_alpha", "swiglu_beta",
        "swiglu_limit", "situ_beta", "situ_linear_beta", "use_fused_finalize",
        "permuted_idx_to_expanded_idx", "token_final_scales", "enable_pdl", "tactic",
    ),
}
FLASHINFER_W4A16_SOURCE_SHA256 = {
    "_W4A16Workspace": "9a4534c8fd89a1093c3007874c77f39cfb200bdc74036f0f6f6a05d9e73aed87",
    "_get_workspace": "c5acedc5bb61dffbb894aba3d015ade3c301886d9e671cb08ac7cd09235b1fa4",
    "_run_grouped_gemm": "430265499f90cf3552e2dfbf2bc4d58649d6d814423563843579bd55df771fc1",
    "launch_w4a16_moe": "ae7abf3e50ea6ef411240f602a9b90980f8994faaefe10e47c5726c3c9abbc55",
}
_FLASHINFER_W4A16_CHECKED = False


def check_flashinfer_w4a16():
    """Fail unless FlashInfer's W4A16 helpers are the reviewed ones."""
    global _FLASHINFER_W4A16_CHECKED
    if _FLASHINFER_W4A16_CHECKED:
        return
    import hashlib
    import inspect

    from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16

    for name, parameters in FLASHINFER_W4A16_SIGNATURES.items():
        actual = tuple(inspect.signature(getattr(moe_w4a16, name)).parameters)
        if actual != parameters:
            raise RuntimeError(f"FlashInfer {name} signature changed: {actual}")
    for name, digest in FLASHINFER_W4A16_SOURCE_SHA256.items():
        source = inspect.getsource(getattr(moe_w4a16, name))
        if hashlib.sha256(source.encode()).hexdigest() != digest:
            raise RuntimeError(
                f"FlashInfer {name} changed; re-review CuteDslRoutedExperts "
                "before training with it"
            )
    _FLASHINFER_W4A16_CHECKED = True


class CuteDslRoutedExperts:
    """FlashInfer CuTe-DSL W4A16 ReLU2 experts (vLLM ``flashinfer_cutedsl``, BI).

    Runs ``launch_w4a16_moe``'s stages in serving order over the ``num_local``
    experts held from ``offset``: ``moe_sort`` -> ``moe_permute`` -> GEMM1
    with the fused ReLU2 epilogue -> GEMM2 -> ``moe_unpermute``. With
    ``return_fc1`` GEMM1 also runs with the identity epilogue (same kernel,
    tactic and K order), giving the visible FC1 pre-activation for the VJP.
    """

    TOP_K = 6

    def __init__(self, up, down, *, num_experts, offset=0):
        """``up``/``down`` are ``(packed, scale, global)`` checkpoint stacks."""
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
            BATCH_INVARIANT_TACTIC,
            prepare_w4a16_scales,
        )

        require_batch_invariance()
        check_flashinfer_w4a16()
        tensors = {}
        for stem, (packed, scale, global_scale) in (("w1", up), ("w2", down)):
            if torch.isnan(scale.float()).any():
                raise RuntimeError(f"NVFP4 {stem} weight_scale was never loaded (NaN)")
            if global_scale.dtype != torch.float32 or global_scale.numel() != len(packed):
                raise ValueError("Expected one FP32 NVFP4 global scale per expert")
            tensors[stem] = packed.detach().clone()
            with torch.cuda.device(packed.device):
                tensors[f"{stem}_sf"] = prepare_w4a16_scales(scale.detach())
            # Serving passes weight_scale_2 as the GEMM alpha.
            tensors[f"{stem}_alpha"] = global_scale.detach().reshape(-1).clone()
        self.tensors = tensors
        self.num_experts = len(up[0])
        self.global_num_experts = num_experts
        self.offset = offset
        if not 0 <= offset <= num_experts - self.num_experts:
            raise ValueError("Local experts out of range")
        self.intermediate = up[0].shape[1]
        self.hidden = down[0].shape[1]
        self.tactic = BATCH_INVARIANT_TACTIC

    def _gemm(self, stem, activations, output, meta, activation_type):
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import _run_grouped_gemm
        from flashinfer.tllm_enums import (
            DEFAULT_SWIGLU_ALPHA,
            DEFAULT_SWIGLU_BETA,
            DEFAULT_SWIGLU_LIMIT,
        )

        t = self.tensors
        _run_grouped_gemm(
            weight=t[stem],
            weight_sf=t[f"{stem}_sf"],
            activations=activations,
            tile_idx_to_expert_idx=meta["tile_idx_to_expert_idx"],
            tile_idx_to_mn_limit=meta["tile_idx_to_mn_limit"],
            num_non_exiting_tiles=meta["num_non_exiting_tiles"],
            alpha=t[f"{stem}_alpha"],
            output=output,
            num_local_experts=self.num_experts,
            activation_type=activation_type,
            swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
            swiglu_beta=DEFAULT_SWIGLU_BETA,
            swiglu_limit=DEFAULT_SWIGLU_LIMIT,
            situ_beta=None,
            situ_linear_beta=None,
            use_fused_finalize=False,
            permuted_idx_to_expanded_idx=None,
            token_final_scales=None,
            enable_pdl=True,
            tactic=self.tactic,
        )

    def _launch(self, x, routes, ids, *, return_fc1):
        """Run the launcher's stages up to GEMM2.

        Returns the permuted GEMM2 output, ``expanded_idx_to_permuted_idx``
        ``[M, topk]`` (-1 for slots of other experts), the permuted visible
        FC1 (or None) and the permuted fused activation.
        """
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import _get_workspace
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            get_max_num_permuted_tokens,
            moe_permute,
            moe_sort,
            normalize_cute_dsl_moe_activation_type,
        )
        from flashinfer.tllm_enums import ActivationType

        rows, topk = ids.shape
        if (
            topk != self.TOP_K
            or x.dtype != torch.bfloat16
            or x.shape != (rows, self.hidden)
            or ids.dtype != torch.int32
            or routes.dtype != torch.float32
            or routes.shape != ids.shape
        ):
            raise ValueError("Expected BF16 tokens, int32 ids, FP32 routes, top-6")
        relu2, _ = normalize_cute_dsl_moe_activation_type(ActivationType.Relu2)
        tile = self.tactic[0][1]
        local = self.num_experts
        workspace = _get_workspace(
            x, topk, self.global_num_experts, local, self.intermediate, tile
        )
        (t2e, t2lim, e2p, p2e, _, live) = moe_sort(
            token_selected_experts=ids,
            token_final_scales=routes,
            num_experts=self.global_num_experts,
            top_k=topk,
            local_expert_offset=self.offset,
            num_local_experts=local,
            tile_tokens_dim=tile,
            enable_pdl=True,
            **workspace.moe_sort_buffers,
        )
        slots = get_max_num_permuted_tokens(rows, topk, local, tile)
        meta = {
            "tile_idx_to_expert_idx": t2e[: slots // tile],
            "tile_idx_to_mn_limit": t2lim[: slots // tile],
            "num_non_exiting_tiles": live,
        }
        hidden = workspace.hidden_workspace[:slots]
        moe_permute(
            input=x,
            permuted_output=hidden,
            tile_idx_to_mn_limit=meta["tile_idx_to_mn_limit"],
            permuted_idx_to_expanded_idx=p2e[:slots],
            num_non_exiting_tiles=live,
            max_num_permuted_tokens=slots,
            top_k=topk,
            tile_size=tile,
            enable_pdl=True,
        )
        activated = workspace.intermediate[:slots]
        fc1 = None
        if return_fc1:
            fc1 = torch.empty_like(activated)
            self._gemm("w1", hidden, fc1, meta, None)
        self._gemm("w1", hidden, activated, meta, relu2)
        self._gemm("w2", activated, hidden, meta, None)
        return hidden, e2p[:rows], fc1, activated

    def _unpermute(self, hidden, e2p, routes):
        from flashinfer.fused_moe.cute_dsl.moe_utils import moe_unpermute

        rows, topk = e2p.shape
        part = hidden.new_empty(rows, self.hidden)
        moe_unpermute(
            permuted_input=hidden,
            output=part,
            expanded_idx_to_permuted_idx=e2p,
            topk_scales=routes,
            num_tokens=rows,
            top_k=topk,
            enable_pdl=True,
        )
        return part

    def rank_partial(self, x, routes, ids, *, save=False):
        """This EP rank's BF16 partial over the slots of its own experts.

        With ``save`` also returns, for those slots only (token-major,
        slot-minor), the visible FC1 pre-activation ``[S, I]``, expert output
        ``[S, H]`` and fused GEMM1 activation GEMM2 consumed ``[S, I]``.
        """
        rows = ids.shape[0]
        if rows == 0:
            part = x.new_empty(0, self.hidden)
            if not save:
                return part
            return part, x.new_empty(0, self.intermediate), part, x.new_empty(0, self.intermediate)
        hidden, e2p, fc1, activated = self._launch(x, routes, ids, return_fc1=save)
        part = self._unpermute(hidden, e2p, routes)
        if not save:
            return part
        index = e2p.reshape(-1)
        index = index.index_select(0, (index >= 0).nonzero().squeeze(1)).long()
        return (
            part,
            fc1.index_select(0, index),
            hidden.index_select(0, index),
            activated.index_select(0, index),
        )

    def ep_partials(self, x, routes, ids, *, ranks=4, return_fc1=False):
        """BF16 EP-rank partials of a deployment holding every expert; with
        ``return_fc1`` also, per route (token-major, slot-minor), the visible
        FC1 pre-activation ``[M*topk, I]``, expert output ``[M*topk, H]`` and
        the fused GEMM1 activation GEMM2 consumed ``[M*topk, I]``."""
        if self.num_experts != self.global_num_experts or self.num_experts % ranks:
            raise ValueError("EP partials need every expert on this deployment")
        hidden, e2p, fc1, activated = self._launch(x, routes, ids, return_fc1=return_fc1)
        owner = ids // (self.num_experts // ranks)
        parts = [
            self._unpermute(hidden, torch.where(owner == rank, e2p, -1), routes)
            for rank in range(ranks)
        ]
        if not return_fc1:
            return parts
        index = e2p.reshape(-1).long()
        return (
            parts,
            fc1.index_select(0, index),
            hidden.index_select(0, index),
            activated.index_select(0, index),
        )


def _is_dtensor(tensor):
    from torch.distributed.tensor import DTensor

    return isinstance(tensor, DTensor)


class Nvfp4ExpertWeights(torch.nn.Module):
    """BF16 masters and NVFP4 deployment bytes of this EP rank's experts.

    EP rank ``r`` owns the contiguous experts ``[r * E / EP, (r + 1) * E / EP)``.
    The deployment bytes are the checkpoint's until the first
    ``refresh_quantized(recompute_scales=True)``; from then on every refresh
    requantizes the masters (``quantization.requantize``).
    """

    def __init__(
        self,
        root,
        prefix,
        quantized_layers,
        *,
        num_experts,
        hidden_size,
        intermediate_size,
        tp_size=1,
        ep_size=1,
        ep_rank=0,
        device="cpu",
    ):
        super().__init__()
        if tp_size != 1 or num_experts % ep_size or not 0 <= ep_rank < ep_size:
            raise ValueError("NVFP4 experts require TP1 and EP dividing the experts")
        if (
            not isinstance(prefix, str)
            or not prefix.startswith("backbone.layers.")
            or not prefix.endswith(".mixer.experts")
        ):
            raise ValueError("Expected explicit HF routed-experts prefix")
        if (
            min(num_experts, hidden_size, intermediate_size) <= 0
            or hidden_size % 16
            or intermediate_size % 16
        ):
            raise ValueError(
                "Both expert contraction dimensions must be positive multiples of 16"
            )
        self.prefix, self.num_experts = prefix, num_experts
        self.num_local = num_experts // ep_size
        self.offset = ep_rank * self.num_local
        self._geometry = {
            "up_proj": (intermediate_size, hidden_size),
            "down_proj": (hidden_size, intermediate_size),
        }
        names = {
            f"{prefix}.{expert}.{projection}"
            for expert in range(num_experts)
            for projection in ("up_proj", "down_proj")
        }
        actual = {key for key in quantized_layers if key.startswith(prefix + ".")}
        if actual != names:
            raise ValueError("Recipes must cover exactly all requested up/down experts")
        for name in names:
            recipe = quantized_layers[name]
            if (
                recipe.get("quant_algo") != "W4A16_NVFP4"
                or recipe.get("group_size") != 16
            ):
                raise ValueError("Every routed projection must use W4A16_NVFP4 group16")
        root = Path(root)
        mapping = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        suffixes = ("weight", "weight_scale", "weight_scale_2")
        expected_keys = {f"{name}.{suffix}" for name in names for suffix in suffixes}
        if {key for key in mapping if key.startswith(prefix + ".")} != expected_keys:
            raise ValueError(
                "Checkpoint must contain exactly packed weights and fixed scales "
                "for all experts"
            )
        self._global_shapes = {}
        for projection, rows, columns in (
            ("up_proj", intermediate_size, hidden_size),
            ("down_proj", hidden_size, intermediate_size),
        ):
            local = self.num_local
            master = torch.empty(local, rows, columns, dtype=torch.bfloat16, device=device)
            packed = torch.empty(
                local, rows, columns // 2, dtype=torch.uint8, device=device
            )
            scales = torch.empty(
                local, rows, columns // 16, dtype=torch.float8_e4m3fn, device=device
            )
            global_scales = torch.empty(local, dtype=torch.float32, device=device)
            shapes = []
            for expert in range(local):
                name = f"{prefix}.{self.offset + expert}.{projection}"
                checkpoint = load_quantized_weight(root, name, quantized_layers[name])
                if checkpoint.tensors["weight"].shape != (rows, columns // 2):
                    raise ValueError(f"Wrong packed geometry for {name}")
                master[expert].copy_(checkpoint.initial_master())
                packed[expert].copy_(checkpoint.tensors["weight"])
                scales[expert].copy_(checkpoint.tensors["weight_scale"])
                global_scales[expert].copy_(
                    checkpoint.tensors["weight_scale_2"].reshape(())
                )
                shapes.append(checkpoint.tensors["weight_scale_2"].shape)
            setattr(self, projection, torch.nn.Parameter(master))
            self.register_buffer(f"_{projection}_packed", packed)
            self.register_buffer(f"_{projection}_scale", scales)
            self.register_buffer(f"_{projection}_global", global_scales)
            self._global_shapes[projection] = shapes
        self.bind_master()
        self._synced_versions = self._versions()
        self._dirty = False

    def bind_master(self):
        """Track the parameters the optimizer updates (their FSDP2 shards once
        wrapped); during an FSDP2 forward the attributes are the unsharded BF16
        copies."""
        self.__dict__["_masters"] = {p: self._parameters[p] for p in self._geometry}

    def _versions(self):
        tensors = []
        masters = []
        for projection in self._geometry:
            master = self._masters[projection]
            masters.append(master_version(master))
            tensors.append(getattr(master, "_local_tensor", master))
            tensors.extend(
                getattr(self, f"_{projection}_{suffix}")
                for suffix in ("packed", "scale", "global")
            )
        return tuple(masters) + tuple(
            (id(t), t.data_ptr(), t._version, t.dtype, t.device, t.shape, t.stride())
            for t in tensors
        )

    def _validate_storage(self):
        device = self._masters["up_proj"].device
        for projection, (rows, columns) in self._geometry.items():
            master = self._masters[projection]
            # FSDP2 keeps FP32 shards of the BF16 masters.
            master_dtype = torch.float32 if _is_dtensor(master) else torch.bfloat16
            specs = [
                (master, (self.num_local, rows, columns), master_dtype),
                (
                    getattr(self, f"_{projection}_packed"),
                    (self.num_local, rows, columns // 2),
                    torch.uint8,
                ),
                (
                    getattr(self, f"_{projection}_scale"),
                    (self.num_local, rows, columns // 16),
                    torch.float8_e4m3fn,
                ),
                (getattr(self, f"_{projection}_global"), (self.num_local,), torch.float32),
            ]
            if not isinstance(master, torch.nn.Parameter):
                raise RuntimeError("Expert master must remain a Parameter")
            for tensor, shape, dtype in specs:
                if (
                    tensor.dtype != dtype
                    or tensor.shape != shape
                    or tensor.device != device
                    or tensor.is_meta
                    or tensor.layout != torch.strided
                    or not tensor.is_contiguous()
                ):
                    raise RuntimeError(
                        "Expert storage must preserve checkpoint dtype, geometry, "
                        "contiguous layout and common real device"
                    )

    def _checkpoint(self, projection, expert):
        return QuantizedWeight(
            "W4A16_NVFP4",
            {
                "weight": getattr(self, f"_{projection}_packed")[expert],
                "weight_scale": getattr(self, f"_{projection}_scale")[expert],
                "weight_scale_2": getattr(self, f"_{projection}_global")[
                    expert
                ].reshape(self._global_shapes[projection][expert]),
            },
        )

    @torch.no_grad()
    def refresh_quantized(self, recompute_scales=False):
        """Once the masters have been updated, requantize them in place."""
        self._dirty = True
        self._validate_storage()
        self._requantized = getattr(self, "_requantized", False) | recompute_scales
        if self._requantized:
            for projection in ("up_proj", "down_proj"):
                parameter = full_master(self._masters[projection])
                for expert in range(self.num_local):
                    tensors = requantize("W4A16_NVFP4", parameter[expert])
                    getattr(self, f"_{projection}_packed")[expert].copy_(tensors["weight"])
                    getattr(self, f"_{projection}_scale")[expert].copy_(
                        tensors["weight_scale"]
                    )
                    getattr(self, f"_{projection}_global")[expert].copy_(
                        tensors["weight_scale_2"]
                    )
        self._synced_versions = self._versions()
        self._dirty = False

    def export_quantized(self, *, local_names=False):
        """Return independent tensors under HF expert keys.

        ``local_names`` numbers the experts locally, for an exporter that maps
        EP shards to global indices.
        """
        self._validate_storage()
        if self._dirty or self._versions() != self._synced_versions:
            raise RuntimeError("Refresh quantized expert weights after master updates")
        base = 0 if local_names else self.offset
        return {
            f"{self.prefix}.{base + expert}.{projection}.{suffix}": tensor.detach().clone()
            for expert in range(self.num_local)
            for projection in ("up_proj", "down_proj")
            for suffix, tensor in self._checkpoint(projection, expert).tensors.items()
        }


def _te_grouped_gemm(lhs, rhs, out, *, layout, m_splits, single_output=False):
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    outputs = [out] if isinstance(out, torch.Tensor) else list(out)
    general_grouped_gemm(
        list(lhs),
        list(rhs),
        outputs,
        [None] * len(lhs),
        torch.bfloat16,
        single_output=single_output,
        layout=layout,
        m_splits=list(m_splits),
        grad=True,
        use_split_accumulator=True,
    )


def routed_vjp(x, fc1, visible, up, down, routes, ids, dy, activated=None, *, per_route=False):
    """ReLU2 routed experts: ``y = sum_s routes[:, s] * down(relu(up(x))**2)``.

    The route-weight gradient is ``<dy, visible expert output>``; the input and
    expert-weight gradients are TE BF16 grouped GEMMs on the masters.

    Args:
        x: BF16 tokens, ``[M, K]``.
        fc1: Visible FC1 output per route (token-major, slot-minor), ``[M*topk, I]``.
        visible: Visible expert output per route (same order), ``[M*topk, K]``.
        up, down: BF16 masters, ``[E, I, K]`` and ``[E, K, I]``.
        routes: FP32 routing weights, ``[M, topk]``.
        ids: Expert ids, ``[M, topk]``; ``-1`` marks a route held elsewhere
            (zero gradients for it).
        dy: BF16 output gradient, ``[M, K]``.
        activated: Visible activation GEMM2 consumed (same order),
            ``[M*topk, I]``, for the down-weight gradient; None recomputes it
            as ``bf16(relu(fc1)**2)``.
        per_route: Return the input gradient of every route, ``[M, topk, K]``
            (zero for absent routes), instead of their sum.

    Returns:
        ``(dx, d_up, d_down, d_routes)``.
    """
    m, k = x.shape
    topk, experts = ids.shape[1], up.shape[0]
    if m == 0:
        dx = x.new_zeros(0, topk, k) if per_route else torch.zeros_like(x)
        return dx, torch.zeros_like(up), torch.zeros_like(down), routes.new_zeros(0, topk)
    flat = ids.reshape(-1).long()
    held = int((flat >= 0).sum())
    # Absent routes sort first (as -1) and are dropped.
    order = torch.argsort(flat, stable=True)[flat.numel() - held :]
    counts = torch.bincount(flat[flat >= 0], minlength=experts).tolist()
    token = order // topk
    u = fc1.index_select(0, order)
    if activated is None:
        h = u.float().relu().square().to(torch.bfloat16)
    else:
        h = activated.index_select(0, order)
    x_rows = x.index_select(0, token)
    dy_rows = dy.index_select(0, token)
    weight = routes.reshape(-1).index_select(0, order)
    dv = (dy_rows.float() * weight[:, None]).to(torch.bfloat16)

    def split(rows):
        return torch.split(rows, counts)

    d_weight = (dy_rows.float() * visible.index_select(0, order).float()).sum(-1)
    dh = torch.empty_like(h)
    _te_grouped_gemm(down.unbind(0), split(dv), dh, layout="NN", m_splits=counts,
                     single_output=True)
    d_down = torch.zeros_like(down)
    _te_grouped_gemm(split(h), split(dv), d_down.unbind(0), layout="NT",
                     m_splits=counts)
    du = (dh.float() * 2 * u.float().relu()).to(torch.bfloat16)
    dx_rows = torch.empty_like(x_rows)
    _te_grouped_gemm(up.unbind(0), split(du), dx_rows, layout="NN", m_splits=counts,
                     single_output=True)
    d_up = torch.zeros_like(up)
    _te_grouped_gemm(split(x_rows), split(du), d_up.unbind(0), layout="NT",
                     m_splits=counts)
    routes_dx = dx_rows.new_zeros(m * topk, k).index_copy_(0, order, dx_rows).view(m, topk, k)
    d_routes = d_weight.new_zeros(m * topk).index_copy_(0, order, d_weight).view(m, topk)
    dx = routes_dx if per_route else sum_route_grads(routes_dx)
    return dx.to(x.dtype), d_up, d_down, d_routes


def sum_route_grads(per_route):
    """[M, topk, K] -> [M, K] as DS4's deterministic scatter backward: slot
    order, rounded to BF16 after each add."""
    total = per_route[:, 0]
    for slot in range(1, per_route.shape[1]):
        total = (total.float() + per_route[:, slot].float()).to(torch.bfloat16)
    return total


class RoutedExpertsVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down, routes, ids, owner):
        out, fc1, visible, activated = owner._visible(x, ids, routes, return_fc1=True)
        ctx.owner, ctx.versions = owner, owner.weights._versions()
        ctx.save_for_backward(x, fc1, visible, up, down, routes, ids, activated)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        if ctx.owner.weights._versions() != ctx.versions:
            raise RuntimeError("Expert masters changed before backward")
        x, fc1, visible, up, down, routes, ids, activated = ctx.saved_tensors
        dx, d_up, d_down, d_routes = routed_vjp(
            x, fc1, visible, up, down, routes, ids, dy, activated
        )
        return dx, d_up, d_down, d_routes, None, None


def ep4_routed_experts(experts, x, topk_weights, topk_ids, *, return_fc1=False):
    """Run all 128 experts locally with the rollout's EP4 combine of the four
    per-rank ``moe_unpermute`` partials."""
    from megatron.lite.model.nemotron_h.vllm.primitive.moe.communication import reduce_ep4_parts

    result = experts.ep_partials(x, topk_weights, topk_ids, return_fc1=return_fc1)
    if not return_fc1:
        return reduce_ep4_parts(result, topk_ids)
    parts, fc1, per_route, activated = result
    return reduce_ep4_parts(parts, topk_ids), fc1, per_route, activated


class Nvfp4RoutedDeployment(torch.nn.Module):
    """W4A16 routed-expert deployment over BF16 expert masters.

    Takes fixed expert ids and routing weights; excludes routed_scaling_factor
    and the shared expert.
    """

    def __init__(self, weights, model_config, *, ep_group=None):
        super().__init__()
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
        self.ep_group = ep_group
        if (weights.num_local != weights.num_experts) != (ep_group is not None):
            raise ValueError("EP experts need the EP group, and only they")
        if weights.up_proj.device.type != "cuda":
            raise ValueError("Routed deployment requires CUDA checkpoint storage")
        self.weights = weights
        self.config = model_config
        self._ready = False
        self._install()

    def _validate_checkpoint(self):
        self.weights._validate_storage()
        if (
            self.weights._dirty
            or self.weights._versions() != self.weights._synced_versions
        ):
            raise RuntimeError(
                "Refresh deployment after changing expert masters/storage"
            )

    @torch.no_grad()
    def _install(self):
        self._ready = False
        self._validate_checkpoint()
        w = self.weights
        stacks = tuple(
            tuple(getattr(w, f"_{projection}_{s}") for s in ("packed", "scale", "global"))
            for projection in ("up_proj", "down_proj")
        )
        self._experts = CuteDslRoutedExperts(
            *stacks, num_experts=self.config.n_routed_experts, offset=w.offset
        )
        self._deployed_versions = w._versions()
        self._ready = True

    @torch.no_grad()
    def refresh_deployment(self, recompute_scales=False):
        """Call after every optimizer/runtime update, including .data writes."""
        self._ready = False
        self.weights.refresh_quantized(recompute_scales=recompute_scales)
        self._install()

    def forward(self, x, ids, routing_weights):
        active_grad = torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (x, routing_weights, self.weights.up_proj, self.weights.down_proj)
        )
        if self.ep_group is not None:
            from megatron.lite.model.nemotron_h.vllm.primitive.moe.communication import (
                ep_routed_experts,
            )

            self._check_inputs(x, ids, routing_weights)
            return ep_routed_experts(self, x, ids, routing_weights, grad=active_grad)
        if not active_grad:
            return self._visible(x, ids, routing_weights)
        return RoutedExpertsVJP.apply(
            x, self.weights.up_proj, self.weights.down_proj, routing_weights, ids, self
        )

    def _visible(self, x, ids, routing_weights, *, return_fc1=False):
        self._check_inputs(x, ids, routing_weights)
        if x.shape[0] == 0:
            if return_fc1:
                raise ValueError("Routed training requires at least one token")
            return torch.empty_like(x)
        return ep4_routed_experts(
            self._experts, x, routing_weights, ids, return_fc1=return_fc1
        )

    def _check_inputs(self, x, ids, routing_weights):
        if not self._ready or self.weights._versions() != self._deployed_versions:
            raise RuntimeError("Refresh deployment before routed forward")
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
