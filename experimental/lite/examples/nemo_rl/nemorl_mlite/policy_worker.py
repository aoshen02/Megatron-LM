# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NeMo-RL policy worker backed by Megatron Lite (mlite).

With ``impl="vllm"`` the training forward calls the same vLLM kernels as the
rollout, so the recomputed logprobs equal the vLLM raw logprobs bitwise.
Select it with ``policy.external_cfg.worker_cls_fqn``; the worker reads its
settings from ``policy.external_cfg``.
"""

import os
import random
from typing import Any, Optional

import numpy as np
import ray
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface, LogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker


def _enable_full_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLitePolicyWorkerImpl(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self) -> str:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return f"{self.__class__.__qualname__}[rank={rank}]"

    def __init__(
        self,
        config: PolicyConfig,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        from megatron.lite.runtime import create_runtime
        from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
        from megatron.lite.runtime.contracts.config import ParallelConfig, RuntimeConfig

        if weights_path or optimizer_path:
            raise NotImplementedError("mlite checkpoint resume is not wired yet")
        self.cfg = config
        mcfg = config["external_cfg"]
        if mcfg.get("full_determinism", True):
            _enable_full_determinism(int(mcfg.get("seed", 42)))

        impl_cfg = dict(mcfg.get("impl_cfg") or {})
        impl_cfg["use_thd"] = True
        impl_cfg.setdefault("cross_entropy_fusion", False)
        impl_cfg.setdefault("deterministic", True)
        extra: dict[str, Any] = {}
        if init_optimizer:
            from megatron.lite.runtime.contracts.config import (
                OptimizerConfig as MLiteOptimizerConfig,
            )

            # Keys mirror verl_mlite's optimizer config (adam, lr, weight_decay,
            # adam_beta1/2, clip_grad, offload_fraction, ...).
            extra["optimizer"] = MLiteOptimizerConfig(**mcfg["optimizer"])
        else:
            impl_cfg["optimizer"] = None
        self.mlite_config = MegatronLiteConfig(
            **extra,
            model_name=mcfg.get("model_name", "nemotron_h"),
            impl=mcfg.get("impl", "vllm"),
            hf_path=config["model_name"],
            parallel=ParallelConfig(
                tp=1,
                etp=1,
                ep=mcfg.get("expert_parallel_size", 1),
                pp=mcfg.get("pipeline_parallel_size", 1),
                vpp=1,
                cp=mcfg.get("context_parallel_size", 1),
            ),
            attention_backend_override=mcfg.get("attention_backend_override"),
            router_aux_loss_coef=0.0,
            load_hf_weights=True,
            impl_cfg=impl_cfg,
        )
        self.runtime = create_runtime(
            RuntimeConfig(
                backend="mlite", hf_path=config["model_name"], backend_cfg=self.mlite_config
            )
        )
        self.handle = self.runtime.build_model()
        if self.handle._optimizer is not None and self.handle._lr_scheduler is None:
            # The runtime leaves the scheduler (and the per-group wd_mult) to the
            # caller; use verl_mlite's so the update matches the verl recipe.
            from verl_mlite.engine.mlite_engine import _build_lr_scheduler

            self.handle._lr_scheduler = _build_lr_scheduler(
                self.handle._optimizer, self.mlite_config.optimizer
            )
        model = self.handle._model
        self.model = model[0] if isinstance(model, (list, tuple)) else model
        self.rank = torch.distributed.get_rank()
        ps = self.handle._parallel_state
        self.dp_size = ps.dp_size
        self.dp_group = ps.dp_group

    def _packed_batch(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        """Pack rows; ``loss_mask`` (per input token) only selects MTP targets."""
        from megatron.lite.runtime.contracts import PackedBatch

        lengths = input_lengths.to(torch.int64).cpu()
        rows = [input_ids[i, : int(n)] for i, n in enumerate(lengths)]
        flat = torch.cat(rows).cuda().contiguous()
        if loss_mask is not None:
            loss_mask = (
                torch.cat([loss_mask[i, : int(n)] for i, n in enumerate(lengths)])
                .to(flat.dtype)
                .cuda()
                .contiguous()
            )
        return PackedBatch(
            input_ids=flat,
            labels=flat,
            loss_mask=loss_mask,
            seq_lens=lengths.cuda(),
            routed_experts=None,
            r3_replay_mask=None,
        )

    @torch.no_grad()
    def get_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Logprob of token i at position i (position 0 is 0), right-padded."""
        from megatron.lite.runtime.contracts import LossContext

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        # Micro-batches of at most logprob_max_tokens tokens (verl's
        # log_prob_max_token_len_per_gpu); a longer sequence goes alone.
        max_tokens = int(self.cfg["external_cfg"].get("logprob_max_tokens", 16384))
        groups: list[list[int]] = [[]]
        used = 0
        for i, n in enumerate(input_lengths.tolist()):
            if groups[-1] and used + n > max_tokens:
                groups.append([])
                used = 0
            groups[-1].append(i)
            used += n
        # Every rank runs the same number of micro-batches (FSDP and EP
        # collectives); ranks with fewer pad with a 2-token row, dropped below.
        num_mb = torch.tensor([len(groups)], device="cuda")
        torch.distributed.all_reduce(num_mb, op=torch.distributed.ReduceOp.MAX)
        batches = [self._packed_batch(input_ids[idx], input_lengths[idx]) for idx in groups]
        pad = torch.tensor([2], dtype=input_lengths.dtype)
        batches += [
            self._packed_batch(input_ids[:1, :2], pad) for _ in range(int(num_mb) - len(groups))
        ]
        items = [
            (
                batch,
                LossContext(
                    temperature=1.0,
                    calculate_entropy=False,
                    return_log_probs=True,
                    loss_scale=1.0,
                    source_batch=None,
                ),
            )
            for batch in batches
        ]
        proto = self.handle._extras["protocol"]
        collected: list[torch.Tensor] = []

        def loss_fn(raw_output, runtime_batch, loss_context):
            per_seq = proto.unpack_forward_output(
                self.model, runtime_batch, raw_output["log_probs"]
            )
            collected.extend(t.float() for t in per_seq.unbind(0))
            return torch.zeros((), device="cuda", dtype=torch.float32), {}

        self.runtime.forward_backward(
            self.handle,
            iter(items),
            loss_fn=loss_fn,
            num_microbatches=len(items),
            forward_only=True,
            router_replay=None,
        )
        # mlite position t holds log p(token t+1); shift to NeMo-RL's layout.
        out = torch.zeros(input_ids.shape, dtype=torch.float32)
        for i, lp in enumerate(collected[: input_ids.shape[0]]):
            n = int(input_lengths[i])
            out[i, 1:n] = lp[: n - 1].cpu()
        return BatchedDataDict[LogprobOutputSpec](logprobs=out)

    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        pass

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        pass

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        self._release_training_memory()

    def offload_before_refit(self) -> None:
        self._release_training_memory()

    def offload_after_refit(self) -> None:
        self._release_training_memory()

    def offload_to_cpu(self) -> None:
        self._release_training_memory()

    def _release_training_memory(self) -> None:
        """Give the colocated rollout its memory back (verl_mlite's train-mode exit).

        Optimizer state goes to the host and cached allocator blocks are
        returned, so vLLM can map its sleeping weights and KV cache.
        """
        import gc

        self.runtime.to(self.handle, "cpu", model=False, optimizer=True, grad=False)
        self.runtime.release_export_scratch(self.handle)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(
            f"[mlite rank {self.rank}] released training memory: "
            f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, "
            f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free",
            flush=True,
        )

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: Any,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        check_dim_skip_keys: Any = None,
    ) -> dict[str, Any]:
        """One optimizer step per global batch; NeMo-RL's loss on mlite logprobs."""
        from megatron.lite.runtime.contracts import LossContext

        if eval_mode:
            raise NotImplementedError("mlite eval_mode train is not wired yet")
        gbs = gbs or self.cfg["train_global_batch_size"]
        mbs = mbs or self.cfg["train_micro_batch_size"]
        dp = self.dp_size
        local_gbs = gbs // dp
        if data.size % local_gbs:
            raise ValueError(f"local batch {data.size} is not a multiple of {local_gbs}")
        proto = self.handle._extras["protocol"]
        losses, grad_norms = [], []
        all_mb_metrics: dict[str, list] = {}
        for gb_start in range(0, data.size, local_gbs):
            gb = data.slice(gb_start, gb_start + local_gbs)
            gb.to("cuda")
            # NeMo-RL losses normalize by the valid sequences/tokens of the whole
            # global batch, summed over data-parallel ranks.
            sample_mask = gb["sample_mask"]
            token_mask = gb["token_mask"][:, 1:] * sample_mask.unsqueeze(-1)
            counts = torch.stack([sample_mask.sum().float(), token_mask.sum().float()]).cuda()
            torch.distributed.all_reduce(counts, group=self.dp_group)
            global_valid_seqs, global_valid_toks = counts[0], counts[1]

            mbs_list = [gb.slice(i, min(i + mbs, local_gbs)) for i in range(0, local_gbs, mbs)]
            nmb = len(mbs_list)
            items = []
            for mb in mbs_list:
                ctx = LossContext(
                    temperature=1.0,
                    calculate_entropy=False,
                    return_log_probs=True,
                    loss_scale=1.0,
                    source_batch=mb,
                )
                # The MTP block trains on the sampled response tokens, as in verl.
                response_mask = mb["token_mask"] * mb["sample_mask"].unsqueeze(-1)
                items.append(
                    (self._packed_batch(mb["input_ids"], mb["input_lengths"], response_mask), ctx)
                )

            def runtime_loss_fn(raw_output, runtime_batch, loss_context):
                mb = loss_context.source_batch
                per_seq = proto.unpack_forward_output(
                    self.model, runtime_batch, raw_output["log_probs"]
                ).unbind(0)
                width = mb["input_ids"].shape[1] - 1
                rows = []
                for i, lp in enumerate(per_seq):
                    n = int(mb["input_lengths"][i])
                    rows.append(torch.nn.functional.pad(lp[: n - 1].float(), (0, width - (n - 1))))
                next_token_logprobs = torch.stack(rows)
                # Only the first update sees the rollout policy; later
                # mini-batches run on already-updated weights.
                if "generation_logprobs" in mb and gb_start == 0:
                    # Training-forward vs rollout logprobs on generated tokens:
                    # bitwise equal everywhere is what ppo_kl == 0 means in verl.
                    mask = (mb["token_mask"][:, 1:] * mb["sample_mask"].unsqueeze(-1)).bool()
                    gen = mb["generation_logprobs"][:, 1:].float()
                    cur = next_token_logprobs.detach()
                    bi_metrics = {
                        "bi_tokens": int(mask.sum()),
                        "bi_equal_tokens": int(((cur == gen) & mask).sum()),
                        "bi_max_abs_diff": float(((cur - gen).abs() * mask).max()),
                    }
                    for k, v in bi_metrics.items():
                        all_mb_metrics.setdefault(k, []).append(v)
                loss, metrics = loss_fn(
                    next_token_logprobs=next_token_logprobs,
                    data=mb,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                )
                if raw_output.get("mtp_loss") is not None:
                    metrics = {
                        **metrics,
                        "mtp_loss": float(raw_output["mtp_loss"]),
                        "mtp_top1": float(raw_output["mtp_acc"]),
                    }
                group = self.handle._optimizer.param_groups[0]
                metrics = {
                    **metrics,
                    "lr": group["lr"],
                    "wd": group.get("weight_decay", 0.0),
                    "global_valid_seqs": float(global_valid_seqs),
                    "global_valid_toks": float(global_valid_toks),
                }
                for k, v in metrics.items():
                    all_mb_metrics.setdefault(k, []).append(v)
                # The runtime divides by the micro-batch count and FSDP averages
                # gradients over data-parallel ranks; NeMo-RL's loss is already
                # the global mean, so undo both (same scaling as verl_mlite).
                return loss * nmb * dp, {}

            self.runtime.zero_grad(self.handle)
            self.runtime.forward_backward(
                self.handle,
                iter(items),
                loss_fn=runtime_loss_fn,
                num_microbatches=nmb,
                forward_only=False,
                router_replay=None,
            )
            self.runtime.to(self.handle, "cuda", model=False, optimizer=True, grad=False)
            ok, grad_norm, _ = self.runtime.optimizer_step(self.handle)
            if not ok:
                raise RuntimeError("mlite optimizer step was skipped")
            self.runtime.lr_scheduler_step(self.handle)
            # Offloaded optimizer: load, step, offload (verl_mlite), so the next
            # mini-batch's forward/backward has the memory back.
            self.runtime.to(self.handle, "cpu", model=False, optimizer=True, grad=False)
            grad_norms.append(grad_norm)
            losses.append(sum(float(v) for v in all_mb_metrics.get("loss", [])[-nmb:]))
        if "bi_tokens" in all_mb_metrics:
            print(
                f"[mlite rank {self.rank}] first-update train logprobs vs rollout: "
                f"{sum(all_mb_metrics['bi_equal_tokens'])}/"
                f"{sum(all_mb_metrics['bi_tokens'])} bitwise equal, max abs diff "
                f"{max(all_mb_metrics['bi_max_abs_diff']):.3g}",
                flush=True,
            )
        if "mtp_loss" in all_mb_metrics:
            n = len(all_mb_metrics["mtp_loss"])
            print(
                f"[mlite rank {self.rank}] mtp loss "
                f"{sum(all_mb_metrics['mtp_loss']) / n:.4f}, top1 "
                f"{sum(all_mb_metrics['mtp_top1']) / n:.4f}",
                flush=True,
            )
        else:
            print(f"[mlite rank {self.rank}] mtp loss: none reported", flush=True)
        return {
            "global_loss": torch.tensor(losses),
            "grad_norm": torch.tensor(grad_norms),
            "all_mb_metrics": all_mb_metrics,
        }

    def get_reference_policy_logprobs(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("mlite has no reference model")

    def get_topk_logits(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def calibrate_qkv_fp8_scales(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def _export_weights(self):
        # Every rank yields the full model (rank0_only=False), so each colocated
        # TP1 vLLM engine receives all tensors. Nemotron exports the deployment
        # bytes (NVFP4/FP8 weights, FP32 scales, k/v scales) as stored.
        return self.runtime.export_weights(
            self.handle, buffer_max_size_bytes=2 * 1024**3, cpu=False
        )

    @torch.no_grad()
    def prepare_refit_info(self, *, refit_payload_mode: Any = "hf_export"):
        del refit_payload_mode
        return {name: (tensor.shape, tensor.dtype) for name, tensor in self._export_weights()}

    @torch.no_grad()
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict] = None
    ) -> None:
        if kv_scales is not None:
            raise ValueError(
                "mlite streams static k/v scales with the weights; set "
                "generation.vllm_cfg.kv_scales_in_refit_weights=true"
            )
        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        self.maybe_init_zmq()
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._export_weights(),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )
        self.runtime.release_export_scratch(self.handle)

    def broadcast_weights_for_collective(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("mlite refits colocated vLLM over IPC only")


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("mlite_policy_worker")
)  # pragma: no cover
class MLitePolicyWorker(MLitePolicyWorkerImpl):
    pass


# Megatron Lite is installed in the training image itself, not in a uv extra.
ACTOR_ENVIRONMENT_REGISTRY.setdefault(f"{__name__}.MLitePolicyWorker", PY_EXECUTABLES.SYSTEM)
