"""Trainer-process vLLM state owned by the quantized Nemotron protocol.

Quantized projections and routed experts are vLLM layers. They need a current
VllmConfig (Humming MoE backend, CustomOp selection), vLLM world/TP/EP/PP
groups and a workspace manager. The protocol creates them once per process so
generic mlite runtimes and engines can drive the model unchanged.
"""

from contextlib import nullcontext

import torch
import torch.distributed as dist

_owned_config = None


def ensure_vllm_runtime(pipeline_size):
    """Return the process vLLM config, initializing groups/workspace once.

    A config that is already current (caller-owned, e.g. a test)
    is reused as-is; caller_runtime() validates either source.
    """
    global _owned_config
    from vllm.config import (
        CompilationConfig,
        ParallelConfig,
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.v1.worker.workspace import init_workspace_manager, is_workspace_manager_initialized

    current = get_current_vllm_config_or_none()
    if current is not None:
        return current
    if _owned_config is not None:
        if _owned_config.parallel_config.pipeline_parallel_size != pipeline_size:
            raise RuntimeError("vLLM runtime already initialized for another PP size")
        return _owned_config
    if not dist.is_initialized():
        raise RuntimeError("Initialize torch.distributed before the vLLM runtime")
    # The trainer never builds a vLLM executor or model runner; its pipeline is
    # mlite's. Do not declare external_launcher: with PP > 1 that requests the
    # runner's PP output broadcast, which Model Runner V2 rejects.
    config = VllmConfig(
        parallel_config=ParallelConfig(
            pipeline_parallel_size=pipeline_size, distributed_executor_backend="mp"
        ),
        compilation_config=CompilationConfig(custom_ops=["none", "+quant_fp8"]),
    )
    config.kernel_config.moe_backend = "humming"
    device = torch.device("cuda", torch.cuda.current_device())
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=dist.get_world_size(), rank=dist.get_rank(), local_rank=device.index
        )
        ensure_model_parallel_initialized(1, pipeline_size)
        if not is_workspace_manager_initialized():
            init_workspace_manager(device)
    _owned_config = config
    return config


def _base(module):
    while hasattr(module, "module"):
        module = module.module
    return module


def vllm_context(module):
    """Make the model's vLLM config current; no-op for unquantized models."""
    config = getattr(_base(module), "_vllm_config", None)
    if config is None:
        return nullcontext()
    from vllm.config import set_current_vllm_config

    return set_current_vllm_config(config)
