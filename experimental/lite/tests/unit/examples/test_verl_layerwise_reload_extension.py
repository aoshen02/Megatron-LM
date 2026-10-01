# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

pytest.importorskip("vllm.model_executor.model_loader.reload")
pytest.importorskip("verl.workers.rollout.vllm_rollout.utils")

from vllm.model_executor.model_loader.reload import (  # noqa: E402
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import (  # noqa: E402
    default_weight_loader,
)

from verl_mlite.rollout.layerwise_reload import (  # noqa: E402
    load_checkpoint_bucket,
    require_layerwise_reload_support,
)

pytestmark = pytest.mark.optional


class _Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.zeros(4), requires_grad=False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _Projection()

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, tensor in weights:
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, tensor)


def test_layer_split_across_reused_buckets_reloads_in_place() -> None:
    """A layer completed by a later bucket must not read the reused buffer."""
    model = _Model()
    record_metadata_for_reloading(model)
    storages = {n: p.data_ptr() for n, p in model.named_parameters()}

    for step in (1.0, 2.0):
        buffer = torch.full((4,), step)
        initialize_layerwise_reload(model)
        load_checkpoint_bucket(model, [("proj.weight", buffer)])
        buffer.fill_(-1.0)
        load_checkpoint_bucket(model, [("proj.weight_scale", buffer)])
        finalize_layerwise_reload(model, None)

        assert torch.equal(model.proj.weight, torch.full((4,), step))
        assert torch.equal(model.proj.weight_scale, torch.full((4,), -1.0))
        assert {n: p.data_ptr() for n, p in model.named_parameters()} == storages


def test_unverified_quant_config_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="MIXED_PRECISION"):
        require_layerwise_reload_support(SimpleNamespace(quant_config=None))
