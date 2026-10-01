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
    LayerwiseReloadSession,
    require_layerwise_reload_support,
)

pytestmark = pytest.mark.optional


def _half_loader(param, loaded_weight, half):
    param.data[half * 2 : half * 2 + 2].copy_(loaded_weight)


class _Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        self.weight.weight_loader = _half_loader
        self.weight_scale = torch.nn.Parameter(torch.zeros(4), requires_grad=False)


class _Experts(torch.nn.Module):
    """Has a tensor the checkpoint never ships, like W4A16 expert input scales."""

    def __init__(self):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        self.w13_input_scale = torch.nn.Parameter(torch.zeros(2), requires_grad=False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _Projection()
        self.other = _Projection()
        self.experts = _Experts()

    def load_weights(self, weights):
        """Like vLLM expert loaders, silently skip names it does not know."""
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            target, _, half = name.partition(":")
            param = params.get(target)
            if param is None:
                continue
            if half:
                param.weight_loader(param, tensor, int(half))
            else:
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
            loaded.add(target)
        return loaded


def _full_payload(value):
    payload = []
    for layer in ("proj", "other"):
        payload += [
            (f"{layer}.weight:0", torch.full((2,), value)),
            (f"{layer}.weight:1", torch.full((2,), value)),
            (f"{layer}.weight_scale", torch.full((4,), value)),
        ]
    return payload + [("experts.w13_weight", torch.full((4,), value))]


# The production table keys on vLLM's W4A16 NVFP4 quant methods; this model
# has none, so it exempts the tensor by class name.
ALLOWED = {("_Experts", "w13_input_scale"): "never in the checkpoint"}


def _reload(model, payload, *, bucket=1):
    """Start a reload and stream the payload through one reused buffer."""
    initialize_layerwise_reload(model)
    session = LayerwiseReloadSession(model)
    for start in range(0, len(payload), bucket):
        chunk = payload[start : start + bucket]
        session.load(chunk)
        for _, tensor in chunk:
            tensor.fill_(-1.0)  # the receiver reuses its buffer
    return session


@pytest.fixture
def model():
    model = _Model()
    record_metadata_for_reloading(model)
    return model


def test_complete_updates_reload_in_place_twice(model) -> None:
    """Two full updates through a reused buffer land in the original storage;
    the allowlisted, never-shipped expert input scale is not a deficit."""
    storages = {n: p.data_ptr() for n, p in model.named_parameters()}
    for step in (1.0, 2.0):
        session = _reload(model, _full_payload(step))
        assert session.deficits(ALLOWED) == []
        finalize_layerwise_reload(model, None)
        for name in ("proj.weight", "other.weight_scale", "experts.w13_weight"):
            assert torch.equal(model.get_parameter(name), torch.full((4,), step))
        assert {n: p.data_ptr() for n, p in model.named_parameters()} == storages


@pytest.mark.parametrize(
    "dropped, layer, missing",
    [
        ("other.weight_scale", "other", ["weight_scale"]),  # missing tensor
        ("proj.weight:1", "proj", []),  # half a tensor: only numel shows it
        ("experts.w13_weight", "experts", ["w13_weight"]),
    ],
)
def test_incomplete_layer_is_reported(model, dropped, layer, missing) -> None:
    payload = [(n, t) for n, t in _full_payload(1.0) if n != dropped]
    (deficit,) = _reload(model, payload).deficits(ALLOWED)
    assert deficit["layer"] == layer and deficit["missing"] == missing
    assert deficit["load_numel"] < deficit["expected_numel"]


def test_absent_layer_is_reported(model) -> None:
    payload = [(n, t) for n, t in _full_payload(1.0) if not n.startswith("other.")]
    (deficit,) = _reload(model, payload).deficits(ALLOWED)
    assert deficit["layer"] == "other" and deficit["load_numel"] == 0


def test_production_table_does_not_exempt_unrelated_layers(model) -> None:
    """Exemptions are tied to W4A16 NVFP4 quant methods, not to names alone."""
    (deficit,) = _reload(model, _full_payload(1.0)).deficits()
    assert deficit["layer"] == "experts"
    assert deficit["missing"] == ["w13_input_scale"]


def test_duplicate_name_is_rejected(model) -> None:
    payload = _full_payload(1.0)
    with pytest.raises(ValueError, match="Duplicate tensor"):
        _reload(model, payload + payload[:1])


def test_name_the_model_ignores_is_rejected(model) -> None:
    with pytest.raises(ValueError, match="loaded nothing"):
        _reload(model, _full_payload(1.0) + [("proj.bogus", torch.zeros(1))])


def test_unverified_quant_config_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="MIXED_PRECISION"):
        require_layerwise_reload_support(SimpleNamespace(quant_config=None))
