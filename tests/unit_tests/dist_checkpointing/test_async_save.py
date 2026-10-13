# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import errno
import io
import os
import sys
from unittest import mock

import pytest
import torch
from torch.distributed.checkpoint import CheckpointException

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
    _write_item_checked,
)
from megatron.core.dist_checkpointing.strategies.nvrx import has_nvrx_async_support
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistSaveShardedStrategy,
    get_async_strategy,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def write_data_os_err_mock_fn(
    transform_list, local_proc_idx, write_bucket, results_queue, count_queue, use_fsync, **kwargs
):
    """Raises an error on worker #2 during storage save"""
    try:
        if Utils.rank == 2 and local_proc_idx == 2:
            raise OSError('worker #2 critical failure')
        output = (local_proc_idx, [])
    except Exception as e:
        output = (local_proc_idx, e)
    results_queue.put(output)
    count_queue.get()
    count_queue.task_done()


class TestAsyncSave:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('persistent', [True, False])
    @pytest.mark.parametrize('abort', [True, False])
    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt, persistent, abort):
        Utils.initialize_model_parallel(2, 4)

        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_async') as async_ckpt_dir,
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_sync') as sync_ckpt_dir,
        ):
            # async
            async_calls = AsyncCallsQueue(persistent)
            async_request = save(
                sharded_state_dict, async_ckpt_dir, async_sharded_save=True, async_strategy="mcore"
            )
            async_calls.schedule_async_request(async_request)

            # sync
            save(sharded_state_dict, sync_ckpt_dir, async_sharded_save=False)

            # finalize async
            async_calls.maybe_finalize_async_calls(blocking=True)

            # load and compare
            loaded_async_state_dict = load(sharded_state_dict, async_ckpt_dir)
            loaded_sync_state_dict = load(sharded_state_dict, sync_ckpt_dir)
            diffs = diff(loaded_async_state_dict, loaded_sync_state_dict)
            assert not any(map(bool, diffs)), diffs
            async_calls.close(abort=abort)

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('async_strategy', ["nvrx", "mcore"])
    def test_get_async_strategy(self, async_strategy):
        strategy, modules = get_async_strategy(async_strategy)

        assert len(modules) > 1
        assert strategy == async_strategy

        _, module = get_async_strategy(async_strategy, module="FileSystemWriterAsync")
        assert type(module) is not dict

    @pytest.mark.parametrize('async_strategy', ["nvrx", "mcore"])
    def test_get_async_strategy_no_nvrx_installed(self, async_strategy):
        with mock.patch.dict(
            'sys.modules', {'nvidia_resiliency_ext.checkpointing.async_ckpt.core': None}
        ):
            from megatron.core.dist_checkpointing.strategies.async_utils import (
                AsyncRequest as MCoreAsyncRequest,
            )

            if async_strategy == "nvrx":
                with pytest.raises(ModuleNotFoundError):
                    strategy, module = get_async_strategy(async_strategy, module="AsyncRequest")
            else:
                strategy, module = get_async_strategy(async_strategy, module="AsyncRequest")

                assert strategy == "mcore"
                assert module == MCoreAsyncRequest

    def test_get_async_strategy_missing_nvrx_cached_metadata_reader(self):
        with mock.patch.dict(
            'sys.modules',
            {
                'nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader': None
            },
        ):
            with pytest.raises(ModuleNotFoundError):
                get_async_strategy("nvrx", module="CachedMetadataFileSystemReader")


_NVRX_SUBMODULES = [
    'nvidia_resiliency_ext.checkpointing.async_ckpt.core',
    'nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader',
    'nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async',
    'nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver',
]


class _FailingLargeWrites(io.BytesIO):
    """Fails successive writes of at least ``min_bytes`` with the errnos in ``errs``."""

    name = "/ckpt/__0_0.distcp"

    def __init__(self, *errs, min_bytes=(1 << 20) + 1):
        super().__init__()
        self.errs, self.min_bytes = list(errs), min_bytes

    def write(self, data):
        if memoryview(data).nbytes >= self.min_bytes and self.errs:
            err = self.errs.pop(0)
            raise OSError(err, os.strerror(err))
        return super().write(data)


def _tensor_write_item(tensor):
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        MetadataIndex,
        TensorProperties,
    )
    from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType

    chunk = ChunkStorageMetadata(offsets=torch.Size([0]), sizes=tensor.size())
    return WriteItem(
        index=MetadataIndex("decoder.weight"),
        type=WriteItemType.TENSOR,
        tensor_data=TensorWriteData(
            chunk=chunk, properties=TensorProperties.create_from_tensor(tensor), size=tensor.size()
        ),
    )


def _write_tensor(stream, tensor):
    from torch.distributed.checkpoint.filesystem import (
        SerializationFormat,
        _StorageWriterTransforms,
    )

    return _write_item_checked(
        _StorageWriterTransforms(),
        stream,
        tensor,
        _tensor_write_item(tensor),
        "__0_0.distcp",
        serialization_format=SerializationFormat.TORCH_SAVE,
    )


class TestWriteItemChecked:
    def test_failed_write_reports_the_original_error(self):
        """torch.save alone would surface only "unexpected pos ..."."""
        stream = _FailingLargeWrites(errno.ENOSPC)
        stream.write(b"x" * 64)
        with pytest.raises(RuntimeError, match="offset=64 item=.*decoder.weight.*No space"):
            _write_tensor(stream, torch.randn(1 << 20))

    def test_efault_is_rewritten_once_through_bounce_buffers(self):
        tensor = torch.randn(1 << 20)
        stream = _FailingLargeWrites(errno.EFAULT)
        stream.write(b"x" * 64)

        result = _write_tensor(stream, tensor)

        info = result.storage_data
        assert info.offset == 64 and info.offset + info.length == len(stream.getvalue())
        restored = torch.load(io.BytesIO(stream.getvalue()[info.offset :]), weights_only=True)
        assert torch.equal(restored, tensor)

    def test_failed_retry_reports_both_errors(self):
        """The bounce write fails too (1 MiB chunks hit ENOSPC)."""
        stream = _FailingLargeWrites(errno.EFAULT, errno.ENOSPC, min_bytes=1 << 20)
        with pytest.raises(
            RuntimeError, match="retry failed: .*No space.*first attempt: .*Bad address"
        ) as info:
            _write_tensor(stream, torch.randn(1 << 20))
        chain, exc = [], info.value
        while exc is not None:
            chain.append(exc)
            exc = exc.__cause__ or exc.__context__
        assert any(isinstance(e, OSError) and e.errno == errno.EFAULT for e in chain)

    def test_efault_retry_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("MCORE_DIST_CKPT_EFAULT_RETRY", "0")
        stream = _FailingLargeWrites(errno.EFAULT)
        with pytest.raises(RuntimeError, match="Bad address"):
            _write_tensor(stream, torch.randn(1 << 20))


class TestHasNvrxAsyncSupport:
    """Tests for has_nvrx_async_support, focusing on the minimum-version assertion."""

    def _fake_modules(self):
        """MagicMock modules that satisfy every symbol and hasattr check in has_nvrx_async_support."""
        return {name: mock.MagicMock() for name in _NVRX_SUBMODULES}

    def test_version_check_passes(self):
        """Returns True when all NVRx symbols are present and version meets the minimum."""
        with (
            mock.patch(
                'megatron.core.dist_checkpointing.strategies.nvrx.import_module',
                side_effect=lambda name: self._fake_modules()[name],
            ),
            mock.patch(
                'megatron.core.dist_checkpointing.strategies.nvrx.is_nvrx_min_version',
                return_value=True,
            ),
        ):
            assert has_nvrx_async_support() is True

    def test_version_check_fails(self):
        """Raises AssertionError when all NVRx symbols are present but version is too old."""
        with (
            mock.patch(
                'megatron.core.dist_checkpointing.strategies.nvrx.import_module',
                side_effect=lambda name: self._fake_modules()[name],
            ),
            mock.patch(
                'megatron.core.dist_checkpointing.strategies.nvrx.is_nvrx_min_version',
                return_value=False,
            ),
        ):
            with pytest.raises(AssertionError, match="Minimum required nvidia-resiliency-ext"):
                has_nvrx_async_support()
