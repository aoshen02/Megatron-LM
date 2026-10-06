import os

import pytest
import torch

# The EP4 group starts from the NCCL environment at collection; vLLM's
# init_batch_invariance exports more.
_COLLECTION_NCCL_ENV = {k: v for k, v in os.environ.items() if "NCCL" in k}


@pytest.fixture(scope="module")
def _ep4_group():
    """One NCCL group over the four torchrun ranks run_tests.sh launches for
    ``gpus(4)``, shared by the module's tests: a second default group in the
    same torchrun process would reuse the first one's NCCL unique id."""
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires torchrun --standalone --nproc-per-node=4")
    with pytest.MonkeyPatch.context() as patch:
        for name in [k for k in os.environ if "NCCL" in k]:
            patch.delenv(name)
        for name, value in _COLLECTION_NCCL_ENV.items():
            patch.setenv(name, value)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        try:
            yield
        finally:
            dist.destroy_process_group()


@pytest.fixture
def ep4(_ep4_group):
    """Run ``worker(rank, *args)`` on this rank of the EP4 group."""
    import torch.distributed as dist

    def run(worker, *args):
        # Exchange outcomes so a failure on one rank fails every rank at once
        # instead of leaving the others in a barrier until the NCCL timeout.
        rank, error = dist.get_rank(), None
        try:
            worker(rank, *args)
        except BaseException as exc:  # pytest skips and fails too
            error = exc
        message = None if error is None else f"{type(error).__name__}: {error}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, message)
        if error is not None:
            raise error
        failed = {r: e for r, e in enumerate(errors) if e is not None}
        if failed:
            pytest.fail(f"failed on other ranks: {failed}")

    return run
