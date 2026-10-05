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
        worker(dist.get_rank(), *args)
        dist.barrier()

    return run
