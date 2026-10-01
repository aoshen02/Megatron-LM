"""Diagnostic SM100 grouped TF32 adapter, dynamic-dummy revision r2.

FP32 storage and TMA format; TFloat32 MMA interpretation, FP32 accumulation/output. New
execution revision: no promise of V2 reduction-order or byte equivalence.
"""

import hashlib
from pathlib import Path

SOURCE_SHA = "8b81ce2f2a197279c6159a8fd246e60b19ce11fded14f9bbe9d133e5ebc8faa5"
REVISION = "compact-grouped-tf32-r4-f32-tma-base"


class GroupedTF32:
    """Reuse the pinned F32-TMA CUTLASS artifact, bypassing only its demo run().

    Calls are synchronous in Python launch order, asynchronous on current CUDA
    stream. Inputs must stay alive until completion. Not thread/Graph-safe yet.
    Metadata contains logical (M,N,K,1), element strides [E,3,2], and element
    offsets [E,3] into three contiguous FP32 backing allocations. Empty groups
    must be represented by >=4 zero rows, not zero-size TMA descriptors.
    """

    def __init__(self, source=None):
        import cutlass
        from cutlass import utils
        from . import grouped_gemm_tma_f32 as module

        installed_source = Path(module.__file__).resolve()
        if source is not None and Path(source).resolve() != installed_source:
            raise ValueError("External CUTLASS grouped source is not supported")
        if hashlib.sha256(installed_source.read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError("Unreviewed CUTLASS grouped source")
        self.kernel_class = module.GroupedGemmKernel
        self.source = str(installed_source)
        self.cutlass = cutlass
        self.hardware = utils.HardwareInfo()
        self.cache = {}
        self.live = []

    def __call__(self, a, b, out, shapes, strides, offsets, *, major):
        import torch
        from cuda.bindings import driver
        from cutlass import cute
        from cutlass.cute.runtime import from_dlpack

        values = (a, b, out)
        if any(
            t.dtype != torch.float32 or not t.is_cuda or not t.is_contiguous()
            for t in values
        ):
            raise ValueError("Contiguous CUDA FP32 backing tensors required")
        if any(t.device != a.device for t in (*values, shapes, strides, offsets)):
            raise ValueError("Mixed devices")
        e = shapes.shape[0]
        if (
            shapes.shape != (e, 4)
            or strides.shape != (e, 3, 2)
            or offsets.shape != (e, 3)
            or shapes.dtype != torch.int32
            or strides.dtype != torch.int32
            or offsets.dtype != torch.int64
            or not all(t.is_contiguous() for t in (shapes, strides, offsets))
            or major not in (("k", "k"), ("k", "mn"), ("mn", "mn"))
        ):
            raise ValueError("Bad grouped metadata contract")
        # Metadata builders below own allocation bounds/alignment; no host route
        # counts, dtype cast, or tensor-copy fallback is hidden in this adapter.
        pointers = offsets * 4 + torch.tensor(
            [t.data_ptr() for t in values], dtype=torch.int64, device=a.device
        )
        stream = driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        key = (e, major, a.device.index)
        if key not in self.cache:
            samples = []
            for layout in (*major, "k"):
                sample = torch.empty((32, 32), device=a.device, dtype=torch.float32)
                samples.append((sample.T if layout == "mn" else sample).unsqueeze(-1))
            initial = [
                from_dlpack(t, assumed_align=16, force_tf32=i < 2).mark_layout_dynamic(
                    leading_dim=0 if i < 2 and major[i] == "mn" else 1
                )
                for i, t in enumerate(samples)
            ]
            active = self.hardware.get_max_active_clusters(1)
            maps = torch.empty(
                (
                    active,
                    self.kernel_class.num_tensormaps,
                    self.kernel_class.bytes_per_tensormap // 8,
                ),
                dtype=torch.int64,
                device=a.device,
            )
            kernel = self.kernel_class(self.cutlass.Float32, False, (128, 128), (1, 1))
            compiled = cute.compile(
                kernel,
                *initial,
                e,
                from_dlpack(shapes, assumed_align=16),
                from_dlpack(strides, assumed_align=16),
                from_dlpack(pointers, assumed_align=16),
                active,
                from_dlpack(maps, assumed_align=16),
                active,
                stream,
            )
            self.cache[key] = (compiled, initial, maps, active, samples)
        compiled, initial, maps, active, samples = self.cache[key]
        compiled(
            *initial,
            from_dlpack(shapes, assumed_align=16),
            from_dlpack(strides, assumed_align=16),
            from_dlpack(pointers, assumed_align=16),
            from_dlpack(maps, assumed_align=16),
            stream,
        )
        # Single diagnostic VJP owns this adapter. Explicit release after sync;
        # don't grow this across unbounded training iterations.
        self.live.append((shapes, strides, pointers, offsets))
        return out

    def release_metadata(self):
        """Caller must synchronize the launch stream before releasing metadata."""
        self.live.clear()


def row_gemm(adapter, rows, weights, counts, starts, *, transpose_weight=False):
    """Per expert rows[n,K] @ weights[N,K].T (or weights[K,N])."""
    import torch

    e, w0, w1 = weights.shape
    n, k = (w1, w0) if transpose_weight else (w0, w1)
    if rows.ndim != 2 or rows.shape[1] != k or k % 4 or n % 4:
        raise ValueError("Expected FP32 16-byte-aligned geometry")
    out = rows.new_zeros((rows.shape[0], n))
    shapes = torch.stack(
        (counts, counts * 0 + n, counts * 0 + k, counts * 0 + 1), dim=1
    ).int()
    stride = ((k, 1), (1, n) if transpose_weight else (k, 1), (n, 1))
    strides = torch.tensor(stride, dtype=torch.int32, device=rows.device)[None].repeat(
        e, 1, 1
    )
    offsets = torch.stack(
        (
            starts * k,
            torch.arange(e, device=rows.device, dtype=torch.int64) * w0 * w1,
            starts * n,
        ),
        1,
    )
    return adapter(
        rows,
        weights,
        out,
        shapes,
        strides,
        offsets,
        major=("k", "mn" if transpose_weight else "k"),
    )


def weight_gemm(adapter, left, right, counts, starts):
    """Per expert left[n,M].T @ right[n,N]; FP32 [E,M,N] result."""
    import torch

    e, m, n = counts.numel(), left.shape[1], right.shape[1]
    if left.shape[0] != right.shape[0] or m % 4 or n % 4:
        raise ValueError("Bad compact dW dimensions")
    out = left.new_zeros((e, m, n))
    shapes = torch.stack(
        (counts * 0 + m, counts * 0 + n, counts, counts * 0 + 1), dim=1
    ).int()
    strides = torch.tensor(
        ((1, m), (1, n), (n, 1)), dtype=torch.int32, device=left.device
    )[None].repeat(e, 1, 1)
    offsets = torch.stack(
        (
            starts * m,
            starts * n,
            torch.arange(e, device=left.device, dtype=torch.int64) * m * n,
        ),
        1,
    )
    return adapter(left, right, out, shapes, strides, offsets, major=("mn", "mn"))
