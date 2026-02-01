"""
Reduction kernel using CuTe DSL.

Implements reduction over a specified dimension:
  dim=1/-1 (row reduction):  mO[row] = sum_{j} mX[row, j]
  dim=0 (column reduction):  mO[col] = sum_{i} mX[i, col]

Variants (for row reduction):
- naive: vecsize=1 (no vectorization), 32 threads/row (single warp)
- improved: vecsize=128/dtype.width (vectorized loads), 32 threads/row (single warp)
- shfl: vecsize=128/dtype.width, up to 128 threads/row (multi-warp with shared memory)

Note: Column reduction (dim=0) uses a simpler per-column accumulation strategy
since column elements are strided in memory (not amenable to vectorized loads).
"""

from typing import Literal, Type

import cutlass
import cutlass.cute as cute

ReductionOp = Literal["sum", "amax", "amin", "prod"]
Variant = Literal["naive", "improved", "shfl"]

# Warp size is always 32 on NVIDIA GPUs
WARP_SIZE = 32


class Reduction:
    _NUM_THREADS = 128
    _VEC_LOAD_BITS = 128

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        M: int,
        reduction_dtype: Type[cutlass.Numeric] | None = cutlass.Float32,
        reduction_op: ReductionOp = "sum",
        dim: int = -1,
        variant: Variant = "shfl",
    ):
        self.dtype = dtype
        self.N = int(N)
        self.M = int(M)
        self.reduction_dtype = dtype if reduction_dtype is None else reduction_dtype
        self.reduction_op = reduction_op
        self.dim = dim
        self.variant = variant

        self._validate_config()

    def _validate_config(self) -> None:
        if self.dim not in (-1, 0, 1):
            raise ValueError(f"dim must be -1, 0, or 1. Got: {self.dim}")

        if self.variant not in ("naive", "improved", "shfl"):
            raise ValueError(f"Unknown variant={self.variant}")

        if self.reduction_op != "sum":
            raise NotImplementedError(f"Only support reduction_op=sum, got {self.reduction_op}")

    def _threads_per_row(self) -> int:
        """
        Threads per row based on variant.

        - naive/improved: 32 threads (one warp)
        - shfl: up to 128 threads (4 warps) for large N
        """
        if self.variant == "shfl" and self.N >= 512:
            return min(self._NUM_THREADS, 128)
        return WARP_SIZE

    def _pick_vecsize(self) -> int:
        """
        Number of elements per vector load.

        - naive: vecsize=1
        - improved/shfl: target 128-bit loads, reduced if N not divisible
        """
        if self.variant == "naive":
            return 1

        elems_per_128b = self._VEC_LOAD_BITS // self.dtype.width
        vecsize = max(1, elems_per_128b)

        while vecsize > 1 and (self.N % vecsize) != 0:
            vecsize //= 2

        return vecsize

    def _adjust_n_blocks(self, n_blocks: int) -> int:
        """Adjust n_blocks to avoid power-of-2 values that can cause codegen issues."""
        is_pow2 = n_blocks.bit_count() == 1
        return n_blocks + 1 if n_blocks >= 8 and is_pow2 else n_blocks

    def _get_tiled_copy(self, vecsize: int):
        """
        Build tile shape (tileM, tileN) and tiled copy operator.
        """
        threads_per_row = self._threads_per_row()
        tile_m = self._NUM_THREADS // threads_per_row

        # Cover N in blocks of threads_per_row; tileN becomes multiple of (threads_per_row * vecsize).
        n_vec_elems = (self.N + vecsize - 1) // vecsize
        n_blocks = (n_vec_elems + threads_per_row - 1) // threads_per_row
        n_blocks = self._adjust_n_blocks(n_blocks)

        tile_n = vecsize * n_blocks * threads_per_row
        tiler_mn = (tile_m, tile_n)

        num_copy_bits = vecsize * self.dtype.width
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=num_copy_bits,
        )

        thr_layout = cute.make_ordered_layout(
            (tile_m, threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, vecsize))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        return tiler_mn, tiled_copy, threads_per_row

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream=None):
        # Dispatch to row or column reduction kernel
        if self.dim == 0:
            self._call_col_reduce(mX, mO, stream)
        else:
            self._call_row_reduce(mX, mO, stream)

    def _call_row_reduce(self, mX: cute.Tensor, mO: cute.Tensor, stream=None):
        """Row reduction: out[m] = sum_n x[m, n]"""
        vecsize = self._pick_vecsize()
        tiler_mn, tiled_copy, threads_per_row = self._get_tiled_copy(vecsize=vecsize)

        num_threads = tiled_copy.size
        warps_per_row = threads_per_row // WARP_SIZE

        self.kernel_row_reduce(
            mX,
            mO,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            warps_per_row,
            self.M,
            self.N,
        ).launch(
            grid=[cute.ceil_div(self.M, tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    def _call_col_reduce(self, mX: cute.Tensor, mO: cute.Tensor, stream=None):
        """Column reduction: out[n] = sum_m x[m, n]"""
        num_threads = self._NUM_THREADS

        self.kernel_col_reduce(
            mX,
            mO,
            self.M,
            self.N,
        ).launch(
            grid=[cute.ceil_div(self.N, num_threads), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_row_reduce(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        warps_per_row: cutlass.Constexpr[int],
        M: int,
        N: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        tile_m = tiler_mn[0]
        tile_n = tiler_mn[1]
        num_threads = threads_per_row * tile_m

        # Tile of X: (tileM, tileN)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXrX = cute.make_rmem_tensor_like(tXgX)

        # Predicate tensor for bounds checking
        tXcX = thr_copy.partition_S(cute.make_identity_tensor((tile_m, tile_n)))

        # Two-phase predicated copy:
        # - Use element 0 to construct a "zero" of the right type
        first_val = tXgX[0]
        zero_val = first_val - first_val

        for i in range(cute.size(tXrX)):
            coord = tXcX[i]
            row = coord[0] + (bidx * tile_m)
            col = coord[1]

            if row < M and col < N:
                tXrX[i] = tXgX[i]
            else:
                tXrX[i] = zero_val

        # Register accumulation then warp shuffle reduction
        x = tXrX.load().to(self.reduction_dtype)
        val = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
        val = cute.arch.warp_reduction_sum(val)

        lane_id = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()

        if warps_per_row == 1:
            # One warp per row: lane 0 writes output
            if lane_id == 0:
                global_row = warp_id + (tile_m * bidx)
                if global_row < M:
                    mO[global_row] = val.to(self.dtype)
        else:
            # Multi-warp per row: shared memory for inter-warp reduction
            smem = cutlass.utils.SmemAllocator()
            num_warps = num_threads // cute.arch.WARP_SIZE
            smem_layout = cute.make_layout((num_warps,), stride=(1,))
            partials = smem.allocate_tensor(
                self.reduction_dtype,
                smem_layout,
                byte_alignment=16,
            )

            if lane_id == 0:
                partials[warp_id] = val

            cute.arch.sync_threads()

            # Warp 0..(tile_m-1) each reduce their row's partials using lanes [0..warps_per_row-1]
            if warp_id < tile_m and lane_id < warps_per_row:
                row_warp_base = warp_id * warps_per_row
                partial_val = partials[row_warp_base + lane_id]
                final_sum = cute.arch.warp_reduction_sum(partial_val)

                if lane_id == 0:
                    row_global = warp_id + (tile_m * bidx)
                    if row_global < M:
                        mO[row_global] = final_sum.to(self.dtype)

    @cute.kernel
    def kernel_col_reduce(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        M: int,
        N: int,
    ):
        """
        Column reduction kernel: out[n] = sum_m x[m, n]

        Each thread handles one column, iterating over all rows.
        This avoids the transpose + contiguous copy at the cost of strided memory access.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        num_threads = self._NUM_THREADS
        col = tidx + bidx * num_threads

        if col < N:
            # Initialize accumulator from first element, converted to reduction dtype
            first_val = mX[0, col]
            acc = first_val.to(self.reduction_dtype) - first_val.to(self.reduction_dtype)

            # Accumulate all rows for this column
            for row in range(M):
                val = mX[row, col]
                acc = acc + val.to(self.reduction_dtype)

            mO[col] = acc.to(self.dtype)
