"""
Reudction kerne using CuTe DSL
"""

from typing import Literal, Type

import cutlass
import cutlass.cute as cute


class Reduction:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        reduction_dtype: Type[cutlass.Numeric] | None = None,
        reduction_op: Literal["sum", "amax", "amin", "prod"] = "sum",
        dim: int = -1,
    ):
        self.dtype = dtype
        self.N = N
        self.reduction_dtype = reduction_dtype if reduction_dtype is not None else dtype
        self.reduction_op = reduction_op
        self.dim = dim

        if self.dim not in (-1, 0, 1):
            raise ValueError
        if self.reduction_op not in ["sum", "amax", "amin", "prod"]:
            raise ValueError

        if self.dim not in [-1, 1]:
            raise NotImplementedError(f"Only support dim=1 or -1, got {self.dim}")
        if self.reduction_op != "sum":
            raise NotImplementedError(f"Only support reduction_op=sum, got {self.reduction_op}")

    def _get_tiled_copy(self, vecsize: int = 1):
        threads_per_row = 32
        num_threads = 128
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)

        num_copy_bits = vecsize * self.dtype.width
        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, self.dtype, num_bits_per_copy=num_copy_bits)
        thr_layout = cute.make_ordered_layout(
            (num_threads // threads_per_row, threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, vecsize))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        return tiler_mn, tiled_copy, threads_per_row

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream=None):
        # if self.dtype != mX.element_type:
        #     raise ValueError("dtype mismatch")

        vecsize = 128 // self.dtype.width
        tiler_mn, tiled_copy, threads_per_row = self._get_tiled_copy(vecsize=vecsize)

        # print(f"[DSL INFO] mX: {mX}")
        # print(f"[DSL INFO] mO: {mO}")

        num_threads = tiled_copy.size

        self.kernel(mX, mO, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        # tv_layout = (thread_layout, value_layout) = ((threads_per_row, num_rows), vec_size)
        # print(tiled_copy.layout_tv_tiled)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # smem = cutlass.utils.SmemAllocator()
        # sX = smem.allocate_tensor(self.dtype, cute.make_ordered_layout(tiler_mn, order=(1, 0)))

        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))  # (tileM, tileN)
        # TODO: vectorize store
        # gO = cute.local_tile(mO, cute.select(tiler_mn, mode=[0]), (bidx,))  # (tileM,)
        # print("gX.shape: ", gX.shape)
        # print("gO.shape: ", gO.shape)
        # print("tiler_mn: ", tiler_mn)

        thr_copy_X = tiled_copy.get_slice(tidx)
        print(thr_copy_X)
        # gmem -> rmem
        tXgX = thr_copy_X.partition_S(gX)
        tXrX = cute.make_rmem_tensor_like(tXgX)
        cute.autovec_copy(tXgX, tXrX)
        # print("tXrX: ", tXrX)

        x = tXrX.load().to(self.reduction_dtype)
        val = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)

        val = cute.arch.warp_reduction_sum(val)

        lane_id = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()

        warps_per_row = threads_per_row // cute.arch.WARP_SIZE

        row_idx = warp_id // warps_per_row
        col_idx = warp_id % warps_per_row

        if lane_id == 0 and col_idx == 0:
            mO[row_idx + tiler_mn[0] * bidx] = val.to(self.dtype)
