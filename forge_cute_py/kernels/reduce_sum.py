"""
Row-wise sum reduction kernel using CuTe DSL (WIP).
"""

import operator
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import const_expr


@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric,
) -> cute.Numeric:
    """Block-wide reduction using warp shuffles + shared memory cross-warp step."""
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    num_warps = cute.size(reduction_buffer)

    if lane_idx == 0:
        reduction_buffer[warp_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < num_warps:
        block_reduce_val = reduction_buffer[lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Thread + warp + block reduction for a single row."""
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = cute.arch.warp_reduction(
        val,
        warp_op,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(cute.size(reduction_buffer) > 1):
        val = block_reduce(val, warp_op, reduction_buffer, init_val=init_val)
    return val


class ReduceSumRow:
    """Row-wise sum reduction (one block per row)."""

    def __init__(self, dtype: type, block_size: int = 256):
        self.dtype = dtype
        self.block_size = block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
    ):
        """
        Launch row-wise reduction.

        Args:
            x: Input tensor of shape (M, N)
            out: Output tensor of shape (M,)
        """
        M, _ = x.shape
        block = const_expr(self.block_size)
        self.kernel(x, out).launch(
            grid=[M, 1, 1],
            block=[block, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
    ):
        """Row-wise sum reduction with vectorized loads and block reduction."""
        tidx, _, _ = cute.arch.thread_idx()
        row_idx, _, _ = cute.arch.block_idx()

        threads_per_block = const_expr(self.block_size)
        num_warps = threads_per_block // 32

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_tensor(
            cutlass.Float32, layout=cute.make_layout(num_warps), byte_alignment=16
        )

        row = x[(row_idx, None)]
        n_cols = row.shape[0]
        acc = cutlass.Float32(0.0)

        vec_size = const_expr(128 // x.element_type.width)
        tile_n = threads_per_block * vec_size
        num_tiles = cute.ceil_div(n_cols, tile_n)

        sX = smem.allocate_tensor(
            x.element_type, layout=cute.make_layout(tile_n), byte_alignment=16
        )
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            x.element_type,
            num_bits_per_copy=128,
        )
        thr_layout = cute.make_layout(threads_per_block)
        val_layout = cute.make_layout(vec_size)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)

        for tile in cutlass.range(0, num_tiles, 1):
            gX = cute.local_tile(row, (tile_n,), (tile,))
            tXgX = thr_copy.partition_S(gX)
            tXsX = thr_copy.partition_D(sX)
            tXrX = cute.make_fragment_like(tXgX)
            cute.copy(copy_atom, tXgX, tXsX)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            cute.autovec_copy(tXsX, tXrX)
            x_vec = tXrX.load().to(cutlass.Float32)
            acc += x_vec.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)

        block_val = row_reduce(
            acc,
            cute.ReductionOp.ADD,
            threads_per_row=threads_per_block,
            reduction_buffer=warp_sums,
            init_val=cutlass.Float32(0.0),
        )
        if tidx == 0:
            out[row_idx] = block_val.to(out.element_type)
