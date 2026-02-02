from concurrent.futures import thread
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUTLASS_CUDA_ARCH'] = '86'
os.environ['CUTE_DSL_ENABLE_TVM_FFI'] = '1'

import math
import torch
import time
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import const_expr
import cutlass

from cutlass import dsl_user_op
from cutlass.cute.arch import nvvm
from cutlass._mlir.dialects.nvvm import AtomicOpKind, MemOrderKind, MemScopeKind
from cutlass.base_dsl.typing import T

_reduce_sum_last_cache = {}
_reduce_sum_first_cache = {}


# old just for future reference
@dsl_user_op
def atomicAddF32(dst_ptr: cute.Pointer, val: cute.Float32, loc=None, ip=None) -> cute.Float32:
    return nvvm.atomicrmw(
        T.f32(),
        AtomicOpKind.FADD,
        dst_ptr.llvm_ptr,
        val.ir_value(loc=loc, ip=ip),
        mem_order=MemOrderKind.RELAXED,
        syncscope=MemScopeKind.SYS,
        loc=loc,
        ip=ip,
    )


class ReduceSumLast:
    """Sum reduction along last dimension using CuTe DSL."""

    def __init__(self, dtype: type):
        self.dtype = dtype
        self.num_warps = 8
        self.threads_per_block = self.num_warps * 32

    @cute.jit
    def __call__(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
        # stream: cuda.CUstream = None,
    ):
        M, N = input.shape
        tiler_mn = (1, self.threads_per_block * 4)
        gX = cute.zipped_divide(input, tiler_mn)

        thr_layout = cute.make_layout((self.threads_per_block,), stride=(1,))
        val_layout = cute.make_layout((4,), stride=(1,))
        _, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        self.kernel(gX, output, tv_layout).launch(
            grid=(M, 1, 1),
            block=(self.threads_per_block, 1, 1),
            # stream=stream,
        )

    @cute.kernel
    def kernel(self, input: cute.Tensor, output: cute.Tensor, tv_layout):
        num_warps = const_expr(self.num_warps)
        
        smem_alloc = cutlass.utils.SmemAllocator()
        shmem = smem_alloc.allocate_tensor(cute.Float32, cute.make_layout((32,)))

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        # op = cute.nvgpu.CopyUniversalOp()
        # atom = cute.make_copy_atom(op, cute.Float32, num_bits_per_copy=128)
        acc = cute.Float32(0.0)

        _, mode1 = input.shape
        ntiles = mode1[1]

        for tile_idx in range(ntiles):
            blk_coord = ((0, None), (bidx, tile_idx))
            cta_tile = input[blk_coord]
            thr_frag = cute.composition(cta_tile, tv_layout)
            thr_src = thr_frag[(tidx, None)]
            # r = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cute.Float32)
            # cute.copy(atom, thr_src, r)
            # acc += r[0] + r[1] + r[2] + r[3]
            acc += thr_src[0] + thr_src[1] + thr_src[2] + thr_src[3]


        acc = cute.arch.warp_reduction_sum(acc)
        if lane == 0:
            shmem[warp] = acc
        cute.arch.sync_threads()

        if warp == 0:
            acc2 = shmem[lane] if lane < num_warps else 0.0
            acc2 = cute.arch.warp_reduction_sum(acc2)
            if lane == 0:
                output[bidx] = acc2


class ReduceSumFirst:
    """Sum reduction along first dimension using CuTe DSL."""

    def __init__(self, dtype: type):
        self.dtype = dtype
        self.num_warps = 4
        self.threads_per_block = self.num_warps * 32

    @cute.jit
    def __call__(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
        # stream: cuda.CUstream = None,
    ):
        M, N = input.shape
        yblocks = cute.ceil_div(N, 4)
        self.kernel(input, output, self.threads_per_block // 4).launch(
            grid=(yblocks, 1, 1),
            block=(self.threads_per_block, 1, 1),
            # stream=stream,
        )

    @cute.kernel
    def kernel(self, input: cute.Tensor, output: cute.Tensor, stride: int):
        smem_alloc = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout((4, 32))
        shmem = smem_alloc.allocate_tensor(cute.Float32, smem_layout)

        M, N = input.shape
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        max_iters = cute.ceil_div(M, stride)
        col_offset = tidx % 4
        row_offset = tidx // 4
        col = 4 * bidx + col_offset
        acc = cute.Float32(0)

        row = row_offset
        for _ in range(max_iters):
            if row < M and col < N:
                acc = acc + input[row, col]
            row = row + 32

        shmem[col_offset, row_offset] = acc
        cute.arch.sync_threads()
        acc = shmem[warp_idx, lane_idx]

        acc = cute.arch.warp_reduction_sum(acc)
        if lane_idx == 0:
            output[bidx * 4 + warp_idx] = acc
