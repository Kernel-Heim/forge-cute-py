from concurrent.futures import thread
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUTLASS_CUDA_ARCH'] = '86'

import math
import torch
from cutlass.cute.runtime import from_dlpack

import cutlass
import cutlass.cute as cute

from cutlass import dsl_user_op
from cutlass.cute.arch import nvvm
from cutlass._mlir.dialects.nvvm import AtomicOpKind, MemOrderKind, MemScopeKind
from cutlass.base_dsl.typing import T


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


@cute.kernel
def reduce_sum_kernel_last(input: cute.Tensor, output: cute.Tensor, num_warps: int):
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((32,))
    shmem = smem_alloc.allocate_tensor(cute.Float32, smem_layout)
    
    _, N = input.shape
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    max_iters = cute.ceil_div(N, bdimx)

    acc = cute.Float32(0)
    for i in range(max_iters):
        idx = tidx + i * bdimx
        if idx < N:
            acc = acc + input[bidx, idx]
    acc = cute.arch.warp_reduction_sum(acc) 
    if lane_idx == 0:
        shmem[warp_idx] = acc
    cute.arch.sync_threads()
    if warp_idx == 0:
        acc = shmem[lane_idx] if lane_idx < num_warps else 0.0
        acc = cute.arch.warp_reduction_sum(acc) 
        if lane_idx == 0:
            output[bidx] = acc


@cute.kernel
def reduce_sum_kernel_first(input: cute.Tensor, output: cute.Tensor, stride: int):
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


@cute.jit
def _reduce_sum_last(x, output):
    num_warps = 4
    threads_per_block = 32 * num_warps
    m, _ = x.shape
    reduce_sum_kernel_last(x, output, num_warps
    ).launch( grid=(m, 1, 1), block=(threads_per_block, 1, 1))


@cute.jit
def _reduce_sum_first(x, output):
    num_warps = 4
    threads_per_block = num_warps * 32
    m, n = x.shape
    yblocks = cute.ceil_div(n, 4)
    reduce_sum_kernel_first(x, output, threads_per_block // 4
    ).launch(
        grid=(yblocks, 1, 1),
        block=(threads_per_block, 1, 1)
    )


def reduce_sum(x, dim=-1):
    shape = list(x.shape)
    shape.pop(dim)
    output = torch.zeros(shape, device=x.device, dtype=x.dtype)
    if dim == -1:
        vadd_compiled = cute.compile(_reduce_sum_last, from_dlpack(x), from_dlpack(output))
        vadd_compiled(from_dlpack(x), from_dlpack(output))
    else:
        vadd_compiled = cute.compile(_reduce_sum_first, from_dlpack(x), from_dlpack(output))
        vadd_compiled(from_dlpack(x), from_dlpack(output))

    return output


def test():
    for dim in [-1, 0]:
        M, N = 1100, 1200
        a = torch.randn((M, N), device="cuda", dtype=torch.float32)
        output = reduce_sum(a, dim=dim)
        close = torch.allclose(output, a.sum(dim), rtol=1e-3)
        assert close, f"Error along dimension: {dim}"
    print("tests pass")