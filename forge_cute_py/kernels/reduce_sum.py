from concurrent.futures import thread
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUTLASS_CUDA_ARCH'] = '86'

import math
import torch
import time
from cutlass.cute.runtime import from_dlpack

import cutlass
import cutlass.cute as cute

from cutlass import dsl_user_op
from cutlass.cute.arch import nvvm
from cutlass._mlir.dialects.nvvm import AtomicOpKind, MemOrderKind, MemScopeKind
from cutlass.base_dsl.typing import T

_reduce_sum_last_cache = {}
_reduce_sum_first_cache = {}


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
def og_reduce_sum_kernel_last(input: cute.Tensor, output: cute.Tensor, num_warps: int):
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
def reduce_sum_kernel_last(input: cute.Tensor, output: cute.Tensor, num_warps: int):
    ROWS_PER_BLOCK = 4
    WARPS_PER_ROW = num_warps // ROWS_PER_BLOCK
    THREADS_PER_ROW = WARPS_PER_ROW * 32
    
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((ROWS_PER_BLOCK, 32))
    shmem = smem_alloc.allocate_tensor(cute.Float32, smem_layout)
    
    M, N = input.shape
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    block_row = warp_idx // WARPS_PER_ROW
    warp_in_row = warp_idx % WARPS_PER_ROW
    tid_in_row = tidx % THREADS_PER_ROW

    og_row = bidx * ROWS_PER_BLOCK
    row = og_row + block_row

    max_iters = cute.ceil_div(N, THREADS_PER_ROW)
    
    acc = cute.Float32(0)
    for i in range(max_iters):
        col = tid_in_row + i * THREADS_PER_ROW
        if col < N and row < M:
            acc = acc + input[row, col]

    acc = cute.arch.warp_reduction_sum(acc)

    if lane_idx == 0:
        shmem[block_row, warp_in_row] = acc
    
    cute.arch.sync_threads()
    if warp_idx < ROWS_PER_BLOCK:
        v = shmem[warp_idx, lane_idx] if lane_idx < WARPS_PER_ROW else 0.0
        v = cute.arch.warp_reduction_sum(v)
        if lane_idx == 0:
            out_row = og_row + warp_idx
            if out_row < M:
                output[out_row] = v


@cute.jit
def _og_reduce_sum_last(x, output):
    num_warps = 4
    threads_per_block = 32 * num_warps
    m, _ = x.shape
    og_reduce_sum_kernel_last(x, output, num_warps
    ).launch( grid=(m, 1, 1), block=(threads_per_block, 1, 1))


@cute.jit
def _reduce_sum_last(x, output):
    # num_warps = 4
    # threads_per_block = 32 * num_warps
    # m, _ = x.shape
    # reduce_sum_kernel_last(x, output, num_warps
    # ).launch( grid=(m, 1, 1), block=(threads_per_block, 1, 1))
    num_warps = 32
    ROWS_PER_BLOCK = 4
    threads_per_block = 32 * num_warps
    m, _ = x.shape
    blocks = cute.ceil_div(m, ROWS_PER_BLOCK)
    reduce_sum_kernel_last(x, output, num_warps
    ).launch( grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))


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
    cache_key = (x.dtype, x.shape)
    if dim == -1:
        output = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
        if cache_key not in _reduce_sum_last_cache:
            print("compiling...")
            _reduce_sum_last_cache[cache_key] = cute.compile(
                _og_reduce_sum_last, from_dlpack(x), from_dlpack(output)
            )
        _reduce_sum_last_cache[cache_key](from_dlpack(x), from_dlpack(output))
    else:
        output = torch.empty((x.size(1),), device=x.device, dtype=x.dtype)
        if cache_key not in _reduce_sum_first_cache:
            print("compiling...")
            _reduce_sum_first_cache[cache_key] = cute.compile(
                _reduce_sum_first, from_dlpack(x), from_dlpack(output)
            )
        _reduce_sum_first_cache[cache_key](from_dlpack(x), from_dlpack(output))

    return output


def test():
    for dim in [-1, 0]:
        M, N = 1100, 1200
        a = torch.randn((M, N), device="cuda", dtype=torch.float32)
        output = reduce_sum(a, dim=dim)
        close = torch.allclose(output, a.sum(dim), rtol=1e-3)
        assert close, f"Error along dimension: {dim}"
    print("tests pass")


def benchmark():
    import time
    M, N = 4096, 4096
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    # Correctness checks
    print("Correctness checks:")
    for dim in [-1, 0]:
        result = reduce_sum(x, dim=dim)
        expected = x.sum(dim=dim)
        is_close = torch.allclose(result, expected, rtol=1e-3, atol=1e-4)
        print(f"  dim={dim:2d}: {'✓ PASS' if is_close else '✗ FAIL'}")
        if not is_close:
            max_diff = (result - expected).abs().max().item()
            print(f"         max diff: {max_diff}")
    
    print("\nBenchmarks:")
    
    # Warmup
    for _ in range(10):
        _ = reduce_sum(x, dim=-1)
        _ = reduce_sum(x, dim=0)
    torch.cuda.synchronize()
    
    # Benchmark dim=-1
    del x
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    start = time.perf_counter()
    for _ in range(100):
        _ = reduce_sum(x, dim=-1)
    torch.cuda.synchronize()
    print(f"  reduce_sum dim=-1: {(time.perf_counter() - start) * 10:.3f} ms")
    
    # Benchmark dim=0
    del x
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    start = time.perf_counter()
    for _ in range(100):
        _ = reduce_sum(x, dim=0)
    torch.cuda.synchronize()
    print(f"  reduce_sum dim=0:  {(time.perf_counter() - start) * 10:.3f} ms")
    
    # Compare to PyTorch
    del x
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    start = time.perf_counter()
    for _ in range(100):
        _ = x.sum(dim=-1)
    torch.cuda.synchronize()
    print(f"  torch.sum dim=-1:  {(time.perf_counter() - start) * 10:.3f} ms")
    
    del x
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    start = time.perf_counter()
    for _ in range(100):
        _ = x.sum(dim=0)
    torch.cuda.synchronize()
    print(f"  torch.sum dim=0:   {(time.perf_counter() - start) * 10:.3f} ms")


'''
sudo systemctl stop dcgm
/usr/local/cuda-12.8/bin/ncu --set full -o reduce_sum_profile uv run python run.py
/usr/local/cuda-12.8/bin/ncu --import reduce_sum_profile.ncu-rep 
'''
def ncu_test():
    x = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)

    # Warmup (compiles the kernel)
    _ = reduce_sum(x, dim=-1)
    torch.cuda.synchronize()

    # Profile this run
    y = reduce_sum(x, dim=-1)
    torch.cuda.synchronize()

benchmark()
# ncu_test()
