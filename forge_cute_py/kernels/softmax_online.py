import torch
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack


class SoftmaxOnlineBackward:
    def __init__(self, dtype, N: int):
        self.dtype = dtype
        self.num_warps = 4
        self.bits_read = 128
        self.vec_load_size = self.bits_read // dtype.width
        self.warp_size = 32
        self.threads_per_block = self.num_warps * self.warp_size
        self.N = N  # N is static at compile time, M is dynamic

    @cute.jit
    def __call__(self, dY: cute.Tensor, y: cute.Tensor, dx: cute.Tensor, stream=None):
        blocks_over_N = cute.ceil_div(self.N, self.vec_load_size * self.warp_size)
        tiler_mn = (  # full covering tile
            self.num_warps,
            self.vec_load_size * self.warp_size * blocks_over_N,
        )

        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, self.dtype, num_bits_per_copy=self.bits_read)

        thr_layout = cute.make_ordered_layout(
            (self.num_warps, self.warp_size),
            order=(1, 0),  # cols move faster
        )
        val_layout = cute.make_layout((1, self.vec_load_size))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        blocks = cute.ceil_div(dY.shape[0], self.num_warps)
        self.kernel(dY, y, dx, tiler_mn, tiled_copy).launch(
            grid=(blocks, 1, 1), block=(self.threads_per_block, 1, 1), stream=stream
        )

    # type hints are not optional!!!!
    @cute.kernel
    def kernel(
        self,
        dY: cute.Tensor,
        y: cute.Tensor,
        dX: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        # Compute gradient (numerically stable)
        # dot_product = (dy * y).sum(dim=dim, keepdim=True)
        # result = y * (dy - dot_product)
        # dx.copy_(result)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        dy_tile = cute.local_tile(dY, tiler_mn, (bidx, 0))
        y_tile = cute.local_tile(y, tiler_mn, (bidx, 0))
        dx_tile = cute.local_tile(dX, tiler_mn, (bidx, 0))

        tidxSlice = tiled_copy.get_slice(tidx)

        dy_idx = tidxSlice.partition_S(dy_tile)
        y_idx = tidxSlice.partition_S(y_tile)
        dx_idx = tidxSlice.partition_D(dx_tile)

        dy_regs = cute.make_rmem_tensor_like(dy_idx)
        y_regs = cute.make_rmem_tensor_like(y_idx)

        cute.autovec_copy(dy_idx, dy_regs)
        cute.autovec_copy(y_idx, y_regs)

        dy_data = dy_regs.load()
        y_data = y_regs.load()

        tidx_local_dp = dy_data * y_data
        tidx_local_sum = tidx_local_dp.reduce(
            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
        )
        row_dp_sum = cute.arch.warp_reduction_sum(tidx_local_sum)

        result = y_data * (dy_data - row_dp_sum)
        dy_regs.store(result)
        cute.autovec_copy(dy_regs, dx_idx)


class SoftmaxOnlineLoop:
    def __init__(self, dtype):
        self.dtype = dtype
        self.num_warps = 1
        self.threads_per_block = self.num_warps * 32
        self.NEG_INF = Float32(float("-inf"))

    @cute.jit
    def __call__(self, gInput: cute.Tensor, gOutput: cute.Tensor, stream: cuda.CUstream = None):
        M, N = gInput.shape
        thr_layout = cute.make_layout((self.threads_per_block,), stride=(1,))
        val_layout = cute.make_layout((1,), stride=(1,))
        tiler_mn_1d, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
        tiler_mn = (1, tiler_mn_1d[0])
        gX = cute.zipped_divide(gInput, tiler_mn)
        gY = cute.zipped_divide(gOutput, tiler_mn)

        self.kernel(gX, gY, N).launch(
            grid=(cute.size(gX, mode=[1, 0]), 1, 1),  # RestM
            block=(cute.size(tv_layout, mode=[0, 0])),  # threads per block
            stream=stream,
        )

    @cute.kernel
    def kernel(self, gInput, gOutput, N):
        maxValue = self.NEG_INF
        sumValue = Float32(0.0)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdimx, _, _ = cute.arch.block_dim()
        (TileM, TileN), (RestM, RestN) = gInput.shape

        for i in range(RestN):
            idx = i * bdimx + tidx
            value = Float32(gInput[(0, tidx), (bidx, i)]) if idx < N else self.NEG_INF

            curMax = cute.arch.warp_reduction_max(value)
            prevMax = maxValue
            maxValue = cute.arch.fmax(prevMax, curMax)

            scale = cute.math.exp(prevMax - maxValue)
            scale_data = cute.math.exp(value - maxValue)
            curSum = cute.arch.warp_reduction_sum(scale_data)
            sumValue = sumValue * scale + curSum

        for i in range(RestN):
            idx = i * bdimx + tidx
            if idx < N:
                value = gInput[(0, tidx), (bidx, i)].to(self.dtype)
                data = cute.math.exp(value - maxValue) / sumValue
                gOutput[(0, tidx), (bidx, i)] = data.to(self.dtype)


class SoftmaxOnline:
    def __init__(self, dtype, N: int):
        self.dtype = dtype
        self.num_warps = 1
        self.threads_per_block = self.num_warps * 32
        self.NEG_INF = Float32(float("-inf"))
        self.N = N

        self.bits_read = 128
        self.vec_load_size = self.bits_read // self.dtype.width
        self.threads_per_row = 32
        self.num_warps = 4
        self.num_threads = self.num_warps * self.threads_per_row

    @cute.jit
    def __call__(self, gInput: cute.Tensor, gOutput: cute.Tensor, stream: cuda.CUstream = None):

        blocks_vector_N = cute.ceil_div(self.N, self.bits_read // self.dtype.width)
        blocks_over_N = cute.ceil_div(blocks_vector_N, self.threads_per_row)
        tiler_mn = (
            self.num_warps,
            self.vec_load_size * blocks_over_N * self.threads_per_row,
        )  # [4, ~N]

        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, self.dtype, num_bits_per_copy=self.bits_read)
        thr_layout = cute.make_ordered_layout(
            (self.num_warps, self.threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, self.vec_load_size))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        blocks = cute.ceil_div(gInput.shape[0], tiler_mn[0])
        self.kernel(gInput, gOutput, tiler_mn, tiled_copy).launch(
            grid=(blocks, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )

    # type hints are not optional!!!!
    @cute.kernel
    def kernel(
        self,
        gInput: cute.Tensor,
        gOutput: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        gX = cute.local_tile(gInput, tiler_mn, (bidx, 0))
        gY = cute.local_tile(gOutput, tiler_mn, (bidx, 0))
        # this thread is response for vectorized loads, striding 4 * 32 across the row
        tidxSlice = tiled_copy.get_slice(tidx)
        tidxIndices = tidxSlice.partition_S(gX)
        tidxRegs = cute.make_rmem_tensor_like(tidxIndices)
        cute.autovec_copy(tidxIndices, tidxRegs)

        tidxValues = tidxRegs.load()
        tidLocalMax = tidxValues.reduce(
            cute.ReductionOp.MAX, init_val=self.NEG_INF, reduction_profile=0
        )
        rowMax = cute.arch.warp_reduction_max(tidLocalMax)
        tidScaledLocalSum = cute.math.exp(tidxValues - rowMax)
        tidLocalSum = tidScaledLocalSum.reduce(
            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
        )
        rowSum = cute.arch.warp_reduction_sum(tidLocalSum)

        writeValues = cute.math.exp(tidxValues - rowMax) / rowSum
        tidxRegs.store(writeValues)
        tidxOutIndices = tidxSlice.partition_D(gY)
        cute.autovec_copy(tidxRegs, tidxOutIndices)


def benchmark(loopless=True):
    import time

    dim = -1
    M, N = 4096, 768
    dtype = torch.float32
    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[dtype]

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    output = torch.zeros_like(x)

    if loopless:
        dx = x
        dy = output
        m = cute.sym_int()
        input_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, N), stride_order=(1, 0))
        output_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, N), stride_order=(1, 0))
        softmax = SoftmaxOnline(dtype_map[dtype], N)
        fn = cute.compile(
            softmax,
            input_cute,
            output_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        fn(x, output)
    else:
        dx = from_dlpack(x, enable_tvm_ffi=True)
        dy = from_dlpack(output, enable_tvm_ffi=True)
        softmax = SoftmaxOnlineLoop(dtype_map[dtype])
        fn = cute.compile(
            softmax,
            dx,
            dy,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        fn(dx, dy)

    print("Correctness check:")
    expected = torch.nn.functional.softmax(x, dim=-1)
    is_close = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    print(f"  dim=-1: {'✓ PASS' if is_close else '✗ FAIL'}")
    if not is_close:
        max_diff = (output - expected).abs().max().item()
        print(f"         max diff: {max_diff}")

    print("\nBenchmarks:")

    # Warmup
    for _ in range(10):
        fn(dx, dy)
    torch.cuda.synchronize()

    # Benchmark our softmax
    start = time.perf_counter()
    for _ in range(100):
        fn(dx, dy)
    torch.cuda.synchronize()
    print(f"  softmax_online dim=-1: {(time.perf_counter() - start) * 10:.3f} ms")

    # Compare to PyTorch
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.nn.functional.softmax(x, dim=-1)
    torch.cuda.synchronize()
    print(f"  torch.softmax dim=-1:  {(time.perf_counter() - start) * 10:.3f} ms")


def bench_back():
    dim = -1
    M, N = 256, 256
    dtype = torch.float32
    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[dtype]
    dy = torch.randn(M, N, device="cuda", dtype=dtype)
    y = torch.randn(M, N, device="cuda", dtype=dtype)
    dx = torch.randn(M, N, device="cuda", dtype=dtype)

    m = cute.sym_int()
    dy_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, N), stride_order=(1, 0))
    y_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, N), stride_order=(1, 0))
    dx_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, N), stride_order=(1, 0))
    softmax = SoftmaxOnlineBackward(dtype_map[dtype], N)
    fn = cute.compile(
        softmax,
        dy_cute,
        y_cute,
        dx_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    fn(dy, y, dx)


# benchmark(loopless=False)
# benchmark(loopless=True)

# bench_back()
