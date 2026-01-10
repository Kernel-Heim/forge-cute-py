import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32

from forge_cute_py.kernels.copy_transpose import CopyTranspose


def _copy_transpose(x: torch.Tensor, tile_size: int = 16) -> torch.Tensor:
    """
    Perform tiled transpose using CuTe DSL.

    Args:
        x: Input tensor of shape (M, N)
        tile_size: Tile size (default: 16)

    Returns:
        Transposed tensor of shape (N, M)
    """
    if not x.is_cuda:
        raise NotImplementedError("copy_transpose is CUDA-only")
    if x.ndim != 2:
        raise ValueError("copy_transpose expects a 2D tensor")

    M, N = x.shape

    # Create output tensor (transposed shape)
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)

    # Map PyTorch dtype to CUTLASS dtype
    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }

    if x.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    cute_dtype = dtype_map[x.dtype]

    compile_key = (cute_dtype, tile_size)

    if compile_key not in _copy_transpose.compile_cache:
        m = cute.sym_int()
        n = cute.sym_int()
        input_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        output_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (n, m), stride_order=(1, 0))
        # Compile and cache the kernel
        _copy_transpose.compile_cache[compile_key] = cute.compile(
            CopyTranspose(cute_dtype, tile_size=tile_size),
            input_cute,
            output_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _copy_transpose.compile_cache[compile_key](x, y)

    return y


# Initialize compile cache for the kernel
_copy_transpose.compile_cache = {}
