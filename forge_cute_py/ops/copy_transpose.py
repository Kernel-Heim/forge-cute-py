import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32

from forge_cute_py.kernels.copy_transpose import CopyTranspose


@torch.library.custom_op("forge_cute_py::_copy_transpose", mutates_args={"out"})
def _copy_transpose(x: torch.Tensor, out: torch.Tensor, tile_size: int = 16) -> None:
    """Tiled transpose using CuTe DSL.

    Args:
        x: Input tensor of shape (M, N)
        out: Output tensor of shape (N, M) (mutated in-place)
        tile_size: Tile size for blocking (default: 16)
    """
    assert x.dim() == 2, "copy_transpose expects a 2D tensor"
    assert x.is_cuda, f"copy_transpose is CUDA-only, got device={x.device}"
    assert out.shape == (x.shape[1], x.shape[0]), "Output shape must be transposed input shape"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

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

    _copy_transpose.compile_cache[compile_key](x, out)


_copy_transpose.compile_cache = {}


def copy_transpose(x: torch.Tensor, tile_size: int = 16) -> torch.Tensor:
    """Tiled transpose with CuTe DSL kernel.

    Args:
        x: Input tensor of shape (M, N)
        tile_size: Tile size for blocking (default: 16)

    Returns:
        Transposed tensor of shape (N, M)

    Examples:
        >>> x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
        >>> y = copy_transpose(x, tile_size=16)
        >>> y.shape
        torch.Size([1024, 1024])
    """
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    _copy_transpose(x, out, tile_size)
    return out
