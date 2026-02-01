import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32

from forge_cute_py.kernels.reduce_sum import Reduction


@torch.library.custom_op("forge_cute_py::_reduce_sum", mutates_args={"out"})
def _reduce_sum(x: torch.Tensor, out: torch.Tensor, dim: int = -1, variant: str = "shfl") -> None:
    """Row/column sum reduction (reference implementation stub).

    Args:
        x: Input tensor of shape (M, N)
        out: Output tensor (mutated in-place)
        dim: Dimension to reduce over (-1, 0, or 1)
        variant: Reduction variant (naive, improved, shfl) - currently unused
    """
    assert x.dim() == 2, "reduce_sum expects a 2D tensor"
    assert x.is_cuda, f"reduce_sum is CUDA-only, got device={x.device}"
    assert dim in (-1, 0, 1), f"reduce_sum expects dim in {{-1, 0, 1}}, got {dim}"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"Unsupported dtype: {x.dtype}"
    )

    # Normalize dim to positive index
    dim = dim if dim >= 0 else x.ndim + dim

    # Map PyTorch dtype to CUTLASS dtype
    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    if x.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    cute_dtype = dtype_map[x.dtype]
    compile_key = (cute_dtype, dim, variant, x.shape[dim])

    if compile_key not in _reduce_sum.compile_cache:
        m = cute.sym_int() if dim != 0 else x.shape[0]
        n = cute.sym_int() if dim != 1 else x.shape[1]
        input_shape = (m, n)
        output_shape = (m,) if dim == 1 else (n,)
        input_cute = cute.runtime.make_fake_compact_tensor(
            cute_dtype, input_shape, stride_order=(1, 0)
        )
        output_cute = cute.runtime.make_fake_compact_tensor(
            cute_dtype, output_shape, stride_order=(0)
        )
        # Compile and cache the kernel
        _reduce_sum.compile_cache[compile_key] = cute.compile(
            Reduction(cute_dtype, n, reduction_op="sum", dim=dim, reduction_dtype=Float32),
            input_cute,
            output_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _reduce_sum.compile_cache[compile_key](x, out)


_reduce_sum.compile_cache = {}


def reduce_sum(x: torch.Tensor, dim: int = -1, variant: str = "shfl") -> torch.Tensor:
    """Row/column sum reduction.

    Args:
        x: Input tensor of shape (M, N)
        dim: Dimension to reduce over (-1 for last dim, 0 or 1)
        variant: Reduction variant (naive, improved, shfl) - currently unused

    Returns:
        Reduced tensor of shape (M,) if dim=1 or (N,) if dim=0

    Examples:
        >>> x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
        >>> y = reduce_sum(x, dim=-1)  # Sum over columns, result shape: (32,)
        >>> y.shape
        torch.Size([32])
    """
    # Normalize dim to positive index
    dim = dim if dim >= 0 else x.ndim + dim

    # Determine output shape
    if dim == 0:
        out_shape = (x.shape[1],)
    elif dim == 1:
        out_shape = (x.shape[0],)
    else:
        raise ValueError(f"Invalid dim={dim} for 2D tensor")

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    _reduce_sum(x, out, dim, variant)
    return out


if __name__ == "__main__":
    M = 1024
    N = 1024
    dtype = torch.float32
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    y = reduce_sum(x, dim=-1)
    ref_y = torch.sum(x, dim=-1)

    torch.testing.assert_close(y, ref_y)
