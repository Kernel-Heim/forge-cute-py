import torch
import cutlass.cute as cute

from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack

from forge_cute_py.kernels.reduce_sum import _reduce_sum_last, _reduce_sum_first

_compile_cache = {}

@torch.library.custom_op("forge_cute_py::_reduce_sum", mutates_args={"out"})
def _reduce_sum(x: torch.Tensor, out: torch.Tensor, dim: int = -1) -> None:
    """Sum reduction using CuTe DSL."""
    assert x.dim() == 2, "reduce_sum expects a 2D tensor"
    assert x.is_cuda, f"reduce_sum is CUDA-only, got device={x.device}"

    dim = dim if dim >= 0 else x.ndim + dim

    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[x.dtype]
    
    compile_key = (cute_dtype, dim, x.shape)

    if compile_key not in _compile_cache:
        jit_fn = _reduce_sum_last if dim == 1 else _reduce_sum_first
        _compile_cache[compile_key] = cute.compile(
            jit_fn,
            from_dlpack(x, assumed_align=16),
            from_dlpack(out),
        )

    _compile_cache[compile_key](from_dlpack(x), from_dlpack(out))


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
    _reduce_sum(x, out, dim)
    return out
