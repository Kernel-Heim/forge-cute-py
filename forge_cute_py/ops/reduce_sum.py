import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack

from forge_cute_py.kernels.reduce_sum import ReduceSum

_compile_cache = {}


@torch.library.custom_op("forge_cute_py::_reduce_sum", mutates_args={"out"})
def _reduce_sum(x: torch.Tensor, out: torch.Tensor, dim: int = -1, variant: str = "default") -> None:
    """Row-wise sum reduction using CuTe DSL.

    Args:
        x: Input tensor of shape (M, N)
        out: Output tensor (mutated in-place)
        dim: Dimension to reduce over (-1 or 1)
        variant: Reduction variant (default only)
    """
    assert x.dim() == 2, "reduce_sum expects a 2D tensor"
    assert x.is_cuda, f"reduce_sum is CUDA-only, got device={x.device}"
    assert dim in (-1, 1), f"reduce_sum expects dim in {{-1, 1}}, got {dim}"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"Unsupported dtype: {x.dtype}"
    )

    dim = dim if dim >= 0 else x.ndim + dim

    if variant not in ("default", ""):
        raise NotImplementedError(f"reduce_sum variant {variant} not implemented")

    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[x.dtype]
    compile_key = (cute_dtype, dim)

    if compile_key not in _compile_cache:
        m = cute.sym_int()
        n = cute.sym_int()
        input_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        output_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m,), stride_order=(0,))
        kernel_class = ReduceSum(cute_dtype)
        _compile_cache[compile_key] = cute.compile(
            kernel_class,
            input_cute,
            output_cute,
            options="--enable-tvm-ffi",
        )

    x_cute = from_dlpack(x, assumed_align=1)
    out_cute = from_dlpack(out, assumed_align=1)
    _compile_cache[compile_key](x_cute, out_cute)


def reduce_sum(x: torch.Tensor, dim: int = -1, variant: str = "default") -> torch.Tensor:
    """Row-wise sum reduction.

    Args:
        x: Input tensor of shape (M, N)
        dim: Dimension to reduce over (-1 for last dim, or 1)
        variant: Reduction variant (default only)

    Returns:
        Reduced tensor of shape (M,)

    Examples:
        >>> x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
        >>> y = reduce_sum(x, dim=-1)  # Sum over columns, result shape: (32,)
        >>> y.shape
        torch.Size([32])
    """
    dim = dim if dim >= 0 else x.ndim + dim

    if dim != 1:
        raise ValueError(f"Invalid dim={dim} for row-wise reduce_sum")

    out = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)
    if not x.is_contiguous():
        x = x.contiguous()
    _reduce_sum(x, out, dim, variant)
    return out
