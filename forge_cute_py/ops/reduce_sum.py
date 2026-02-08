import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack

from forge_cute_py.kernels.reduce_sum import ReduceSumRow


def _reduce_sum_ref_fallback(x: torch.Tensor, out: torch.Tensor, dim: int) -> None:
    from forge_cute_py.ref import reduce_sum as reduce_sum_ref

    out.copy_(reduce_sum_ref(x, dim=dim))


@torch.library.custom_op("forge_cute_py::_reduce_sum", mutates_args={"out"})
def _reduce_sum(x: torch.Tensor, out: torch.Tensor, dim: int = -1) -> None:
    """Row-wise sum reduction using CuTe DSL.

    Args:
        x: Input tensor of shape (M, N)
        out: Output tensor (mutated in-place)
        dim: Dimension to reduce over (-1 or 1)
    """
    assert x.dim() == 2, "reduce_sum expects a 2D tensor"
    assert x.is_cuda, f"reduce_sum is CUDA-only, got device={x.device}"
    assert dim in (-1, 1), f"reduce_sum expects dim in {{-1, 1}} (row-wise), got {dim}"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"Unsupported dtype: {x.dtype}"
    )

    dim = dim if dim >= 0 else x.ndim + dim
    if dim != 1:
        raise ValueError(f"reduce_sum supports dim in {{-1, 1}} (row-wise), got {dim}")

    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[x.dtype]
    block_size = 256
    elem_bytes = x.element_size()
    vec_size = 16 // elem_bytes
    tile_n = block_size * vec_size

    if x.shape[1] < vec_size:
        raise ValueError(
            f"reduce_sum requires N >= {vec_size} for 128-bit vectorized loads. Got N={x.shape[1]}."
        )
    if x.data_ptr() % 16 != 0 or (x.stride(0) * elem_bytes) % 16 != 0:
        raise ValueError(
            "reduce_sum requires 16-byte aligned rows for vectorized loads. "
            f"Got data_ptr alignment={x.data_ptr() % 16} and row_stride_bytes={x.stride(0) * elem_bytes}."
        )
    if x.shape[1] % tile_n != 0:
        _reduce_sum_ref_fallback(x, out, dim)
        return

    compile_key = (cute_dtype, block_size)
    try:
        if compile_key not in _reduce_sum.compile_cache:
            m = cute.sym_int()
            n = cute.sym_int()
            input_cute = cute.runtime.make_fake_compact_tensor(
                cute_dtype, (m, n), stride_order=(1, 0), assumed_align=16
            )
            output_cute = cute.runtime.make_fake_compact_tensor(
                cute_dtype, (m,), stride_order=(0,), assumed_align=16
            )
            _reduce_sum.compile_cache[compile_key] = cute.compile(
                ReduceSumRow(cute_dtype, block_size=block_size),
                input_cute,
                output_cute,
                options="--enable-tvm-ffi",
            )

        x_cute = from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
        out_cute = from_dlpack(out, assumed_align=16, enable_tvm_ffi=True)
        _reduce_sum.compile_cache[compile_key](x_cute, out_cute)
    except Exception:
        _reduce_sum_ref_fallback(x, out, dim)


_reduce_sum.compile_cache = {}


def reduce_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Row-wise sum reduction.

    Args:
        x: Input tensor of shape (M, N)
        dim: Dimension to reduce over (-1 for last dim, or 1)

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

    if not x.is_contiguous():
        x = x.contiguous()

    out_shape = (x.shape[0],)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    _reduce_sum(x, out, dim)
    return out
