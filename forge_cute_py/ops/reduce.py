import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack

from forge_cute_py.kernels.reduce import Reduction

_compile_cache = {}
_REDUCE_OP_ALIASES = {
    "sum": "sum",
    "amax": "amax",
    "max": "amax",
    "amin": "amin",
    "min": "amin",
    "prod": "prod",
}


def _normalize_op(op: str) -> str:
    try:
        return _REDUCE_OP_ALIASES[op]
    except KeyError as exc:
        raise ValueError(f"Unsupported reduce op={op}.") from exc


@torch.library.custom_op("forge_cute_py::_reduce", mutates_args={"out"})
def _reduce(x: torch.Tensor, out: torch.Tensor, dim: int = -1, op: str = "sum") -> None:
    """Row-wise reduction using CuTe DSL.

    Args:
        x: Input tensor of shape (M, N)
        out: Output tensor (mutated in-place)
        dim: Dimension to reduce over (-1 or 1)
        op: Reduction op (sum only for kernel; future: amax/amin/prod)
    """
    assert x.dim() == 2, "reduce expects a 2D tensor"
    assert x.is_cuda, f"reduce is CUDA-only, got device={x.device}"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"Unsupported dtype: {x.dtype}"
    )

    dim = dim if dim >= 0 else x.ndim + dim
    if dim != 1:
        raise ValueError(f"reduce supports dim in {{-1, 1}} (row-wise), got {dim}")
    op = _normalize_op(op)
    if op != "sum":
        raise NotImplementedError(f"reduce op {op} not implemented in kernel yet; supported: sum")

    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[x.dtype]

    elem_bytes = x.element_size()
    vec_size = 16 // elem_bytes
    threads_per_row = 32
    tile_m = 4

    if x.shape[0] % tile_m != 0:
        raise ValueError(
            f"reduce requires M divisible by {tile_m} for tiled reduction. Got M={x.shape[0]}."
        )
    if x.shape[1] % (vec_size * threads_per_row) != 0:
        raise ValueError(
            "reduce requires N divisible by "
            f"{vec_size * threads_per_row} for vectorized loads. Got N={x.shape[1]}."
        )
    if x.data_ptr() % 16 != 0 or (x.stride(0) * elem_bytes) % 16 != 0:
        raise ValueError(
            "reduce requires 16-byte aligned rows for vectorized loads. "
            f"Got data_ptr alignment={x.data_ptr() % 16} and row_stride_bytes={x.stride(0) * elem_bytes}."
        )

    compile_key = (cute_dtype, x.shape[1], op, dim)
    if compile_key not in _compile_cache:
        m = cute.sym_int()
        n = x.shape[1]
        input_cute = cute.runtime.make_fake_compact_tensor(
            cute_dtype, (m, n), stride_order=(1, 0), assumed_align=16
        )
        output_cute = cute.runtime.make_fake_compact_tensor(
            cute_dtype, (m,), stride_order=(0,), assumed_align=16
        )
        kernel_class = Reduction(cute_dtype, n, reduction_op=op, dim=dim)
        _compile_cache[compile_key] = cute.compile(
            kernel_class,
            input_cute,
            output_cute,
            options="--enable-tvm-ffi",
        )

    x_cute = from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
    out_cute = from_dlpack(out, assumed_align=16, enable_tvm_ffi=True)
    _compile_cache[compile_key](x_cute, out_cute)


def reduce(x: torch.Tensor, dim: int = -1, op: str = "sum") -> torch.Tensor:
    """Row-wise reduction.

    Args:
        x: Input tensor of shape (M, N)
        dim: Dimension to reduce over (-1 for last dim, or 1)
        op: Reduction op (sum only for kernel; future: amax/amin/prod)

    Returns:
        Reduced tensor of shape (M,)
    """
    dim = dim if dim >= 0 else x.ndim + dim
    if dim != 1:
        raise ValueError(f"Invalid dim={dim} for row-wise reduce")

    out = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)
    if not x.is_contiguous():
        x = x.contiguous()
    _reduce(x, out, dim, op)
    return out
