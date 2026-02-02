import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack

from forge_cute_py.kernels.reduce_sum import ReduceSumLast, ReduceSumFirst

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
    compile_key = (cute_dtype, dim)

    if compile_key not in _compile_cache:
        m = cute.sym_int()
        n = cute.sym_int()
        input_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        
        if dim == 1:  # Reduce last dim
            output_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m,))
            kernel_class = ReduceSumLast(cute_dtype)
        else:  # dim == 0
            output_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (n,))
            kernel_class = ReduceSumFirst(cute_dtype)

        _compile_cache[compile_key] = cute.compile(
            kernel_class,
            input_cute,
            output_cute,
            # cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    x_cute = from_dlpack(x, assumed_align=16)
    out_cute = from_dlpack(out, assumed_align=16)
    _compile_cache[compile_key](x_cute, out_cute)


def reduce_sum(x: torch.Tensor, dim: int = -1, variant='') -> torch.Tensor:
    """Sum reduction with CuTe DSL kernel."""
    dim = dim if dim >= 0 else x.ndim + dim
    out_shape = (x.shape[1],) if dim == 0 else (x.shape[0],)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    _reduce_sum(x, out, dim)
    return out
