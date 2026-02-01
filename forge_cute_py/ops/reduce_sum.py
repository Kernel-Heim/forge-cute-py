from __future__ import annotations

from typing import Callable, Type

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32

from forge_cute_py.kernels.reduce_sum import Reduction, Variant

_DTYPE_TO_CUTLASS: dict[torch.dtype, Type[cutlass.Numeric]] = {
    torch.float16: Float16,
    torch.float32: Float32,
    torch.bfloat16: BFloat16,
}

_SUPPORTED_VARIANTS: tuple[str, ...] = ("naive", "improved", "shfl")
_SUPPORTED_DTYPE: tuple[torch.dtype, ...] = tuple(_DTYPE_TO_CUTLASS.keys())

# Compile cache keyed by: (cute_dtype, dim, variant, M, N)
_COMPILE_CACHE: dict[tuple[Type[cutlass.Numeric], int, str, int, int], Callable] = {}


def _normalize_dim(ndim: int, dim: int) -> int:
    if dim < 0:
        return ndim + dim
    return dim


def _get_cute_dtype(dtype: torch.dtype) -> Type[cutlass.Numeric]:
    if dtype not in _DTYPE_TO_CUTLASS:
        raise AssertionError(f"Unsupported dtype: {dtype}")
    return _DTYPE_TO_CUTLASS[dtype]


def _compile_reduction_kernel(
    cute_dtype: Type[cutlass.Numeric],
    M: int,
    N: int,
    dim: int = 1,
    variant: Variant = "shfl",
) -> Callable:
    m = cute.sym_int()
    n = cute.sym_int()
    input_cute = cute.runtime.make_fake_compact_tensor(
        cute_dtype,
        (m, n),
        stride_order=(1, 0),
    )
    # Output shape depends on dim: (M,) for row reduction, (N,) for column reduction
    out_sym = m if dim == 1 else n
    output_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (out_sym,))
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        Reduction(
            cute_dtype,
            N=N,
            M=M,
            dim=dim,
            variant=variant,
        ),
        input_cute,
        output_cute,
        fake_stream,
        options="--enable-tvm-ffi",
    )


def _validate_inputs(x: torch.Tensor, out: torch.Tensor | None, dim: int, variant: str) -> None:
    assert x.dim() == 2, "reduce_sum expects a 2D tensor"
    assert x.is_cuda, f"reduce_sum is CUDA-only, got device={x.device}"
    assert dim in (-1, 0, 1), f"reduce_sum expects dim in {{-1, 0, 1}}, got {dim}"
    assert variant in _SUPPORTED_VARIANTS, f"Unsupported variant: {variant}"
    assert x.dtype in _SUPPORTED_DTYPE, f"Unsupported dtype: {x.dtype}"

    if out is None:
        return

    assert out.is_cuda, "out must be CUDA tensor"
    assert out.is_contiguous(), "out must be contiguous"
    assert out.dtype == x.dtype, "out dtype must match x dtype"


def _get_or_compile_kernel(
    cute_dtype: Type[cutlass.Numeric],
    dim: int,
    variant: str,
    M: int,
    N: int,
) -> Callable:
    key = (cute_dtype, dim, variant, M, N)
    kernel = _COMPILE_CACHE.get(key)
    if kernel is None:
        kernel = _compile_reduction_kernel(cute_dtype, M, N, dim=dim, variant=variant)
        _COMPILE_CACHE[key] = kernel
    return kernel


@torch.library.custom_op("forge_cute_py::_reduce_sum", mutates_args={"out"})
def _reduce_sum(
    x: torch.Tensor,
    out: torch.Tensor,
    dim: int = -1,
    variant: str = "shfl",
) -> None:
    """Row/column sum reduction using CuTe kernel.

    Args:
        x: Input tensor of shape (M, N)
        out: Output tensor (mutated in-place)
        dim: Dimension to reduce over (-1, 0, or 1)
        variant: Reduction variant ("naive", "improved", "shfl")
    """
    _validate_inputs(x, out, dim=dim, variant=variant)

    normalized_dim = _normalize_dim(x.ndim, dim)
    M, N = x.shape
    cute_dtype = _get_cute_dtype(x.dtype)

    if normalized_dim == 1:
        assert out.shape == (M,), f"out must be shape {(M,)}, got {tuple(out.shape)}"
        kernel = _get_or_compile_kernel(cute_dtype, dim=1, variant=variant, M=M, N=N)
        kernel(x, out)
        return

    if normalized_dim == 0:
        assert out.shape == (N,), f"out must be shape {(N,)}, got {tuple(out.shape)}"
        # Direct column reduction: out[n] = sum_m x[m, n]
        kernel = _get_or_compile_kernel(cute_dtype, dim=0, variant=variant, M=M, N=N)
        kernel(x, out)
        return

    raise ValueError(f"Invalid dim={normalized_dim} for 2D tensor")


def reduce_sum(x: torch.Tensor, dim: int = -1, variant: str = "shfl") -> torch.Tensor:
    """Row/column sum reduction.

    Args:
        x: Input tensor of shape (M, N)
        dim: -1 / 0 / 1
        variant: "naive" / "improved" / "shfl"
    """
    _validate_inputs(x, out=None, dim=dim, variant=variant)

    normalized_dim = _normalize_dim(x.ndim, dim)
    M, N = x.shape
    out_shape = (M,) if normalized_dim == 1 else (N,)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    _reduce_sum(x, out, dim=normalized_dim, variant=variant)
    return out
