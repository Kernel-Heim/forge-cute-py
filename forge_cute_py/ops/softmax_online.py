import torch
from cutlass import BFloat16, Float16, Float32
import cutlass.cute as cute
from forge_cute_py.kernels.softmax_online import SoftmaxOnline, SoftmaxOnlineBackward


@torch.library.custom_op("forge_cute_py::_softmax_fwd", mutates_args={"out"})
def _softmax_fwd(x: torch.Tensor, out: torch.Tensor, dim: int = -1) -> None:
    """Softmax forward pass.

    Args:
        x: Input tensor of shape (M, N)
        out: Output tensor of same shape as x (mutated in-place)
        dim: Dimension to apply softmax over
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64], (
        "Unsupported dtype"
    )
    assert out.shape == x.shape, "Output shape must match input"

    # Normalize dim to positive index
    dim = dim if dim >= 0 else x.ndim + dim
    assert dim in [0, 1], f"dim must be 0 or 1 for 2D tensors, got {dim}"
    assert x.shape[1] % 32 == 0, f"Inner dimension N must be a multiple of 32, got {x.shape[1]}"

    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }

    if x.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    cute_dtype = dtype_map[x.dtype]
    compile_key = (cute_dtype, x.shape[1])

    if compile_key not in _softmax_fwd.compile_cache:
        m = cute.sym_int()
        # n = cute.sym_int()
        n = x.shape[1]
        input_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        output_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        # Compile and cache the kernel
        _softmax_fwd.compile_cache[compile_key] = cute.compile(
            SoftmaxOnline(cute_dtype, n),
            input_cute,
            output_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _softmax_fwd.compile_cache[compile_key](x, out)


_softmax_fwd.compile_cache = {}


def softmax_fwd(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Forward-only softmax (no autograd)."""
    out = torch.empty_like(x)
    _softmax_fwd(x, out, dim)
    return out


@torch.library.custom_op("forge_cute_py::_softmax_backward", mutates_args={"dx"})
def _softmax_backward(dy: torch.Tensor, y: torch.Tensor, dx: torch.Tensor, dim: int = -1) -> None:
    """Softmax backward pass.

    For softmax output y = softmax(x), gradient: grad_x = y * (grad_y - dot)
    where dot = (grad_y * y).sum(dim, keepdim=True)

    Args:
        dy: Upstream gradients (M, N)
        y: Softmax output (M, N)
        dx: Input gradients (mutated in-place)
        dim: Dimension softmax was applied over
    """
    assert dy.dim() == 2 and y.dim() == 2, "Tensors must be 2D"
    assert dy.shape == y.shape == dx.shape, "All tensors must have same shape"
    assert dy.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    assert dy.dtype == y.dtype, "dy and y must have same dtype"
    assert dy.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64], (
        "Unsupported dtype"
    )

    # Normalize dim
    dim = dim if dim >= 0 else dy.ndim + dim
    assert dim in [0, 1], f"dim must be 0 or 1 for 2D, got {dim}"
    assert dy.shape[1] % 32 == 0, f"Inner dimension N must be a multiple of 32, got {dy.shape[1]}"

    # Compute gradient (numerically stable)
    # dot_product = (dy * y).sum(dim=dim, keepdim=True)
    # result = y * (dy - dot_product)
    # dx.copy_(result)

    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }

    cute_dtype = dtype_map[dy.dtype]
    compile_key = (cute_dtype, dy.shape[1])

    if compile_key not in _softmax_backward.compile_cache:
        m = cute.sym_int()
        n = dy.shape[1]
        dy_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        y_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        dx_cute = cute.runtime.make_fake_compact_tensor(cute_dtype, (m, n), stride_order=(1, 0))
        # Compile and cache the kernel
        _softmax_backward.compile_cache[compile_key] = cute.compile(
            SoftmaxOnlineBackward(cute_dtype, n),
            dy_cute,
            y_cute,
            dx_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _softmax_backward.compile_cache[compile_key](dy, y, dx)


_softmax_backward.compile_cache = {}


def softmax_bwd(dy: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Backward-only softmax (no autograd)."""
    dx = torch.empty_like(dy)
    _softmax_backward(dy, y, dx, dim)
    return dx


class SoftmaxOnlineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        y = softmax_fwd(x, dim)
        ctx.save_for_backward(y)
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        dx = softmax_bwd(dy, y, ctx.dim)
        return dx, None


def softmax_online(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Online softmax with automatic differentiation support.

    Args:
        x: Input tensor of shape (M, N)
        dim: Dimension to apply softmax (-1, 0, or 1)

    Returns:
        Softmax output tensor of same shape and dtype as input

    Examples:
        >>> x = torch.randn(32, 128, device='cuda', requires_grad=True)
        >>> y = softmax_online(x, dim=-1)
        >>> loss = y.sum()
        >>> loss.backward()  # Gradients computed automatically
    """
    return SoftmaxOnlineFunction.apply(x, dim)
