import torch


@torch.library.custom_op("forge_cute_py::_reduce_sum", mutates_args={"out"})
def _reduce_sum(x: torch.Tensor, out: torch.Tensor, dim: int = -1) -> None:
    """Row/column sum reduction (reference implementation stub).

    Args:
        x: Input tensor of shape (M, N)
        out: Output tensor (mutated in-place)
        dim: Dimension to reduce over (-1, 0, or 1)
    """
    assert x.dim() == 2, "reduce_sum expects a 2D tensor"
    assert x.is_cuda, f"reduce_sum is CUDA-only, got device={x.device}"
    assert dim in (-1, 0, 1), f"reduce_sum expects dim in {{-1, 0, 1}}, got {dim}"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"Unsupported dtype: {x.dtype}"
    )

    # Normalize dim to positive index
    dim = dim if dim >= 0 else x.ndim + dim
    if dim != 1:
        raise ValueError(f"reduce_sum supports dim in {{-1, 1}} (row-wise), got {dim}")

    # For now, use reference implementation
    # Future: call kernel implementation based on variant when available
    from forge_cute_py.ref import reduce_sum as reduce_sum_ref

    result = reduce_sum_ref(x, dim=dim)
    out.copy_(result)


_reduce_sum.compile_cache = {}


def reduce_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Row/column sum reduction.

    Args:
        x: Input tensor of shape (M, N)
        dim: Dimension to reduce over (-1 for last dim, or 1)

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

    if dim != 1:
        raise ValueError(f"Invalid dim={dim} for row-wise reduce_sum")

    out_shape = (x.shape[0],)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    _reduce_sum(x, out, dim)
    return out
