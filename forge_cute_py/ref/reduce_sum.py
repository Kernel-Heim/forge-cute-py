import torch


def reduce_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("reduce_sum expects a 2D tensor")
    if dim not in (-1, 1):
        raise ValueError("reduce_sum expects dim in {-1, 1} for 2D tensors")
    dim = dim if dim >= 0 else x.ndim + dim
    x_dtype = x.dtype
    return x.float().sum(dim=dim).to(x_dtype)
