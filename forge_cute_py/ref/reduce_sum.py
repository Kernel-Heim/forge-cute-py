import torch


def reduce_sum(x: torch.Tensor, dim: int = -1, variant: str = "shfl") -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("reduce_sum expects a 2D tensor")
    if dim not in (-1, 0, 1):
        raise ValueError("reduce_sum expects dim in {-1, 0, 1} for 2D tensors")
    x_dtype = x.dtype
    return x.float().sum(dim=dim).to(x_dtype)
