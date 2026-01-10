import torch


def softmax_online(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("softmax_online expects a 2D tensor")
    if dim not in (-1, 0, 1):
        raise ValueError("softmax_online expects dim in {-1, 0, 1} for 2D tensors")
    x_dtype = x.dtype
    return x.float().softmax(dim=dim).to(x_dtype)
