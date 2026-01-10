import torch


def copy_transpose(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("copy_transpose expects a 2D tensor")
    return x.transpose(0, 1).contiguous()
