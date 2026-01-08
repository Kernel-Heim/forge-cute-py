import torch


def copy_transpose(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "Input must be a 2D tensor"
    return x.transpose(0, 1).contiguous()
