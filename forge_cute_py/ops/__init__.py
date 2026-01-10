import torch

from forge_cute_py.ops.copy_transpose import _copy_transpose

# from forge_cute_py.ops.reduce_sum import reduce_sum
# from forge_cute_py.ops.softmax_online import softmax_online
# Using reference stubs for now
from forge_cute_py.ref import reduce_sum as reduce_sum_ref
from forge_cute_py.ref import softmax_online as softmax_online_ref

_LIB = torch.library.Library("forge_cute_py", "DEF")
_LIB.define("copy_transpose(Tensor x, int tile_size=16) -> Tensor")
_LIB.define("reduce_sum(Tensor x, int dim=-1, str variant='shfl') -> Tensor")
_LIB.define("softmax_online(Tensor x, int dim=-1) -> Tensor")


def _require_cuda(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise NotImplementedError("forge_cute_py ops are CUDA-only")


@torch.library.impl(_LIB, "copy_transpose", "CUDA")
def _copy_transpose_cuda(x: torch.Tensor, tile_size: int = 16) -> torch.Tensor:
    return _copy_transpose(x, tile_size=tile_size)


@torch.library.impl(_LIB, "reduce_sum", "CUDA")
def _reduce_sum_cuda(x: torch.Tensor, dim: int = -1, variant: str = "shfl") -> torch.Tensor:
    _require_cuda(x)
    if x.ndim != 2:
        raise ValueError("reduce_sum expects a 2D tensor")
    if dim not in (-1, 0, 1):
        raise ValueError("reduce_sum expects dim in {-1, 0, 1} for 2D tensors")
    return reduce_sum_ref(x, dim=dim)


@torch.library.impl(_LIB, "softmax_online", "CUDA")
def _softmax_online_cuda(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    _require_cuda(x)
    if x.ndim != 2:
        raise ValueError("softmax_online expects a 2D tensor")
    if dim not in (-1, 0, 1):
        raise ValueError("softmax_online expects dim in {-1, 0, 1} for 2D tensors")
    return softmax_online_ref(x, dim=dim)


def copy_transpose(x: torch.Tensor, tile_size: int = 16) -> torch.Tensor:
    return torch.ops.forge_cute_py.copy_transpose(x, tile_size)


def reduce_sum(x: torch.Tensor, dim: int = -1, variant: str = "shfl") -> torch.Tensor:
    return torch.ops.forge_cute_py.reduce_sum(x, dim, variant)


def softmax_online(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.ops.forge_cute_py.softmax_online(x, dim)


__all__ = ["copy_transpose", "reduce_sum", "softmax_online"]
