import torch


_REDUCE_OP_ALIASES = {
    "sum": "sum",
    "amax": "amax",
    "max": "amax",
    "amin": "amin",
    "min": "amin",
    "prod": "prod",
}


def _normalize_op(op: str) -> str:
    try:
        return _REDUCE_OP_ALIASES[op]
    except KeyError as exc:
        raise ValueError(f"Unsupported reduce op={op}.") from exc


def reduce(x: torch.Tensor, dim: int = -1, op: str = "sum") -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("reduce expects a 2D tensor")
    if dim not in (-1, 0, 1):
        raise ValueError("reduce expects dim in {-1, 0, 1} for 2D tensors")
    op = _normalize_op(op)
    x_dtype = x.dtype
    x_fp32 = x.float()
    if op == "sum":
        return x_fp32.sum(dim=dim).to(x_dtype)
    if op == "amax":
        return x_fp32.amax(dim=dim).to(x_dtype)
    if op == "amin":
        return x_fp32.amin(dim=dim).to(x_dtype)
    if op == "prod":
        return x_fp32.prod(dim=dim).to(x_dtype)
    raise ValueError(f"Unhandled reduce op={op}.")
