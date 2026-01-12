import math
from typing import Dict, Iterable, List, Sequence

import torch

_L2_CACHE_TENSORS: Dict[tuple[int, int], torch.Tensor] = {}


def _get_l2_cache_tensor(size_bytes: int, device: int) -> torch.Tensor:
    key = (device, size_bytes)
    cache = _L2_CACHE_TENSORS.get(key)
    if cache is None or cache.device.index != device:
        cache = torch.empty(size_bytes, device=device, dtype=torch.uint8)
        _L2_CACHE_TENSORS[key] = cache
    return cache


def _clear_l2_cache(size_bytes: int, device: int) -> None:
    cache = _get_l2_cache_tensor(size_bytes, device)
    # Touch enough memory to evict L2 cache lines.
    cache.zero_()


def _default_l2_cache_bytes(device: int) -> int:
    props = torch.cuda.get_device_properties(device)
    l2_size = getattr(props, "l2_cache_size", 0)
    if not l2_size:
        # Fall back to a conservative size if the property is missing.
        return 32 * 1024 * 1024
    return int(l2_size) * 2


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def do_bench(
    fn,
    warmup: int = 10,
    rep: int = 100,
    *,
    grad_to_none: Iterable[torch.Tensor] | None = None,
    flush_l2: bool = True,
    l2_cache_bytes: int | None = None,
) -> List[float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking")
    device = torch.cuda.current_device()
    if flush_l2:
        if l2_cache_bytes is None:
            l2_cache_bytes = _default_l2_cache_bytes(device)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms: List[float] = []
    for _ in range(rep):
        if grad_to_none is not None:
            for tensor in grad_to_none:
                tensor.grad = None
        if flush_l2:
            _clear_l2_cache(l2_cache_bytes, device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def summarize_times(times_ms: Iterable[float]) -> Dict[str, float]:
    times = list(times_ms)
    if not times:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p90_ms": float("nan")}
    times_sorted = sorted(times)
    mean = sum(times_sorted) / len(times_sorted)
    return {
        "mean_ms": mean,
        "p50_ms": _percentile(times_sorted, 0.5),
        "p90_ms": _percentile(times_sorted, 0.9),
    }


def bytes_per_ms_to_gbps(bytes_per_ms: float) -> float:
    return bytes_per_ms / 1e6


def estimate_bandwidth(bytes_moved: int, time_ms: float) -> float:
    if time_ms <= 0:
        return float("inf")
    return bytes_per_ms_to_gbps(bytes_moved / time_ms)
