import pytest


def require_cuda():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import error path
        pytest.skip(f"torch unavailable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; skipping CUDA-only tests")
    return torch
