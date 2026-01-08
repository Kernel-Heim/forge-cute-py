import pytest
import torch


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
