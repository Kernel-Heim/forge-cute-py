import pytest
import torch

from forge_cute_py.ops import copy_transpose
from forge_cute_py.ref import copy_transpose as ref_copy_transpose


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 64), (128, 256), (15, 37)])
@pytest.mark.parametrize("tile_size", [16, 32])
def test_copy_transpose_correctness(dtype, shape, tile_size):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    y = copy_transpose(x, tile_size=tile_size)
    y_ref = ref_copy_transpose(x)
    # Use exact comparison since transpose should be exact
    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 64), (128, 256), (15, 37)])
@pytest.mark.parametrize("tile_size", [16, 32])
def test_copy_transpose_torch_ops(dtype, shape, tile_size):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    y = torch.ops.forge_cute_py.copy_transpose(x, tile_size)
    y_ref = ref_copy_transpose(x)
    # Use exact comparison since transpose should be exact
    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)
