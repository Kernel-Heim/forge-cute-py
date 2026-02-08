import pytest
import torch

from forge_cute_py.ops import reduce_sum
from forge_cute_py.ref import reduce_sum as ref_reduce_sum


BASE_M = [128, 512, 2048]
BASE_N = [128, 256, 1024, 2048, 4096, 8192]
BASE_SHAPES = [(m, n) for m in BASE_M for n in BASE_N]


@pytest.mark.parametrize("shape", BASE_SHAPES)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_reduce_sum_correctness(shape, dtype, atol, rtol):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    y = reduce_sum(x, dim=-1)
    y_ref = ref_reduce_sum(x, dim=-1)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

    assert torch.isfinite(y).all()


def test_reduce_sum_dim1_alias():
    x = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    y = reduce_sum(x, dim=1)
    y_ref = ref_reduce_sum(x, dim=1)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)


def test_reduce_sum_torch_compile():
    unsupported_exc = ()
    try:
        from torch._dynamo.exc import Unsupported as DynamoUnsupported

        unsupported_exc = (DynamoUnsupported,)
    except Exception:
        unsupported_exc = ()
    try:
        compiled = torch.compile(reduce_sum, fullgraph=True)
    except Exception as exc:
        pytest.skip(f"torch.compile not available for reduce_sum: {exc}")
    x = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    try:
        y = compiled(x)
    except unsupported_exc as exc:
        pytest.skip(f"torch.compile unsupported for reduce_sum op: {exc}")
    y_ref = ref_reduce_sum(x, -1)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
