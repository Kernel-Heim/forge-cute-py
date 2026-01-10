import pytest
import torch

from forge_cute_py.ops import reduce_sum
from forge_cute_py.ref import reduce_sum as ref_reduce_sum


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((4, 8), -1),
        ((8, 4), 0),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize("variant", ["naive", "improved", "shfl"])
def test_reduce_sum_correctness(shape, dim, dtype, atol, rtol, variant):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    try:
        y = reduce_sum(x, dim=dim, variant=variant)
    except NotImplementedError:
        pytest.skip(f"reduce_sum variant {variant} not implemented")
    y_ref = ref_reduce_sum(x, dim=dim)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

    assert torch.isfinite(y).all()


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
    except NotImplementedError:
        pytest.skip("reduce_sum shfl variant not implemented")
    y_ref = ref_reduce_sum(x, -1)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
