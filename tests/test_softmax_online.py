import pytest
import torch

from forge_cute_py.ops import softmax_online
from forge_cute_py.ref import softmax_online as ref_softmax_online


@pytest.mark.parametrize("shape", [(4, 8), (2, 128)])
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-3, 1e-3),
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_softmax_online_correctness(shape, dim, dtype, atol, rtol):
    x = (0.1 * torch.randn(*shape, device="cuda", dtype=dtype)).requires_grad_(True)
    y = softmax_online(x, dim)
    y_ref = ref_softmax_online(x, dim=dim)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

    assert torch.isfinite(y).all()


@pytest.mark.parametrize("shape", [(4, 8), (2, 128)])
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-3, 1e-3),
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_softmax_online_torch_compile(shape, dim, dtype, atol, rtol):
    unsupported_exc = ()
    try:
        from torch._dynamo.exc import Unsupported as DynamoUnsupported

        unsupported_exc = (DynamoUnsupported,)
    except Exception:
        unsupported_exc = ()
    try:
        compiled = torch.compile(softmax_online, fullgraph=True)
    except Exception as exc:
        pytest.skip(f"torch.compile not available for softmax_online: {exc}")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    try:
        y = compiled(x, dim=dim)
    except unsupported_exc as exc:
        pytest.skip(f"torch.compile unsupported for softmax_online op: {exc}")
    y_ref = ref_softmax_online(x, dim=dim)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_softmax_online_properties(input_dtype):
    x = torch.randn(16, 256, device="cuda", dtype=input_dtype)
    y = softmax_online(x, -1)
    sums = torch.sum(y, dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3)
    assert (y >= 0).all()
    assert (y <= 1).all()


def test_softmax_online_translation_invariance():
    x = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    y = softmax_online(x, -1)
    y_shifted = softmax_online(x + 100.0, -1)
    torch.testing.assert_close(y, y_shifted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_softmax_online_extreme_values(input_dtype):
    m, n = 8, 256
    x_large = torch.full((m, n), 10.0, device="cuda", dtype=input_dtype)
    out_large = softmax_online(x_large, -1)
    expected = torch.full_like(out_large, 1.0 / n)
    torch.testing.assert_close(out_large, expected, atol=1e-3, rtol=1e-3)
    x_small = torch.full((m, n), -10.0, device="cuda", dtype=input_dtype)
    out_small = softmax_online(x_small, -1)
    torch.testing.assert_close(out_small, expected, atol=1e-3, rtol=1e-3)
    x_mixed = torch.zeros((m, n), device="cuda", dtype=input_dtype)
    x_mixed[:, 0] = 10.0
    x_mixed[:, 1:] = -10.0
    out_mixed = softmax_online(x_mixed, -1)
    assert (out_mixed[:, 0] > 0.99).all()
    assert (out_mixed[:, 1:] < 0.01).all()
