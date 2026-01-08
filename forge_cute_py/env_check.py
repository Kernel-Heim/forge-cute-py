import sys

import torch

from forge_cute_py.ref import copy_transpose as copy_transpose_ref


def _print_env():
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    props = torch.cuda.get_device_properties(0)
    print(f"gpu: {props.name}")
    print(f"sm: {props.major}.{props.minor}")
    print(f"arch list: {torch.cuda.get_arch_list()}")


def _check_cute_import():
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"CuTe DSL import failed: {exc}") from exc
    print("cute dsl: import ok")


def _check_copy_transpose():
    if not hasattr(torch.ops, "forge_cute_py") or not hasattr(
        torch.ops.forge_cute_py, "copy_transpose"
    ):
        raise RuntimeError("torch.ops.forge_cute_py.copy_transpose not registered")
    x = torch.arange(0, 16, device="cuda", dtype=torch.float32).reshape(4, 4)
    y = torch.ops.forge_cute_py.copy_transpose(x, 16)
    y_ref = copy_transpose_ref(x)
    torch.testing.assert_close(y, y_ref)
    print("copy_transpose: ok")


def main():
    _print_env()
    _check_cute_import()
    _check_copy_transpose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
