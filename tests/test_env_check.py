import importlib.util
import subprocess
import sys

import pytest


def test_env_check_module_runs():
    if importlib.util.find_spec("forge_cute_py.env_check") is None:
        pytest.skip("forge_cute_py.env_check not implemented yet")
    result = subprocess.run(
        [sys.executable, "-m", "forge_cute_py.env_check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"env_check failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}\n")
