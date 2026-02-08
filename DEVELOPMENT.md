# DEVELOPMENT

This guide focuses on developing CuTe DSL kernels in `forge-cute-py`.
For quickstart usage, see `README.md`.

## Scope and goals

v0.1 targets KernelHeim Weeks 0-2:
- Week 0: copy/transpose
- Week 1: reductions (sum) variants
- Week 2: single-pass online softmax

Out of scope for v0: FlashAttention kernels, KV-cache decode, FP8, NCCL, C++
extensions.

## Project architecture

Each op follows a 3-layer pattern:
- `forge_cute_py/ops/<op>.py`: Python API, validation, torch.ops registration,
  and compile cache.
- `forge_cute_py/kernels/<op>.py`: CuTe DSL kernel implementation.
- `forge_cute_py/ref/<op>.py`: PyTorch reference implementation for validation.

### Modern PyTorch custom op pattern

All ops use `@torch.library.custom_op` for PyTorch integration with a two-function
pattern:

1. **Internal op** (`_op_name`): Decorated with `@torch.library.custom_op`,
   mutates output tensor in-place, handles kernel dispatch and compilation caching.
2. **Public API** (`op_name`): Allocates output tensor, calls internal op, returns
   result. For ops with autograd, wraps a `torch.autograd.Function`.

This enables usage via both `forge_cute_py.ops.op_name()` and
`torch.ops.forge_cute_py._op_name()`.

### Current implementation status (v0.1)

| Op | Kernel | Status |
|----|--------|--------|
| `copy_transpose` | CuTe DSL | Fully implemented with tile-based shared memory |
| `reduce` | CuTe DSL | Implemented for sum with benchmark coverage |
| `reduce_sum` | Reference | Stub/reference path with benchmark coverage |
| `softmax_online` | Reference + CuTe backend | Backend-registry flow (`ref` default, `kernel` optional); row-wise `dim=-1` only |

### Development flow

1) add or update reference (`ref/`)
2) implement kernel (`kernels/`)
3) expose in op wrapper (`ops/`)
4) add tests (`tests/`)
5) add benchmarks (`bench/`)

## Environment setup

```bash
uv sync
uv run python -m forge_cute_py.env_check
```

If `env_check` shows a warning about GPU capability vs PyTorch support, confirm
your CUDA/PyTorch stack matches the GPU on the machine.

## CuTe DSL constraints (must-read)

CuTe DSL supports a subset of Python and has strict control-flow/type rules.
Review these before writing kernels:

- Control flow: https://github.com/Dao-AILab/quack/blob/main/docs/dsl_control_flow.rst
- Limitations: https://github.com/Dao-AILab/quack/blob/main/docs/limitations.rst

Key constraints to keep in mind:
- Dynamic control flow cannot use early exit (`break`, `continue`, `return`,
  `raise`, `pass`).
- Values created inside dynamic control flow are not visible outside it.
- Variable types must stay consistent across branches and loops.
- Composite Python types (list/dict/tuple) are compile-time only.

## Kernel development workflow

1) **Reference implementation**
   - Create a correct PyTorch reference in `forge_cute_py/ref/<op>.py`.
   - Keep it simple and obvious; tests will compare against it.

2) **Kernel implementation**
   - Implement the CuTe DSL kernel in `forge_cute_py/kernels/<op>.py`.
   - Follow the existing kernel layout and launch patterns.
   - Cache compiled kernels by configuration (dtype, tile size, etc.).

3) **Op wrapper**
   - Add a Python entry point in `forge_cute_py/ops/<op>.py`.
   - Validate inputs and shapes.
   - Use `@torch.library.custom_op` decorator for PyTorch integration.
   - For ops requiring gradients, implement `torch.autograd.Function` with
     separate forward and backward custom ops.

4) **Tests**
   - Add tests under `tests/` comparing kernel output to reference.
   - Prefer parametrized tests for dtype, shape, and tile size.

5) **Benchmarks**
   - Add standalone benchmarks in `bench/benchmark_<op>.py`.
   - Add cases to `bench/suites.yaml` so `bench/run.py` can pick them up.

## Testing and correctness

```bash
# Run all tests
uv run pytest -q

# Run specific test file
uv run pytest tests/test_copy_transpose.py

# Run specific test function
uv run pytest tests/test_copy_transpose.py::test_copy_transpose_correctness

# Filter tests by name pattern with -k (matches test function names)
uv run pytest tests/test_softmax_online.py -k correctness
uv run pytest -k "not correctness and not compile"
```

Correctness is the primary gate. Use explicit tolerances and document any
non-zero tolerance in tests.

## Benchmarking

```bash
uv run python bench/run.py --suite smoke
uv run python bench/benchmark_copy_transpose.py --tile-size 16
uv run python bench/benchmark_reduce.py
uv run python bench/benchmark_online_softmax.py --backend ref
uv run python bench/benchmark_online_softmax.py --backend kernel
modal run bench/modal_bench.py --suite smoke --out results.json
modal run bench/modal_bench.py --suite smoke --op reduce_sum --out results.json
```

`softmax_online` backend selection is controlled by:
- Python API: `set_softmax_online_backend("ref")` / `set_softmax_online_backend("kernel")`
- Benchmark CLI: `bench/benchmark_online_softmax.py --backend {ref,kernel}`

Current `softmax_online` constraints:
- Input is 2D CUDA tensor
- Dtype in `float16`, `bfloat16`, `float32`
- `dim=-1` only

> **Warning:** Modal benchmarks incur GPU costs. Review `bench/modal_bench.py`
> and verify timeout/GPU settings before running. Start with `--suite smoke`
> to validate your setup. You are responsible for any credits consumed.

Modal benchmarks run on B200 GPUs using CUDA 13.1 and PyTorch 2.9.1.

Modal GPU types and CLI usage:
https://modal.com/docs/guide/gpu
https://modal.com/docs/reference/cli/run

Add new benchmark cases to `bench/suites.yaml` and keep outputs reproducible.

## Profiling

```bash
./scripts/profile.sh ncu -- uv run python bench/benchmark_copy_transpose.py
./scripts/profile.sh nsys -- uv run pytest tests/test_copy_transpose.py
./scripts/profile.sh sanitizer -- uv run python -m forge_cute_py.env_check
```

Nsight Compute with `--set full` can be slow; use lighter sets if needed
(`--set=launchstats`, `--section=SpeedOfLight`, etc.).

### Metric extraction

Extract curated metrics from NCU reports using the `--extract` flag or standalone:

```bash
# Automatic extraction after profiling
./scripts/profile.sh ncu --extract -- uv run python bench/benchmark_copy_transpose.py

# Standalone extraction from existing report
./scripts/ncu_extract.py profiles/ncu_*.ncu-rep

# JSON output
./scripts/ncu_extract.py profiles/ncu_*.ncu-rep --json

# Filter by kernel name
./scripts/ncu_extract.py profiles/ncu_*.ncu-rep --kernel "CopyTranspose"
```

Output includes three metric categories:
- **GPU Throughput**: Memory and Compute (SM) utilization percentages
- **Pipe Utilization**: Tensor Core, FMA, ALU activity
- **Warp Stalls**: Long scoreboard, barrier, memory throttle stalls

### Profiling tips

**For NCU profiling**, use minimal iterations since you're analyzing kernel behavior, not
timing variance. The L2 cache flush adds extra kernel launches per iteration.

```bash
# Fast profiling (recommended)
./scripts/profile.sh ncu --extract -- uv run python bench/benchmark_copy_transpose.py --warmup 2 --iterations 1

# Profile only your kernel (skip reference kernels)
./scripts/profile.sh ncu --kernel-name "copy_transpose" --extract -- uv run python bench/benchmark_copy_transpose.py
```

| Goal | Warmup | Iterations | Notes |
|------|--------|------------|-------|
| Timing benchmarks | 10 | 100 | Default, good for stable p50/p90 |
| Quick profiling | 2 | 1-3 | Sufficient for kernel analysis |
| Deep profiling (`--set full`) | 1 | 1 | Full metrics are slow |

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```

If you are using `quack/` as a reference checkout, exclude it from linting
or run lint from the repo root and keep `quack/` untracked.

## Definition of done

- Reference implementation exists or equivalence is documented.
- Tests pass with documented tolerances.
- Bench coverage added when performance is relevant.
- Profiling notes captured when kernel behavior changes.
