`forge-cute-py` is a project for developing and evaluating **CuTe DSL** kernels in Python.

As initially planned, it provides a workflow to **run kernels, validate correctness against PyTorch references, benchmark performance, and profile**.

## Current scope (v0.1)

Target kernels aligned to KernelHeim **Weeks 0–2**:

- **Week 0:** tiled **copy / transpose**
- **Week 1:** **reductions (sum)** with multiple implementations (e.g., naive -> improved -> shuffle)
- **Week 2:** **single-pass online softmax**

Not currently in scope for v0: FlashAttention kernels (FA1+), decode/KV-cache, FP8, distributed/NCCL, C++ extension builds.

---

## Requirements

- Linux + NVIDIA GPU (CUDA-capable)
- Python (managed via `uv`)
- PyTorch installed with CUDA support
- Recommended tooling for profiling:
  - Nsight Compute (`ncu`)
  - Nsight Systems (`nsys`)
  - compute-sanitizer

---

## Install (uv)

```bash
uv sync
```

If you need an editable/dev install, use your normal `uv` workflow (project is expected to be runnable via `uv run ...`).

---

## Sanity check

```bash
uv run python -m forge_cute_py.env_check
```

This should validate CUDA/PyTorch visibility and basic runtime assumptions.

---

## Correctness tests (PyTorch reference-gated)

```bash
uv run pytest -q
```

Correctness is the primary gate for changes: kernels must match reference behavior within defined tolerances.

---

## User guide (quickstart)

Run a single op in Python:

```bash
uv run python - <<'PY'
import torch
from forge_cute_py.ops import copy_transpose

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = copy_transpose(x, tile_size=16)
print(y.shape)  # torch.Size([1024, 1024])
PY
```

Ops are also accessible via `torch.ops.forge_cute_py._op_name()` (note the
underscore prefix) for direct custom op access.

Run a smoke benchmark suite (JSON output):

```bash
uv run python bench/run.py --suite smoke --out results.json
```

Run a standalone benchmark:

```bash
uv run python bench/benchmark_copy_transpose.py --tile-size 16
uv run python bench/benchmark_online_softmax.py --backend ref
uv run python bench/benchmark_online_softmax.py --backend kernel
```

Profile a kernel with helper script:

```bash
./scripts/profile.sh ncu -- uv run python bench/benchmark_copy_transpose.py
./scripts/profile.sh nsys -- uv run pytest tests/test_copy_transpose.py
```

Or use profiling tools directly:

```bash
ncu --set full -o profiles/copy_transpose uv run python bench/benchmark_copy_transpose.py
```

---

## Kernel status (v0.1)

| Op | Status | Variants | Notes |
| --- | --- | --- | --- |
| copy_transpose | Implemented | tile_size=16/32 | CuTe DSL kernel with tiled shared memory |
| reduce | Implemented (sum only) | - | CuTe kernel with benchmark coverage |
| reduce_sum | Stub (ref) | - | Reference path with benchmark coverage |
| softmax_online | Implemented (registry) | ref, kernel | Default backend is `ref`; `kernel` is available for contributor benchmarking and remains non-production |

---

## Package layout (high level)

* `forge_cute_py/ops/`
  Python-facing op wrappers, input validation, optional `torch.library` registration.
* `forge_cute_py/kernels/`
  CuTe DSL kernel implementations (organized by week/op).
* `forge_cute_py/ref/`
  Reference implementations (PyTorch) used by tests and validation.
* `tests/`
  Environment checks + correctness tests (pytest).
* `bench/`
  Benchmark CLI, suites, and JSON reporting.
* `scripts/`
  Profiling and sanitizer runners (`ncu`, `nsys`, compute-sanitizer).

---

## Quick Reference

### Setup and Validation
```bash
uv sync                                    # Install dependencies
uv run python -m forge_cute_py.env_check  # Verify CUDA/PyTorch setup
```

### Testing
```bash
uv run pytest -q                                      # Run all tests
uv run pytest tests/test_copy_transpose.py            # Run specific test file
uv run pytest -k correctness                          # Filter by test name pattern
uv run pre-commit run --all-files                     # Run linting/formatting
```

### Benchmarking
```bash
uv run python bench/run.py --suite smoke              # Run benchmark suite
uv run python bench/run.py --suite smoke --out out.json  # Save results
uv run python bench/benchmark_copy_transpose.py       # Standalone benchmark
uv run python bench/benchmark_reduce.py               # Standalone benchmark
uv run python bench/benchmark_online_softmax.py --backend ref     # softmax fwd+bwd on reference backend
uv run python bench/benchmark_online_softmax.py --backend kernel  # softmax fwd+bwd on CuTe kernel backend
modal run bench/modal_bench.py --suite smoke --out results.json  # Remote run on B200 via Modal
```

`softmax_online` backend selection is controlled via Python API or benchmark CLI:
- Python API: `set_softmax_online_backend("ref")` or `set_softmax_online_backend("kernel")`
- Benchmark CLI: `bench/benchmark_online_softmax.py --backend {ref,kernel}`

Current softmax contract:
- supported tensors: 2D CUDA
- supported dtypes: `float16`, `bfloat16`, `float32`
- supported dim: `dim=-1` (row-wise only)

### Modal setup (remote benchmarks)
```bash
uv pip install modal
modal token new
modal run bench/modal_bench.py --suite smoke --out results.json
modal run bench/modal_bench.py --suite smoke --op reduce_sum --out results.json
```

Modal benchmarks run on B200 GPUs using CUDA 13.1 and PyTorch 2.9.1.
See https://modal.com/docs/guide/gpu for more info.

### Profiling
```bash
./scripts/profile.sh ncu -- uv run python bench/benchmark_copy_transpose.py  # Nsight Compute
./scripts/profile.sh nsys -- uv run pytest tests/test_copy_transpose.py      # Nsight Systems
./scripts/profile.sh sanitizer -- uv run python -m forge_cute_py.env_check   # Memory check
```

---

## Contributing

**Note:** We are not accepting unsolicited pull requests during v0 stabilization. Please open an issue first and wait for maintainer approval before starting work.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

For kernel development workflow and architecture details, see:
- [DEVELOPMENT.md](DEVELOPMENT.md) - Kernel development guide

---

## Roadmap (v0.1 completion)

* [x] Week 0 copy/transpose: end-to-end correctness + benchmark + profile scripts
* [ ] Week 1 reductions: multiple variants, correctness + benchmark coverage
* [ ] Week 2 online softmax: correctness + benchmark coverage + profiling notes
* [ ] CI: run correctness on supported GPU runners; optional perf smoke checks

See [ROADMAP.md](ROADMAP.md) for detailed breakdown and progress tracking.
