`forge-cute-py` is a project for developing and evaluating **CuTe DSL** kernels in Python.

As initially planned, it should provides a workflow to **run kernels, validate correctness against PyTorch references, benchmark performance, and profile**.

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
````

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
import forge_cute_py

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = torch.ops.forge_cute_py.copy_transpose(x, 16)
print(y.shape)
PY
```

Run a smoke benchmark suite (JSON output):

```bash
uv run python bench/run.py --suite smoke --out results.json
```

Profile a kernel (Nsight Compute):

```bash
ncu --set full -o profiles/copy_transpose_profile \
  uv run python -m forge_cute_py.env_check
```

---

## Kernel status (v0.1)

| Op | Status | Variants | Notes |
| --- | --- | --- | --- |
| copy_transpose | Implemented | tile_size=16/32 | CuTe DSL kernel in `forge_cute_py/kernels/copy_transpose.py` |
| reduce_sum | Stub (ref) | naive/improved/shfl | CUDA path currently uses reference; kernel to be implemented |
| softmax_online | Stub (ref) | single-pass | CUDA path currently uses reference; kernel to be implemented |

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Roadmap (v0.1 completion)

* [ ] Week 0 copy/transpose: end-to-end correctness + benchmark + profile scripts
* [ ] Week 1 reductions: multiple variants, correctness + benchmark coverage
* [ ] Week 2 online softmax: correctness + benchmark coverage + profiling notes
* [ ] CI: run correctness on supported GPU runners; optional perf smoke checks
