# Roadmap

This document tracks detailed progress toward forge-cute-py v0.1 completion.

## Overview

Target: Harness infrastructure + Week 0-2 kernel implementations aligned to KernelHeim curriculum.

**Status**: Harness infrastructure complete. Week 0 kernel complete. Week 1-2 kernels pending.

---

## Week 0: Copy/Transpose

**Goal**: End-to-end correctness + benchmark + profile scripts

### Harness Infrastructure
- [x] Package scaffolding and build system
- [x] Three-layer architecture (ops/kernels/ref)
- [x] PyTorch ops registration via `torch.library`
- [x] Test infrastructure with pytest
- [x] Benchmark framework with suite system
- [x] Profiling documentation and helper scripts
- [x] CI/CD with ruff linting and formatting
- [x] Pre-commit hooks configuration
- [x] Documentation (README, CONTRIBUTING, DEVELOPMENT)

### Copy/Transpose Kernel
- [x] CuTe DSL kernel implementation (`forge_cute_py/kernels/copy_transpose.py`)
- [x] Ops layer with compilation caching (`forge_cute_py/ops/copy_transpose.py`)
- [x] PyTorch reference implementation (`forge_cute_py/ref/copy_transpose.py`)
- [x] Correctness tests with exact tolerance (atol=0, rtol=0)
- [x] Support for float16, bfloat16, float32 dtypes
- [x] Support for tile_size=16 and tile_size=32 variants
- [x] Benchmark integration in `bench/run.py`
- [x] Profiling examples in README

**Status**: ✅ Complete

---

## Week 1: Reductions (Sum)

**Goal**: Multiple variants (naive → improved → shuffle) with correctness + benchmark coverage

### Infrastructure
- [x] Test infrastructure (using PyTorch reference)
- [x] Reference implementation (`forge_cute_py/ref/reduce_sum.py`)
- [x] Ops registration and API design
- [x] Benchmark integration

### Kernel Implementations
- [ ] **Naive variant**: Simple reduction without optimizations
  - [ ] CuTe DSL kernel implementation
  - [ ] Correctness tests vs PyTorch reference
  - [ ] Benchmark baseline

- [ ] **Improved variant**: Optimized reduction with shared memory
  - [ ] CuTe DSL kernel implementation
  - [ ] Correctness tests vs PyTorch reference
  - [ ] Benchmark comparison vs naive

- [ ] **Shuffle variant**: Warp-level shuffle reduction
  - [ ] CuTe DSL kernel implementation
  - [ ] Correctness tests vs PyTorch reference
  - [ ] Benchmark comparison vs improved

- [ ] **Documentation**: Profiling notes and performance analysis

**Status**: ⏳ Test infrastructure ready, kernels pending

---

## Week 2: Online Softmax

**Goal**: Single-pass online softmax with correctness + benchmark coverage + profiling notes

### Infrastructure
- [x] Test infrastructure (using PyTorch reference)
- [x] Reference implementation (`forge_cute_py/ref/softmax_online.py`)
- [x] Ops registration and API design
- [x] Benchmark integration

### Kernel Implementation
- [ ] **Single-pass online softmax kernel**
  - [ ] CuTe DSL kernel implementation
  - [ ] Numerical stability handling (max subtraction)
  - [ ] Correctness tests vs PyTorch reference
  - [ ] Support for float16, bfloat16, float32
  - [ ] Benchmark integration

- [ ] **Documentation**:
  - [ ] Profiling notes
  - [ ] Performance characteristics
  - [ ] Comparison with PyTorch softmax

**Status**: ⏳ Test infrastructure ready, kernel pending

---

## CI/CD Infrastructure

### Local CI (Completed)
- [x] Ruff linting with GitHub Actions
- [x] Ruff formatting checks
- [x] Pre-commit hooks for local development
- [x] Manual workflow dispatch support

### GPU CI (Pending)
- [ ] Configure GPU runners for correctness tests
- [ ] Run full test suite on GPU CI
- [ ] Optional: Performance smoke checks
- [ ] Optional: Nightly profiling runs

**Status**: ⏳ Local CI complete, GPU CI pending

---

## Future Work (Post v0.1)

Not currently in scope but may be added later:

- FlashAttention kernels (FA1, FA2)
- Decode/KV-cache operations
- FP8 support
- Distributed operations (NCCL)
- C++ extension builds
- Additional optimization variants
- Multi-GPU support

---

## Issue Tracking

Active issues and milestones are tracked on GitHub:
- [Open Issues](https://github.com/Kernel-Heim/forge-cute-py/issues)
- [Milestones](https://github.com/Kernel-Heim/forge-cute-py/milestones)

For detailed change history, see [CHANGELOG.md](CHANGELOG.md).
