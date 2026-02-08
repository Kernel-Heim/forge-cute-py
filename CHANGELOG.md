# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial harness infrastructure for KernelHeim v0.1
- Three-layer architecture (ops/kernels/ref) for kernel implementations (#5)
- Week 0: `copy_transpose` kernel with CuTe DSL implementation
- Week 1-2: `reduce_sum` and `softmax_online` ops with reference implementations
- Modern PyTorch custom op pattern with `@torch.library.custom_op` decorator (#12)
- Autograd support for `softmax_online` via `torch.autograd.Function` (#12)
- GitHub Issue Forms with structured templates for bugs, features, docs, and performance (#13)
- Kernel compilation caching in ops layer with symbolic compilation
- Pre-commit hooks configuration with ruff linting and formatting
- GitHub Actions CI workflow for automated code quality checks (#10)
- Benchmark framework with YAML-driven suite system (`bench/run.py`)
- L2 cache clearing and gradient flush in `do_bench` utility
- Reference implementations for all Week 0-2 operations
- Comprehensive test suite with PyTorch reference validation
- Environment check utility (`forge_cute_py.env_check`)
- Documentation: README, DEVELOPMENT.md (#11), CONTRIBUTING.md with contribution policy (#7)
- Pre-commit hooks installation instructions in CONTRIBUTING.md (#7)
- Support for float16, bfloat16, and float32 dtypes
- `FORGE_SOFTMAX_IMPL` softmax mode selection (`auto`, `ref`, `kernel`)

### Changed
- Modal benchmarks now use strict GPU matching (`!` suffix) for consistent hardware
- **BREAKING:** Renamed `tile` parameter to `tile_size` in `copy_transpose` API (#5)
- Modernized all ops to use two-function pattern: internal `_op_name` (mutates output) and public `op_name` (allocates + returns) (#12)
- Updated `env_check` to use public API instead of `torch.ops` for testing
- Replaced markdown GitHub issue templates with structured YAML forms (#13)
- Improved test structure with parametrized fixtures for dtypes, shapes, and tile sizes
- Enhanced error messages with device information for better debugging
- Set tolerance to 0 (exact comparison) for transpose correctness tests
- Updated documentation to reflect current three-layer architecture and modern PyTorch patterns
- Added `--impl` to `bench/benchmark_online_softmax.py` and updated benchmark handling for strict kernel mode

### Infrastructure
- Package scaffolding with `pyproject.toml` and proper Python 3.13+ support
- PyTorch ops registration via `@torch.library.custom_op` with autograd support
- Benchmark utilities with timing statistics (p50/p90/p99) and bandwidth estimation
- Test configuration with pytest
- Ruff configuration for linting and formatting (line-length=100)
- Scripts for architecture detection and profiling setup

## [0.1.0] - Unreleased

Initial development release targeting KernelHeim Weeks 0-2.

### Roadmap
- [x] Week 0: Harness infrastructure with copy/transpose kernel
- [ ] Week 1: Reduction operations (sum) with naive/improved/shuffle variants
- [ ] Week 2: Single-pass online softmax
- [ ] CI: GPU runners for correctness validation

[Unreleased]: https://github.com/Kernel-Heim/forge-cute-py/compare/main...init-harness
