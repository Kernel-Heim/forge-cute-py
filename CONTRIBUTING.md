# Contributing to forge-cute-py

*This is an active document, expect changes.*

Thanks for your interest in `forge-cute-py`. This repository is currently in an early, fast-moving phase while the KernelHeim v0 harness is being stabilized.

## Current contribution policy (important)

We are **not** accepting unsolicited pull requests at this time.

To avoid wasted work and long review cycles during rapid iteration, contributions are coordinated through maintainers. Unsolicited PRs may be closed without detailed review (you are welcome to keep your work in a fork).

What we *do* welcome right now:
- Bug reports (with reproduction details)
- Design feedback (on issues/discussions)
- Benchmark results, profiler traces, and hardware notes
- Requests for features or missing coverage

Current LLM policy (augmented from [nanochat](https://github.com/karpathy/nanochat/tree/be56d29b87bc51f60c527062389ccd6a14cd0e89?tab=readme-ov-file#contributing)): disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand. The current aim of the repo is to be educational. It is now simpler than ever to submit a PR with LLM contributions, so please avoid adding more work on maintainers.

## How to propose a change

1. **Open an Issue** describing what you want to change:
   - Motivation / problem statement
   - Proposed approach
   - Affected ops
   - How you will test correctness
   - Any expected perf impact and how you’ll measure it

2. (TBA) Wait for a maintainer to label the issue **`approved-for-work`** (or similar)
   - Only then should implementation start.
   - If you’re unsure, start a **Discussion** first.

3. (TBA) Once approved, you may:
   - Ask to be assigned, or
   - Share a fork/branch link for early feedback

## If you are an approved contributor

*We will update this section as we stabilize the contribution process.*

### Pre-commit hooks
Please install hooks with `pre-commit` to ensure code quality checks run locally:

```bash
uv run pre-commit install
```

To run checks manually on all files (`-a` is equivalent to `--all-files`):

```bash
uv run pre-commit run -a
```

### Development expectations
- Keep PRs small and scoped to a single issue.
- Include:
  - Correctness tests (PyTorch reference-gated)
  - Benchmark coverage where relevant
  - Notes on profiling changes if performance-sensitive

### Definition of done
A change that affects kernels or harness behavior should include:
- A reference implementation (or verified equivalence to an existing reference)
- Deterministic and reproducible tests (tolerances documented)
- Benchmark entry (if it changes performance or adds a kernel/variant)
- Clear failure modes and input validation where applicable

### PR hygiene
- Link the approved issue in the PR description: `Closes #123` or `Refs #123`
- Include:
  - What changed
  - Why it changed
  - How it was tested (commands + environment)
  - Bench results (before/after) if relevant


## Code of Conduct

Be respectful and constructive. Harassment and abuse are not tolerated.

## Security

If you believe you’ve found a security issue, do not open a public issue.
Instead, contact the maintainers privately (see repository contacts).
