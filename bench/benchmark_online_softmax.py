"""Benchmark softmax_online op against torch.softmax and torch.compile(torch.softmax)."""

import argparse

import torch

from forge_cute_py.ops.softmax_online import softmax_fwd, softmax_bwd
from forge_cute_py.util.bench import do_bench, estimate_bandwidth, summarize_times

SHORT_M = [128, 512, 2048, 8192]
SHORT_N = [1024, 2048, 4096, 8192]

LONG_M = [64, 128, 256]
LONG_N = [16384, 32768, 65536, 131072]

DEFAULT_DTYPES = ["float16", "bfloat16", "float32"]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Benchmark softmax_online op")
    parser.add_argument(
        "--long", action="store_true", help="Use long-N benchmark suite (small M, large N)"
    )
    parser.add_argument("--m-sizes", type=parse_int_list, default=None)
    parser.add_argument("--n-sizes", type=parse_int_list, default=None)
    parser.add_argument("--dtypes", type=parse_str_list, default=DEFAULT_DTYPES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.m_sizes is None:
        args.m_sizes = LONG_M if args.long else SHORT_M
    if args.n_sizes is None:
        args.n_sizes = LONG_N if args.long else SHORT_N

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmarking")

    gpu_name = torch.cuda.get_device_name(0)
    suite = "long" if args.long else "short"
    print(f"softmax_online benchmarks [{suite}] ({gpu_name})")
    print()

    header = (
        f"{'M':>6}  {'N':>6}  {'Dtype':<10} {'Op':<18} {'Pass':<5} "
        f"{'p50 (ms)':>10} {'BW (GB/s)':>10} {'vs torch':>10}"
    )
    print(header)
    print("-" * len(header))

    for m in args.m_sizes:
        for n in args.n_sizes:
            for dtype_str in args.dtypes:
                dtype = getattr(torch, dtype_str)
                x = torch.randn(m, n, device="cuda", dtype=dtype)
                assert n % 32 == 0, f"Inner dimension N must be a multiple of 32, got {n}"
                elem = x.element_size()

                # --- Forward bandwidth: read input + write output ---
                fwd_bytes = 2 * m * n * elem

                # --- torch.softmax fwd baseline ---
                torch_fn = lambda: torch.softmax(x, dim=-1)
                torch_times = do_bench(torch_fn, warmup=args.warmup, rep=args.iterations)
                torch_stats = summarize_times(torch_times)
                torch_fwd_p50 = torch_stats["p50_ms"]
                torch_fwd_bw = estimate_bandwidth(fwd_bytes, torch_fwd_p50)
                print(
                    f"{m:>6}  {n:>6}  {dtype_str:<10} {'torch.softmax':<18} {'fwd':<5} "
                    f"{torch_fwd_p50:>10.4f} {torch_fwd_bw:>10.2f} {1.0:>10.2f}x"
                )

                # --- torch.compile fwd ---
                try:
                    compiled_ref = torch.compile(lambda t: torch.softmax(t, dim=-1))
                    compiled_ref(x)
                    fn = lambda: compiled_ref(x)
                    compiled_times = do_bench(fn, warmup=args.warmup, rep=args.iterations)
                    compiled_stats = summarize_times(compiled_times)
                    compiled_p50 = compiled_stats["p50_ms"]
                    compiled_bw = estimate_bandwidth(fwd_bytes, compiled_p50)
                    ratio = compiled_p50 / torch_fwd_p50 if torch_fwd_p50 > 0 else float("inf")
                    print(
                        f"{m:>6}  {n:>6}  {dtype_str:<10} {'torch.compile':<18} {'fwd':<5} "
                        f"{compiled_p50:>10.4f} {compiled_bw:>10.2f} {ratio:>10.2f}x"
                    )
                except Exception as e:
                    print(
                        f"{m:>6}  {n:>6}  {dtype_str:<10} {'torch.compile':<18} {'fwd':<5} "
                        f"{'ERROR':>10} {'':>10} {'':>10}  {e}"
                    )

                # --- softmax_online fwd ---
                try:
                    softmax_fwd(x, dim=-1)
                    fn = lambda: softmax_fwd(x, dim=-1)
                    times = do_bench(fn, warmup=args.warmup, rep=args.iterations)
                    stats = summarize_times(times)
                    p50 = stats["p50_ms"]
                    bw = estimate_bandwidth(fwd_bytes, p50)
                    ratio = p50 / torch_fwd_p50 if torch_fwd_p50 > 0 else float("inf")
                    print(
                        f"{m:>6}  {n:>6}  {dtype_str:<10} {'softmax_online':<18} {'fwd':<5} "
                        f"{p50:>10.4f} {bw:>10.2f} {ratio:>10.2f}x"
                    )
                except Exception as e:
                    print(
                        f"{m:>6}  {n:>6}  {dtype_str:<10} {'softmax_online':<18} {'fwd':<5} "
                        f"{'ERROR':>10} {'':>10} {'':>10}  {e}"
                    )

                # --- Backward pass benchmarks ---
                # Pre-compute softmax output y and fake upstream gradient dy
                y = torch.softmax(x, dim=-1)
                dy = torch.randn_like(y)

                # Backward bandwidth: read dy + read y + write dx = 3 * M * N * elem
                bwd_bytes = 3 * m * n * elem

                # --- torch backward baseline ---
                torch_bwd_fn = lambda: torch._softmax_backward_data(dy, y, -1, x.dtype)
                torch_bwd_times = do_bench(torch_bwd_fn, warmup=args.warmup, rep=args.iterations)
                torch_bwd_stats = summarize_times(torch_bwd_times)
                torch_bwd_p50 = torch_bwd_stats["p50_ms"]
                torch_bwd_bw = estimate_bandwidth(bwd_bytes, torch_bwd_p50)
                print(
                    f"{m:>6}  {n:>6}  {dtype_str:<10} {'torch.softmax':<18} {'bwd':<5} "
                    f"{torch_bwd_p50:>10.4f} {torch_bwd_bw:>10.2f} {1.0:>10.2f}x"
                )

                # --- softmax_online bwd ---
                try:
                    y_ours = softmax_fwd(x, dim=-1)
                    softmax_bwd(dy, y_ours, dim=-1)
                    fn = lambda: softmax_bwd(dy, y_ours, dim=-1)
                    times = do_bench(fn, warmup=args.warmup, rep=args.iterations)
                    stats = summarize_times(times)
                    p50 = stats["p50_ms"]
                    bw = estimate_bandwidth(bwd_bytes, p50)
                    ratio = p50 / torch_bwd_p50 if torch_bwd_p50 > 0 else float("inf")
                    print(
                        f"{m:>6}  {n:>6}  {dtype_str:<10} {'softmax_online':<18} {'bwd':<5} "
                        f"{p50:>10.4f} {bw:>10.2f} {ratio:>10.2f}x"
                    )
                except Exception as e:
                    print(
                        f"{m:>6}  {n:>6}  {dtype_str:<10} {'softmax_online':<18} {'bwd':<5} "
                        f"{'ERROR':>10} {'':>10} {'':>10}  {e}"
                    )

                print()


if __name__ == "__main__":
    main()
