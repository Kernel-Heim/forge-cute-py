"""Benchmark reduce op against torch.sum and torch.compile(torch.sum)."""

import argparse

import torch

from forge_cute_py.ops import reduce
from forge_cute_py.util.bench import do_bench, estimate_bandwidth, summarize_times

DEFAULT_M = [128, 512, 2048]
DEFAULT_N = [256, 1024, 2048, 4096, 8192]
DEFAULT_DTYPES = ["float16", "bfloat16", "float32"]
DEFAULT_DIMS = [1]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Benchmark reduce op")
    parser.add_argument("--m-sizes", type=parse_int_list, default=DEFAULT_M)
    parser.add_argument("--n-sizes", type=parse_int_list, default=DEFAULT_N)
    parser.add_argument("--dtypes", type=parse_str_list, default=DEFAULT_DTYPES)
    parser.add_argument("--dims", type=parse_int_list, default=DEFAULT_DIMS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmarking")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"reduce benchmarks ({gpu_name})")
    print()

    header = (
        f"{'M':>6}  {'N':>6}  {'Dtype':<10} {'Dim':>3}  {'Op':<14} "
        f"{'p50 (ms)':>10} {'BW (GB/s)':>10} {'vs torch':>10}"
    )
    print(header)
    print("-" * len(header))

    for m in args.m_sizes:
        for n in args.n_sizes:
            for dtype_str in args.dtypes:
                dtype = getattr(torch, dtype_str)
                for dim in args.dims:
                    x = torch.randn(m, n, device="cuda", dtype=dtype)

                    input_bytes = x.numel() * x.element_size()
                    output_bytes = m * x.element_size()
                    total_bytes = input_bytes + output_bytes

                    torch_fn = lambda: torch.sum(x, dim=dim)
                    torch_times = do_bench(torch_fn, warmup=args.warmup, rep=args.iterations)
                    torch_stats = summarize_times(torch_times)
                    torch_p50 = torch_stats["p50_ms"]
                    torch_bw = estimate_bandwidth(total_bytes, torch_p50)
                    print(
                        f"{m:>6}  {n:>6}  {dtype_str:<10} {dim:>3}  {'torch.sum':<14} "
                        f"{torch_p50:>10.4f} {torch_bw:>10.2f} {1.0:>10.2f}x"
                    )

                    try:
                        compiled_ref = torch.compile(lambda t: torch.sum(t, dim=dim))
                        compiled_ref(x)
                        fn = lambda: compiled_ref(x)
                        compiled_times = do_bench(fn, warmup=args.warmup, rep=args.iterations)
                        compiled_stats = summarize_times(compiled_times)
                        compiled_p50 = compiled_stats["p50_ms"]
                        compiled_bw = estimate_bandwidth(total_bytes, compiled_p50)
                        ratio = compiled_p50 / torch_p50 if torch_p50 > 0 else float("inf")
                        print(
                            f"{m:>6}  {n:>6}  {dtype_str:<10} {dim:>3}  {'torch.compile':<14} "
                            f"{compiled_p50:>10.4f} {compiled_bw:>10.2f} {ratio:>10.2f}x"
                        )
                    except Exception as e:
                        print(
                            f"{m:>6}  {n:>6}  {dtype_str:<10} {dim:>3}  {'torch.compile':<14} "
                            f"{'ERROR':>10} {'':>10} {'':>10}  {e}"
                        )

                    try:
                        reduce(x, dim=dim, op="sum")
                        fn = lambda: reduce(x, dim=dim, op="sum")
                        times = do_bench(fn, warmup=args.warmup, rep=args.iterations)
                        stats = summarize_times(times)
                        p50 = stats["p50_ms"]
                        bw = estimate_bandwidth(total_bytes, p50)
                        ratio = p50 / torch_p50 if torch_p50 > 0 else float("inf")
                        print(
                            f"{m:>6}  {n:>6}  {dtype_str:<10} {dim:>3}  {'reduce':<14} "
                            f"{p50:>10.4f} {bw:>10.2f} {ratio:>10.2f}x"
                        )
                    except Exception as e:
                        print(
                            f"{m:>6}  {n:>6}  {dtype_str:<10} {dim:>3}  {'reduce':<14} "
                            f"{'ERROR':>10} {'':>10} {'':>10}  {e}"
                        )

                    print()


if __name__ == "__main__":
    main()
