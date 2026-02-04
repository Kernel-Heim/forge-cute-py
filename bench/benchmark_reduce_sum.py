"""Benchmark reduce_sum kernel variants against torch.sum."""

import argparse

import torch

from forge_cute_py.ops.reduce_sum import reduce_sum
from forge_cute_py.util.bench import do_bench, estimate_bandwidth, summarize_times

DEFAULT_SIZES = [1024, 2048, 4096, 8192]
DEFAULT_DTYPES = ["float16", "bfloat16", "float32"]
DEFAULT_DIMS = [1]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Benchmark reduce_sum variants")
    parser.add_argument("--sizes", type=parse_int_list, default=DEFAULT_SIZES)
    parser.add_argument("--dtypes", type=parse_str_list, default=DEFAULT_DTYPES)
    parser.add_argument("--dims", type=parse_int_list, default=DEFAULT_DIMS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmarking")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"reduce_sum benchmarks ({gpu_name})")
    print()

    header = f"{'Size':>6}  {'Dtype':<10} {'Dim':>3}  {'Op':<12} {'p50 (ms)':>10} {'BW (GB/s)':>10} {'vs torch':>10}"
    print(header)
    print("-" * len(header))

    for size in args.sizes:
        for dtype_str in args.dtypes:
            dtype = getattr(torch, dtype_str)
            for dim in args.dims:
                x = torch.randn(size, size, device="cuda", dtype=dtype)

                # Output shape determines output bytes
                if dim == 1:
                    out_numel = size
                else:
                    out_numel = size
                input_bytes = x.numel() * x.element_size()
                output_bytes = out_numel * x.element_size()
                total_bytes = input_bytes + output_bytes

                # Benchmark torch.sum as reference
                torch_fn = lambda: torch.sum(x, dim=dim)
                torch_times = do_bench(torch_fn, warmup=args.warmup, rep=args.iterations)
                torch_stats = summarize_times(torch_times)
                torch_p50 = torch_stats["p50_ms"]
                torch_bw = estimate_bandwidth(total_bytes, torch_p50)
                print(
                    f"{size:>6}  {dtype_str:<10} {dim:>3}  {'torch.sum':<14} "
                    f"{torch_p50:>10.4f} {torch_bw:>10.2f} {1.0:>10.2f}x"
                )

                if dim not in (-1, 1):
                    raise ValueError(f"Unsupported dim={dim}; reduce_sum supports only -1 or 1")

                # Warm the JIT cache
                try:
                    reduce_sum(x, dim=dim)
                except Exception as e:
                    print(
                        f"{size:>6}  {dtype_str:<10} {dim:>3}  {'reduce_sum':<12} "
                        f"{'ERROR':>10} {'':>10} {'':>10}  {e}"
                    )
                    print()
                    continue

                fn = lambda d=dim: reduce_sum(x, dim=d)
                times = do_bench(fn, warmup=args.warmup, rep=args.iterations)
                stats = summarize_times(times)
                p50 = stats["p50_ms"]
                bw = estimate_bandwidth(total_bytes, p50)
                ratio = p50 / torch_p50 if torch_p50 > 0 else float("inf")
                print(
                    f"{size:>6}  {dtype_str:<10} {dim:>3}  {'reduce_sum':<12} "
                    f"{p50:>10.4f} {bw:>10.2f} {ratio:>10.2f}x"
                )

                print()


if __name__ == "__main__":
    main()
