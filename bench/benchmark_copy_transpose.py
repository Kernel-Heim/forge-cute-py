import argparse

import torch

from forge_cute_py.util.bench import do_bench, estimate_bandwidth, summarize_times


def main():
    parser = argparse.ArgumentParser(description="Benchmark copy/transpose")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compile-ref", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmarking")
    if not hasattr(torch.ops, "forge_cute_py") or not hasattr(
        torch.ops.forge_cute_py, "copy_transpose"
    ):
        raise RuntimeError("torch.ops.forge_cute_py.copy_transpose not registered")

    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.m, args.n, device="cuda", dtype=dtype)
    op = torch.ops.forge_cute_py.copy_transpose

    def fn():
        return op(x, args.tile_size)

    times = do_bench(fn, warmup=args.warmup, rep=args.iterations)
    stats = summarize_times(times)
    bytes_moved = 2 * x.numel() * x.element_size()
    bw = estimate_bandwidth(bytes_moved, stats["p50_ms"])
    print(f"copy_transpose p50: {stats['p50_ms']:.4f} ms, BW: {bw:.2f} GB/s")

    ref = lambda: x.transpose(-2, -1).contiguous()
    if args.compile_ref and hasattr(torch, "compile"):
        ref = torch.compile(ref, fullgraph=True)
    ref_times = do_bench(ref, warmup=args.warmup, rep=args.iterations)
    ref_stats = summarize_times(ref_times)
    ref_bw = estimate_bandwidth(bytes_moved, ref_stats["p50_ms"])
    print(f"reference p50: {ref_stats['p50_ms']:.4f} ms, BW: {ref_bw:.2f} GB/s")


if __name__ == "__main__":
    main()
