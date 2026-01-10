import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from forge_cute_py.util.bench import do_bench, estimate_bandwidth, summarize_times


def _env_metadata():
    metadata = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metadata.update(
            {
                "gpu_name": props.name,
                "sm": f"{props.major}{props.minor}",
            }
        )
        driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
        if driver_version is not None:
            metadata["cuda_driver_version"] = driver_version()
    return metadata


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_str}")
    return mapping[dtype_str]


def _ops_namespace():
    if not hasattr(torch.ops, "forge_cute_py"):
        return None
    return torch.ops.forge_cute_py


def _estimate_bytes(op: str, shape, dtype: torch.dtype, dim=None):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = 1
    for dim_size in shape:
        numel *= dim_size
    if op == "reduce_sum" and dim is not None:
        out_numel = numel // shape[dim]
        return (numel + out_numel) * elem_size
    return 2 * numel * elem_size


def _bench_case(case, warmup: int, iterations: int):
    ops = _ops_namespace()
    if ops is None:
        return {"status": "skipped", "reason": "forge_cute_py ops not registered"}

    op_name = case["op"]
    dtype = _dtype_from_str(case.get("dtype", "float16"))
    shape = case.get("shape", [1024, 1024])

    if op_name == "copy_transpose":
        if not hasattr(ops, "copy_transpose"):
            return {"status": "skipped", "reason": "copy_transpose not registered"}
        tile_size = case.get("tile_size", 16)
        x = torch.randn(*shape, device="cuda", dtype=dtype)

        def fn():
            return ops.copy_transpose(x, tile_size)

        times = do_bench(fn, warmup=warmup, rep=iterations)
        stats = summarize_times(times)
        bytes_moved = _estimate_bytes(op_name, shape, dtype)
        bw = estimate_bandwidth(bytes_moved, stats["p50_ms"])
        return {
            "status": "ok",
            "op": op_name,
            "shape": shape,
            "dtype": str(dtype).replace("torch.", ""),
            "tile_size": tile_size,
            "times_ms": stats,
            "bandwidth_gbps": bw,
        }

    if op_name == "reduce_sum":
        if not hasattr(ops, "reduce_sum"):
            return {"status": "skipped", "reason": "reduce_sum not registered"}
        dim = case.get("dim", -1)
        variant = case.get("variant", "shfl")
        x = torch.randn(*shape, device="cuda", dtype=dtype)

        def fn():
            return ops.reduce_sum(x, dim, variant)

        try:
            times = do_bench(fn, warmup=warmup, rep=iterations)
        except NotImplementedError:
            return {"status": "skipped", "reason": f"variant {variant} not implemented"}
        stats = summarize_times(times)
        bytes_moved = _estimate_bytes(op_name, shape, dtype, dim=dim)
        bw = estimate_bandwidth(bytes_moved, stats["p50_ms"])
        return {
            "status": "ok",
            "op": op_name,
            "shape": shape,
            "dtype": str(dtype).replace("torch.", ""),
            "dim": dim,
            "variant": variant,
            "times_ms": stats,
            "bandwidth_gbps": bw,
        }

    if op_name == "softmax_online":
        if not hasattr(ops, "softmax_online"):
            return {"status": "skipped", "reason": "softmax_online not registered"}
        dim = case.get("dim", -1)
        x = torch.randn(*shape, device="cuda", dtype=dtype)

        def fn():
            return ops.softmax_online(x, dim)

        times = do_bench(fn, warmup=warmup, rep=iterations)
        stats = summarize_times(times)
        bytes_moved = _estimate_bytes(op_name, shape, dtype, dim=dim)
        bw = estimate_bandwidth(bytes_moved, stats["p50_ms"])
        return {
            "status": "ok",
            "op": op_name,
            "shape": shape,
            "dtype": str(dtype).replace("torch.", ""),
            "dim": dim,
            "times_ms": stats,
            "bandwidth_gbps": bw,
        }

    return {"status": "skipped", "reason": f"unknown op {op_name}"}


def main():
    parser = argparse.ArgumentParser(description="forge-cute-py benchmark runner")
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--out", default=None)
    parser.add_argument("--suites", default=str(Path(__file__).parent / "suites.yaml"))
    args = parser.parse_args()

    suites_path = Path(args.suites)
    payload = yaml.safe_load(suites_path.read_text(encoding="utf-8"))
    suites = payload.get("suites", {})
    if args.suite not in suites:
        raise ValueError(f"Unknown suite {args.suite}")
    suite = suites[args.suite]
    warmup = int(suite.get("warmup", 10))
    iterations = int(suite.get("iterations", 50))
    cases = suite.get("cases", [])

    results = {
        "suite": args.suite,
        "warmup": warmup,
        "iterations": iterations,
        "env": _env_metadata(),
        "cases": [],
    }

    for case in cases:
        results["cases"].append(_bench_case(case, warmup, iterations))

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
