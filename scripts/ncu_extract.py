#!/usr/bin/env python3
"""
NCU Metric Extraction Tool

Parse Nsight Compute .ncu-rep files and extract curated performance metrics.
Outputs GPU throughput, pipe utilization, and warp stall information.

Usage:
    ./scripts/ncu_extract.py profile.ncu-rep
    ./scripts/ncu_extract.py profile.ncu-rep --json
    ./scripts/ncu_extract.py profile.ncu-rep --kernel "CopyTranspose"
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path


# Metric configurations for different categories
# Each tuple: (display_name, metric_patterns, unit)
# Patterns are tried in order; first available metric is used
METRIC_CONFIGS = {
    "gpu_throughput": [
        ("Memory [%]", ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"], "%"),
        ("Compute (SM) [%]", ["sm__throughput.avg.pct_of_peak_sustained_elapsed"], "%"),
    ],
    "pipe_utilization": [
        (
            "Tensor Core",
            [
                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
            ],
            "%",
        ),
        ("FMA", ["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"], "%"),
        ("ALU", ["sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"], "%"),
        ("FP64", ["sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active"], "%"),
        ("Shared Memory", ["sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active"], "%"),
    ],
    "warp_stalls": [
        (
            "Stall Long Scoreboard",
            ["smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio"],
            "inst",
        ),
        ("Stall Wait", ["smsp__warps_issue_stalled_wait_per_issue_active.ratio"], "inst"),
        (
            "Stall Short Scoreboard",
            ["smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio"],
            "inst",
        ),
        ("Stall Barrier", ["smsp__warps_issue_stalled_barrier_per_issue_active.ratio"], "inst"),
        (
            "Stall Memory Throttle",
            [
                "smsp__warps_issue_stalled_membar_per_issue_active.ratio",
                "smsp__warps_issue_stalled_drain_per_issue_active.ratio",
            ],
            "inst",
        ),
        (
            "Selected",
            [
                "smsp__warps_issue_stalled_selected_per_issue_active.ratio",
                "smsp__issue_active.avg.per_cycle_active",
            ],
            "inst",
        ),
    ],
}


@dataclass
class KernelMetrics:
    """Metrics for a single kernel invocation."""

    kernel_name: str
    gpu_throughput: dict[str, float] = field(default_factory=dict)
    pipe_utilization: dict[str, float] = field(default_factory=dict)
    warp_stalls: dict[str, float] = field(default_factory=dict)


def check_ncu_available() -> bool:
    """Check if NCU is available in PATH."""
    try:
        subprocess.run(
            ["ncu", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_available_metrics(ncu_rep_path: Path) -> set[str]:
    """Query available metrics from the NCU report."""
    try:
        result = subprocess.run(
            ["ncu", "--import", str(ncu_rep_path), "--list-metrics"],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(line.strip() for line in result.stdout.splitlines() if line.strip())
    except subprocess.CalledProcessError:
        return set()


def extract_raw_metrics(ncu_rep_path: Path, kernel_filter: str | None = None) -> list[dict]:
    """
    Extract raw metrics from NCU report using CSV export.

    Returns list of dicts with kernel name and all metric values.
    """
    cmd = [
        "ncu",
        "--import",
        str(ncu_rep_path),
        "--csv",
        "--page",
        "raw",
    ]

    if kernel_filter:
        cmd.extend(["--kernel-name", kernel_filter])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running NCU: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse CSV output
    reader = csv.DictReader(StringIO(result.stdout))
    rows = list(reader)

    return rows


def find_metric_value(
    row: dict, metric_patterns: list[str], available_metrics: set[str]
) -> float | None:
    """Find first available metric from patterns and return its value."""
    for pattern in metric_patterns:
        if pattern in available_metrics:
            # NCU CSV uses "Metric Value" column
            value_key = None
            for key in row.keys():
                if "Metric Value" in key or key == pattern:
                    value_key = key
                    break

            if value_key and row.get(value_key):
                try:
                    return float(row[value_key].replace(",", ""))
                except (ValueError, AttributeError):
                    pass

            # Try direct metric name lookup
            if pattern in row and row[pattern]:
                try:
                    return float(row[pattern].replace(",", ""))
                except (ValueError, AttributeError):
                    pass

    return None


def process_metrics(raw_rows: list[dict], available_metrics: set[str]) -> list[KernelMetrics]:
    """Process raw NCU output into structured kernel metrics."""
    # Group rows by kernel name
    kernels: dict[str, KernelMetrics] = {}

    for row in raw_rows:
        # Find kernel name - try common column names
        kernel_name = None
        for key in ["Kernel Name", "kernel_name", "Name"]:
            if key in row and row[key]:
                kernel_name = row[key]
                break

        if not kernel_name:
            kernel_name = "Unknown"

        if kernel_name not in kernels:
            kernels[kernel_name] = KernelMetrics(kernel_name=kernel_name)

        km = kernels[kernel_name]

        # Extract metrics for each category
        metric_name = row.get("Metric Name", "")

        for category, configs in METRIC_CONFIGS.items():
            for display_name, patterns, _ in configs:
                if metric_name in patterns:
                    value = None
                    for key in row:
                        if "Value" in key:
                            try:
                                value = float(row[key].replace(",", ""))
                                break
                            except (ValueError, AttributeError):
                                pass

                    if value is not None:
                        target = getattr(km, category)
                        if display_name not in target:
                            target[display_name] = value

    return list(kernels.values())


def print_table(title: str, data: dict[str, float], unit: str) -> None:
    """Print a formatted metric table."""
    if not data:
        return

    print(f"\n{title}")
    print("-" * 50)
    print(f"{'Metric Name':<30} {'Unit':<10} {'Value':>10}")
    print("-" * 50)

    for name, value in sorted(data.items(), key=lambda x: -x[1]):
        print(f"{name:<30} {unit:<10} {value:>10.2f}")


def print_metrics(kernel_metrics: list[KernelMetrics]) -> None:
    """Print metrics in human-readable table format."""
    for km in kernel_metrics:
        print(f"\n{'=' * 60}")
        print(f"Kernel: {km.kernel_name}")
        print("=" * 60)

        print_table("GPU Throughput", km.gpu_throughput, "%")
        print_table("Pipe Utilization", km.pipe_utilization, "%")
        print_table("Warp Stalls", km.warp_stalls, "inst")


def print_json(kernel_metrics: list[KernelMetrics]) -> None:
    """Print metrics as JSON."""
    output = []
    for km in kernel_metrics:
        output.append(
            {
                "kernel_name": km.kernel_name,
                "gpu_throughput": km.gpu_throughput,
                "pipe_utilization": km.pipe_utilization,
                "warp_stalls": km.warp_stalls,
            }
        )
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract curated metrics from NCU .ncu-rep files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s profile.ncu-rep
  %(prog)s profile.ncu-rep --json
  %(prog)s profile.ncu-rep --kernel "CopyTranspose"
        """,
    )
    parser.add_argument("ncu_rep", type=Path, help="Path to .ncu-rep file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--kernel", "-k", help="Filter by kernel name (substring match)")

    args = parser.parse_args()

    if not args.ncu_rep.exists():
        print(f"Error: File not found: {args.ncu_rep}", file=sys.stderr)
        sys.exit(1)

    if not check_ncu_available():
        print("Error: NCU (Nsight Compute) not found in PATH", file=sys.stderr)
        print("Install NVIDIA Nsight Compute and ensure 'ncu' is in PATH", file=sys.stderr)
        sys.exit(1)

    # Get available metrics for this report
    available_metrics = get_available_metrics(args.ncu_rep)

    # Extract raw metrics
    raw_rows = extract_raw_metrics(args.ncu_rep, args.kernel)

    if not raw_rows:
        print("No metrics found in report", file=sys.stderr)
        if args.kernel:
            print(f"  (filtered by kernel: {args.kernel})", file=sys.stderr)
        sys.exit(1)

    # Process into structured format
    kernel_metrics = process_metrics(raw_rows, available_metrics)

    # Output
    if args.json:
        print_json(kernel_metrics)
    else:
        print_metrics(kernel_metrics)


if __name__ == "__main__":
    main()
