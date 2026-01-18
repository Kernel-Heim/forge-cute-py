#!/usr/bin/env bash
set -euo pipefail

# Helper script for profiling forge-cute-py kernels with Nsight tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILES_DIR="${REPO_ROOT}/profiles"

# Ensure profiles directory exists
mkdir -p "${PROFILES_DIR}"

show_usage() {
    cat <<EOF
Usage: $(basename "$0") [TOOL] [OPTIONS] -- [COMMAND]

Profile forge-cute-py kernels with NVIDIA Nsight tools.

TOOLS:
  ncu               Nsight Compute (GPU kernel profiling)
  nsys              Nsight Systems (system-wide profiling)
  sanitizer         compute-sanitizer (memory/race detection)

NCU OPTIONS:
  --extract         Extract curated metrics after profiling (GPU throughput,
                    pipe utilization, warp stalls)

EXAMPLES:
  # Profile with Nsight Compute (detailed kernel metrics)
  $0 ncu -- python -m forge_cute_py.env_check
  $0 ncu --set full -- python bench/benchmark_copy_transpose.py
  $0 ncu --extract -- python bench/benchmark_copy_transpose.py

  # Profile with Nsight Systems (timeline view)
  $0 nsys -- python -m forge_cute_py.env_check
  $0 nsys --stats=true -- python bench/run.py --suite smoke

  # Memory checking with compute-sanitizer
  $0 sanitizer -- python -m forge_cute_py.env_check
  $0 sanitizer --tool memcheck -- python -m forge_cute_py.env_check

OUTPUT:
  Profiles are saved to: ${PROFILES_DIR}/
  Filenames are auto-generated with timestamps.

NOTES:
  - Ensure Nsight tools are installed and in PATH
  - Use 'uv run' prefix for commands that need the virtual environment
  - For custom output locations, use tool-specific flags before --
EOF
}

if [[ $# -lt 3 ]] || [[ "$2" != "--" && "$3" != "--" ]]; then
    show_usage
    exit 1
fi

TOOL="$1"
shift

# Parse options until we hit --
TOOL_OPTS=()
EXTRACT_METRICS=false
while [[ $# -gt 0 ]] && [[ "$1" != "--" ]]; do
    if [[ "$1" == "--extract" ]]; then
        EXTRACT_METRICS=true
    else
        TOOL_OPTS+=("$1")
    fi
    shift
done

if [[ "$1" == "--" ]]; then
    shift
fi

COMMAND=("$@")

if [[ ${#COMMAND[@]} -eq 0 ]]; then
    echo "Error: No command specified after --"
    show_usage
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case "${TOOL}" in
    ncu)
        if [[ ${#TOOL_OPTS[@]} -eq 0 ]]; then
            # Default: full metrics
            TOOL_OPTS=("--set" "full")
        fi
        OUTPUT_FILE="${PROFILES_DIR}/ncu_${TIMESTAMP}"
        echo "Running Nsight Compute profiling..."
        echo "Output: ${OUTPUT_FILE}.ncu-rep"
        ncu "${TOOL_OPTS[@]}" -o "${OUTPUT_FILE}" "${COMMAND[@]}"
        echo "✓ Profile saved to: ${OUTPUT_FILE}.ncu-rep"
        echo "  View with: ncu-ui ${OUTPUT_FILE}.ncu-rep"
        
        # Extract metrics if requested
        if [[ "${EXTRACT_METRICS}" == "true" ]]; then
            echo ""
            echo "Extracting curated metrics..."
            "${SCRIPT_DIR}/ncu_extract.py" "${OUTPUT_FILE}.ncu-rep"
        fi
        ;;

    nsys)
        OUTPUT_FILE="${PROFILES_DIR}/nsys_${TIMESTAMP}"
        echo "Running Nsight Systems profiling..."
        echo "Output: ${OUTPUT_FILE}.nsys-rep"
        nsys profile "${TOOL_OPTS[@]}" -o "${OUTPUT_FILE}" "${COMMAND[@]}"
        echo "✓ Profile saved to: ${OUTPUT_FILE}.nsys-rep"
        echo "  View with: nsys-ui ${OUTPUT_FILE}.nsys-rep"
        ;;

    sanitizer|compute-sanitizer)
        echo "Running compute-sanitizer..."
        if [[ ${#TOOL_OPTS[@]} -eq 0 ]]; then
            # Default: memcheck
            TOOL_OPTS=("--tool" "memcheck")
        fi
        compute-sanitizer "${TOOL_OPTS[@]}" "${COMMAND[@]}"
        ;;

    *)
        echo "Error: Unknown tool '${TOOL}'"
        echo "Supported tools: ncu, nsys, sanitizer"
        show_usage
        exit 1
        ;;
esac
