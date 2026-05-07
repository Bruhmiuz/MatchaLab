#!/bin/bash

set -euo pipefail

log() {
    echo "[run_bench] $*"
}

usage() {
    echo "Usage: $0 [--clear-cache] <CONFIG>"
    echo "       $0 clear-cache"
    echo ""
    echo "Available CONFIG options:"
    echo "  1 - Run bench_1 on [i7-7700*1 & i7-13700*1], 24 tasks, 2 nodes"
    echo "  2 - Run bench_2 on [i7-7700*1 & w5-3423*1], 32 tasks, 2 nodes"
    echo "  3 - Run bench_3 on [i7-13700*2], 32 tasks, 2 nodes"
    echo "  4 - Run bench_4 on [i7-13700*1 & w5-3423*1], 40 tasks, 2 nodes"
    echo "  all - Run all above configurations"
    echo ""
    echo "Options:"
    echo "  --clear-cache  Remove cached benchmark outputs before running"
    echo "  clear-cache    Remove cached benchmark outputs and exit"
    echo ""
    echo "Benchmark outputs are cached per exact allocated node combination."
}

clear_cache() {
    mkdir -p outputs

    local removed=0
    local cache_patterns=(
        "outputs/bench_*.out"
        "outputs/bench_*.err"
        "outputs/.bench_*.tmp.*"
    )

    shopt -s nullglob
    for pattern in "${cache_patterns[@]}"; do
        for path in $pattern; do
            rm -f "$path"
            removed=$((removed + 1))
        done
    done
    shopt -u nullglob

    log "Cleared ${removed} cached benchmark file(s)"
}

VALID_CONFIGS=("1" "2" "3" "4" "all")
CLEAR_CACHE_BEFORE_RUN=0

if [ "$#" -eq 1 ] && [ "$1" = "clear-cache" ]; then
    clear_cache
    exit 0
fi

if [ "$#" -eq 2 ] && [ "$1" = "--clear-cache" ]; then
    CLEAR_CACHE_BEFORE_RUN=1
    CONFIG=$2
elif [ "$#" -eq 1 ]; then
    CONFIG=$1
else
    usage
    exit 1
fi

if [[ ! " ${VALID_CONFIGS[*]} " =~ [[:space:]]${CONFIG}[[:space:]] ]]; then
    echo "Error: Invalid CONFIG option '$CONFIG'"
    echo ""
    usage
    exit 1
fi

compare_times() {
    local bench_err=$1
    local engine_err=$2

    if [[ ! -f "$bench_err" || ! -f "$engine_err" ]]; then
        echo "Error: Missing .err files for comparison ($bench_err or $engine_err not found)."
        return 1
    fi

    local bench_time
    local engine_time
    bench_time=$(grep -oP 'Time taken:\s*\K[0-9]+' "$bench_err" | tail -n 1 || true)
    engine_time=$(grep -oP 'Time taken:\s*\K[0-9]+' "$engine_err" | tail -n 1 || true)

    if [[ -z "$bench_time" || -z "$engine_time" ]]; then
        echo "Error: Could not extract timing information from .err files."
        return 1
    fi

    echo ""
    echo "=== Performance Comparison ==="
    echo "Benchmark time: ${bench_time} ms"
    echo "Engine time:    ${engine_time} ms"

    local diff=$((engine_time - bench_time))
    local abs_diff=${diff#-}

    if [ "$bench_time" -ne 0 ]; then
        local percent
        percent=$(awk "BEGIN {printf \"%.2f\", ($engine_time - $bench_time) / $bench_time * 100}")

        if (( $(echo "$percent > 0" | bc -l) )); then
            echo "Difference:     +${abs_diff} ms (${percent}% slower)"
        elif (( $(echo "$percent < 0" | bc -l) )); then
            local faster_percent=${percent#-}
            echo "Difference:     -${abs_diff} ms (${faster_percent}% faster)"
        else
            echo "Difference:     0 ms (No difference)"
        fi
    fi

    echo "=============================="
    echo ""
}

correctness_check() {
    local bench_out=$1
    local engine_out=$2

    if [[ ! -f "$bench_out" || ! -f "$engine_out" ]]; then
        echo "Error: Missing .out files for comparison ($bench_out or $engine_out not found)."
        return 1
    fi

    echo ""
    echo "=== Correctness Check ==="

    if cmp -s "$bench_out" "$engine_out"; then
        echo "Correctness check passed"
    else
        echo "Correctness check failed"
    fi

    echo "========================="
    echo ""
}

run_config() {
    local config_id=$1
    local constraint=$2
    local ntasks=$3
    local benchmark=$4
    local input_file=$5
    local meta_file="outputs/run_${config_id}.meta"

    log "Preparing config ${config_id}: benchmark=${benchmark}, input=${input_file}, constraint=${constraint}, ntasks=${ntasks}"

    if [[ ! -f "$input_file" ]]; then
        log "Missing input file ${input_file}"
        log "Extract inputs.zip first, for example: unzip inputs.zip"
        return 1
    fi

    rm -f "$meta_file"

    log "Requesting Slurm allocation for config ${config_id}"
    salloc --constraint="$constraint" --ntasks="$ntasks" -N 2 --exclusive --time=00:10:00 bash -c '
        set -euo pipefail

        has_valid_timing() {
            local err_file=$1
            [[ -f "$err_file" ]] && grep -qE "Time taken:[[:space:]]*[0-9]+[[:space:]]*ms" "$err_file"
        }

        config_id=$1
        benchmark=$2
        input_file=$3
        meta_file=$4

        node_key=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr "\n" "_")
        node_key=${node_key%_}

        bench_err="outputs/bench_${config_id}_${node_key}.err"
        bench_out="outputs/bench_${config_id}_${node_key}.out"
        engine_err="outputs/engine_${config_id}_${node_key}.err"
        engine_out="outputs/engine_${config_id}_${node_key}.out"
        nodes_file="outputs/nodes_${config_id}_${node_key}.txt"
        tmp_bench_out=$(mktemp "outputs/.bench_${config_id}_${node_key}.out.tmp.XXXXXX")
        tmp_bench_err=$(mktemp "outputs/.bench_${config_id}_${node_key}.err.tmp.XXXXXX")

        cleanup_tmp() {
            if [[ -n "${tmp_bench_out:-}" ]]; then
                rm -f "$tmp_bench_out"
            fi
            if [[ -n "${tmp_bench_err:-}" ]]; then
                rm -f "$tmp_bench_err"
            fi
        }

        trap cleanup_tmp EXIT INT TERM

        scontrol show hostnames "$SLURM_JOB_NODELIST" > "$nodes_file"
        echo "[run_bench][config ${config_id}] Allocation granted"
        echo "[run_bench][config ${config_id}] Allocated nodes:"
        cat "$nodes_file"
        echo "[run_bench][config ${config_id}] Benchmark stdout: $bench_out"
        echo "[run_bench][config ${config_id}] Benchmark stderr: $bench_err"
        echo "[run_bench][config ${config_id}] Engine stdout:    $engine_out"
        echo "[run_bench][config ${config_id}] Engine stderr:    $engine_err"

        if [[ -f "$bench_err" && -f "$bench_out" ]] && has_valid_timing "$bench_err"; then
            echo "[run_bench][config ${config_id}] Benchmark cache hit for this exact node combination"
        else
            if [[ -f "$bench_err" || -f "$bench_out" ]]; then
                echo "[run_bench][config ${config_id}] Existing benchmark cache is missing or invalid; rebuilding it"
                rm -f "$bench_err" "$bench_out"
            fi

            echo "[run_bench][config ${config_id}] Running reference benchmark ${benchmark}"
            mpirun --timeout 300 --bind-to hwthread "./benchmarks/${benchmark}" < "$input_file" > "$tmp_bench_out" 2> "$tmp_bench_err"

            if ! has_valid_timing "$tmp_bench_err"; then
                echo "[run_bench][config ${config_id}] Reference benchmark did not produce a valid timing line; refusing to cache this run" >&2
                exit 1
            fi

            mv "$tmp_bench_out" "$bench_out"
            mv "$tmp_bench_err" "$bench_err"
            tmp_bench_out=""
            tmp_bench_err=""
            echo "[run_bench][config ${config_id}] Reference benchmark completed"
            echo "[run_bench][config ${config_id}] Cached valid benchmark result for this exact node combination"
        fi

        echo "[run_bench][config ${config_id}] Running student engine"
        mpirun --timeout 300 --bind-to hwthread ./engine < "$input_file" > "$engine_out" 2> "$engine_err"
        echo "[run_bench][config ${config_id}] Student engine completed"

        printf "%s\n%s\n%s\n%s\n%s\n" \
            "$bench_err" \
            "$bench_out" \
            "$engine_err" \
            "$engine_out" \
            "$nodes_file" > "$meta_file"
    ' _ "$config_id" "$benchmark" "$input_file" "$meta_file"

    mapfile -t meta < "$meta_file"

    log "Config ${config_id} finished. Comparing benchmark and engine outputs."
    log "Node list recorded in ${meta[4]}"
    compare_times "${meta[0]}" "${meta[2]}"
    correctness_check "${meta[1]}" "${meta[3]}"
}

log "Building binaries"
make
log "Build complete"

log "Ensuring outputs directory exists"
mkdir -p outputs
log "Outputs directory ready at outputs/"

if [ "$CLEAR_CACHE_BEFORE_RUN" -eq 1 ]; then
    clear_cache
fi

case "$CONFIG" in
    1)
        run_config "1" "[i7-7700*1&i7-13700*1]" "24" "bench_1" "inputs/input1.in"
        ;;
    2)
        run_config "2" "[i7-7700*1&w5-3423*1]" "32" "bench_2" "inputs/input2.in"
        ;;
    3)
        run_config "3" "[i7-13700*2]" "32" "bench_3" "inputs/input2.in"
        ;;
    4)
        run_config "4" "[i7-13700*1&w5-3423*1]" "40" "bench_4" "inputs/input3.in"
        ;;
    all)
        run_config "1" "[i7-7700*1&i7-13700*1]" "24" "bench_1" "inputs/input1.in"
        run_config "2" "[i7-7700*1&w5-3423*1]" "32" "bench_2" "inputs/input2.in"
        run_config "3" "[i7-13700*2]" "32" "bench_3" "inputs/input2.in"
        run_config "4" "[i7-13700*1&w5-3423*1]" "40" "bench_4" "inputs/input3.in"
        ;;
esac
