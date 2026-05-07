#!/bin/bash

# https://gemini.google.com/share/e899014c726a

# Default Base Variables
POINTS_STR="10000"
QUERIES_STR="1000"
DIMS_STR="20"
AVG_K_STR="20"

usage() {
    echo "Usage: $0 exp_name [options]"
    echo "Options accept space-separated ranges in quotes:"
    echo "  -N 'ranges'       Points / N (default: $POINTS_STR)"
    echo "  -Q 'ranges'       Queries / Q (default: $QUERIES_STR)"
    echo "  -A 'ranges'       Dimensions / A (default: $DIMS_STR)"
    echo "  -K 'ranges'       Average neighbours / K (default: $AVG_K_STR)"
    echo "  --arg1 'ranges'   First executable arg (e.g., DATA_SPLIT)"
    echo "  --arg2 'ranges'   Second executable arg (e.g., MODE)"
    echo "  (You can chain as many --argX flags as you need in order)"
    echo "  -h, --help        Show this help"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

EXP_NAME=$1
shift

# Array to dynamically hold all --argX inputs
declare -a EXEC_ARGS_STRS

# Manual parsing loop to support long flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        -N) POINTS_STR="$2"; shift 2 ;;
        -Q) QUERIES_STR="$2"; shift 2 ;;
        -A) DIMS_STR="$2"; shift 2 ;;
        -K) AVG_K_STR="$2"; shift 2 ;;
        --arg*) 
            # Catch anything starting with --arg and save its range
            EXEC_ARGS_STRS+=("$2")
            shift 2 
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: Unknown option $1"; usage; exit 1 ;;
    esac
done

MANIFEST_FILE="experiments/${EXP_NAME}_manifest.csv"

# Write the header
echo "POINTS,QUERIES,DIMS,AVG_K,EXEC_ARGS" > "$MANIFEST_FILE"

read -ra n_arr <<< "$POINTS_STR"
read -ra q_arr <<< "$QUERIES_STR"
read -ra d_arr <<< "$DIMS_STR"
read -ra k_arr <<< "$AVG_K_STR"

# ---------------------------------------------------------
# Dynamic Cartesian Product for Executable Arguments
# ---------------------------------------------------------
exec_combinations=("")
for arg_str in "${EXEC_ARGS_STRS[@]}"; do
    read -ra elements <<< "$arg_str"
    new_exec=()
    for prefix in "${exec_combinations[@]}"; do
        for el in "${elements[@]}"; do
            if [ -z "$prefix" ]; then
                new_exec+=("$el")
            else
                new_exec+=("$prefix $el")
            fi
        done
    done
    exec_combinations=("${new_exec[@]}")
done

# ---------------------------------------------------------
# Write all combinations to the CSV
# ---------------------------------------------------------
total_configs=0
for n in "${n_arr[@]}"; do
    for q in "${q_arr[@]}"; do
        for d in "${d_arr[@]}"; do
            for k in "${k_arr[@]}"; do
                for a in "${exec_combinations[@]}"; do
                    echo "$n,$q,$d,$k,$a" >> "$MANIFEST_FILE"
                    ((total_configs++))
                done
            done
        done
    done
done

echo "=================================================="
echo "Manifest created : $MANIFEST_FILE"
echo "Total configs    : $total_configs"
echo "=================================================="