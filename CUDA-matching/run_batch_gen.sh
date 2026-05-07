#!/bin/bash

# --- 1. Defaults ---
# These match your create_testcase.sh defaults
N_SG=210
L_SG=6500
N_SP=400
L_SP=1500000
V=10
P=0.01
R=0.1

# --- 2. Arguments ---
EXP_NAME=$1
SWEEP_FLAG=$2
VALUES=$3
EXTRA_ARGS=${@:4}

if [ -z "$EXP_NAME" ] || [ -z "$SWEEP_FLAG" ] || [ -z "$VALUES" ]; then
    echo "Usage: $0 <experiment_name> <flag_to_sweep> \"<values>\" [overrides]"
    echo "Example: $0 vary_percent p \"0.01 0.05 0.1\" -n-sg 500"
    echo "Flags:"
    echo "  -n-sg  Number of signatures (default: $N_SG)"
    echo "  -l-sg  Avg signature length (default: $L_SG)"
    echo "  -n-sp  Number of samples    (default: $N_SP)"
    echo "  -l-sp  Avg sample length    (default: $L_SP)"
    echo "  -v  Max viruses          (default: $V)"
    echo "  -p  Percent virus (0-1)  (default: $P)"
    echo "  -r  N-ratio              (default: $R)"
    exit 1
fi

# --- 3. Parse Overrides (EXTRA_ARGS) ---
# We use a manual loop because getopts doesn't handle -n-sg style well
set -- $EXTRA_ARGS
while [ $# -gt 0 ]; do
    case "$1" in
        -n-sg) N_SG="$2"; shift ;;
        -l-sg) L_SG="$2"; shift ;;
        -n-sp) N_SP="$2"; shift ;;
        -l-sp) L_SP="$2"; shift ;;
        -v)    V="$2";    shift ;;
        -p)    P="$2";    shift ;;
        -r)    R="$2";    shift ;;
    esac
    shift
done

# --- 4. Setup Experiment File ---
mkdir -p experiments
EXP_FILE="experiments/${EXP_NAME}.txt"
> "$EXP_FILE" # Clear existing file

echo "🚀 Starting Sweep: $EXP_NAME"

# --- 5. Sweep Loop ---
for VAL in $VALUES; do
    # Override the sweep variable for this iteration
    case "$SWEEP_FLAG" in
        n-sg) N_SG=$VAL ;;
        l-sg) L_SG=$VAL ;;
        n-sp) N_SP=$VAL ;;
        l-sp) L_SP=$VAL ;;
        v)    V=$VAL ;;
        p)    P=$VAL ;;
        r)    R=$VAL ;;
    esac

    echo "------------------------------------------------"
    echo "🧪 Iteration: $SWEEP_FLAG = $VAL"

    # Call the positional create_testcase.sh
    # ORDER: num_sigs avg_sig_len num_samples avg_samp_len max_viruses percent_virus n_ratio
    OUTPUT=$(./create_testcase.sh "$N_SG" "$L_SG" "$N_SP" "$L_SP" "$V" "$P" "$R")
    
    # Echo output to terminal for monitoring
    echo "$OUTPUT"
    # Extract filenames from output (using your existing grep logic)
    FASTA=$(echo "$OUTPUT" | grep -E ":" | grep -o "inputs/[^ ]*\.fasta" | head -n 1)
    FASTQ=$(echo "$OUTPUT" | grep -E ":" | grep -o "inputs/[^ ]*\.fastq" | head -n 1)

    if [ ! -z "$FASTA" ] && [ ! -z "$FASTQ" ]; then
        echo "$FASTQ $FASTA" >> "$EXP_FILE"
    fi
done

echo "------------------------------------------------"
echo "🏁 Done. Manifest: $EXP_FILE"