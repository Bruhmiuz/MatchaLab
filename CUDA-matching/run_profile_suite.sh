#!/bin/bash

PROGRAM_NAME=$1
MANIFEST=$2
GPU_TYPE=$3
TASK=$4

if [[ -z "$MANIFEST" || -z "$GPU_TYPE" ]]; then
    echo "Usage: $0 <program_name> <manifest_file> <a100|h100|all> <time|nsys|ncu>"
    exit 1
fi

GPUS_TO_RUN=()
[[ "$GPU_TYPE" == "a100" || "$GPU_TYPE" == "all" ]] && GPUS_TO_RUN+=("a100")
[[ "$GPU_TYPE" == "h100" || "$GPU_TYPE" == "all" ]] && GPUS_TO_RUN+=("h100")

echo "🚀 Starting Profiling Suite using $MANIFEST"
mkdir -p reports outputs

for CURRENT_GPU in "${GPUS_TO_RUN[@]}"; do
    if [[ "$CURRENT_GPU" == "a100" ]]; then
        BENCHMARKS=("./bench-a100-1" "./bench-a100-2")
        SLURM_ARGS="--gpus=a100-40:1 --constraint=xgpg"
    else 
        BENCHMARKS=("./bench-h100-1")
        SLURM_ARGS="--gpus=h100-47:1 --constraint=xgpi"
    fi

    SRUN_BASE="srun --ntasks=1 --cpus-per-task=1 --cpu_bind=core --mem=20G $SLURM_ARGS"

    echo "🏗️  PROFILING ON ARCHITECTURE: $CURRENT_GPU"

    # --- FIX 1: Use File Descriptor 3 to prevent stdin "theft" ---
    while read -u 3 -r samp_file sig_file; do
        # Skip empty lines
        [[ -z "$sig_file" ]] && continue

        BASE_ID=$(basename "$samp_file" .fastq)
        
        echo "🔍 Running $PROGRAM_NAME on $CURRENT_GPU ($BASE_ID)"
        
        # NSYS: Timeline & API trace
        # We pass /dev/null to matchar's stdin to be absolutely safe
        if [[ "$TASK" == "nsys" ]]; then
            $SRUN_BASE nsys profile --cuda-event-trace=false -f true -o "reports/nsys_${CURRENT_GPU}_${PROGRAM_NAME}_${BASE_ID}" "./${PROGRAM_NAME}" "$samp_file" "$sig_file" < /dev/null
        
        elif [[ "$TASK" == "ncu" ]]; then
            $SRUN_BASE ncu --set=full --clock-control=none --import-source=yes -f -o "reports/ncu_${CURRENT_GPU}_${PROGRAM_NAME}_${BASE_ID}" "./${PROGRAM_NAME}" "$samp_file" "$sig_file" < /dev/null

        # --- Time (Repeat 10x and Extract) ---
        else
            TIME_FILE="reports/time_${CURRENT_GPU}_${PROGRAM_NAME}_${BASE_ID}.txt"
            OUTPUT_FILE="outputs/${CURRENT_GPU}_${PROGRAM_NAME}_${BASE_ID}.txt"
            echo "⏱️ Benchmarking time for $BASE_ID (10 runs)..."
            
            # Clear or create the specific time file for this testcase
            > "$TIME_FILE"

            for i in {1..10}; do
                # 1. Run the program and capture output to a temp string
                # Note: We still use &> for the log, but we'll parse the log file immediately after
                $SRUN_BASE "./${PROGRAM_NAME}" "$samp_file" "$sig_file" < /dev/null &> $OUTPUT_FILE
                
                # 2. Extract the time value (looking for the line shown in your image)
                # This regex looks for the number following "Total runMatcher time:"
                EXTRACTED_TIME=$(grep "Total runMatcher time:" $OUTPUT_FILE | awk -F':' '{print $2}' | sed 's/s//g' | xargs)
                
                # 3. Save only the number to the output file
                if [[ ! -z "$EXTRACTED_TIME" ]]; then
                    echo "$EXTRACTED_TIME" >> "$TIME_FILE"
                    echo "  Run $i: ${EXTRACTED_TIME}s"
                else
                    echo "  Run $i: ❌ Failed to extract time"
                fi
            done

        fi

    done 3< "$MANIFEST"  # Bind manifest to descriptor 3
done

echo "------------------------------------------------"
echo "🏁 All tasks complete."