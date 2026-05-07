#!/bin/bash

# ==========================================
# 1. DEFAULTS & CLI ARGUMENTS
# ==========================================
MPI_PROGRAM="./tester"
REPEATS=1
BASE_SEED=42
MASTER_FILE="inputs/master.txt"

usage() {
    echo "Usage: $0 MANIFEST_FILE [options]"
    echo "  MANIFEST_FILE  Path to the CSV manifest file (Required)"
    echo "  -p PROGRAM     MPI program path (default: $MPI_PROGRAM)"
    echo "  -H HW_SPEC     1. Hardware spec, e.g. \"[i7-7700*1&xs-4114*2]\""
    echo "  -N NODE_SPEC   2. Explicit node spec, e.g. \"[033&034]\""
    echo "  -R RAND_SPEC   3. Random nodes min,max, e.g. [2,4]"
    echo "  -t TRIALS      Number of times to run each config (default: $REPEATS)"
    echo "  -s SEED        Base random seed (default: $BASE_SEED)"
    echo "  -h             Show this help"
}

if [ "$#" -lt 1 ] || [[ "$1" == -* ]]; then
    echo "Error: Manifest file is required as the first argument."
    usage
    exit 1
fi

MANIFEST_FILE="$1"
shift

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "Error: Manifest file '$MANIFEST_FILE' does not exist."
    exit 1
fi

MANIFEST_BASENAME=$(basename "$MANIFEST_FILE")
EXP_NAME="${MANIFEST_BASENAME%_manifest.csv}"
OUTPUT_FILE="outputs/${EXP_NAME}.txt"

mkdir -p outputs
> "$OUTPUT_FILE"
echo "Log initialized for manifest: $MANIFEST_FILE" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"

MODE=""
SPEC=""

while getopts "p:H:N:R:t:s:h" opt; do
    case $opt in
        p) MPI_PROGRAM="$OPTARG" ;;
        H) MODE="HARDWARE"; SPEC="$OPTARG" ;;
        N) MODE="NODE"; SPEC="$OPTARG" ;;
        R) MODE="RANDOM"; SPEC="$OPTARG" ;;
        t) REPEATS="$OPTARG" ;;
        s) BASE_SEED="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Error: You must specify exactly one allocation option: -H, -N, or -R"
    usage
    exit 1
fi

# ==========================================
# 2. HARDWARE MAP
# ==========================================
declare -A PARTITION_CORES=(
    # ["xs-4114"]=20 
    ["i7-7700"]=8 
    # ["dxs-4114"]=40 
    # ["i7-9700"]=8 
    ["w5-3423"]=24 
    ["i7-13700"]=16
)

declare -A PARTITION_NODES=(
    # ["xs-4114"]="008"
    ["i7-7700"]="015 016 012"
    # ["dxs-4114"]="018 019"
    # ["i7-9700"]="020"
    ["w5-3423"]="029 030 031 032 027"
    ["i7-13700"]="034 035 033 036 037 038 039 040"
)

declare -A NODE_TO_CORES
ALL_KNOWN_NODES=()
for part in "${!PARTITION_NODES[@]}"; do
    cores=${PARTITION_CORES[$part]}
    for node in ${PARTITION_NODES[$part]}; do
        NODE_TO_CORES[$node]=$cores
        ALL_KNOWN_NODES+=("$node")
    done
done

# Deep clean to strip all brackets, spaces, and invisible carriage returns
CLEAN_SPEC="${SPEC//[/}"
CLEAN_SPEC="${CLEAN_SPEC//]/}"
CLEAN_SPEC=$(echo "$CLEAN_SPEC" | tr -d ' \r')

# ==========================================
# 3. DYNAMIC NODE ALLOCATOR FUNCTION
# ==========================================
allocate_nodes() {
    SELECTED_NODES=()
    
    if [[ "$MODE" == "NODE" ]]; then
        IFS='&' read -ra SELECTED_NODES <<< "$CLEAN_SPEC"
        FORMATTED_SPEC="[$CLEAN_SPEC]"

    elif [[ "$MODE" == "HARDWARE" ]]; then
        # BUGFIX: Changed variable from 'GROUPS' (which is a reserved Bash system array) to 'HW_GROUPS'
        IFS='&' read -ra HW_GROUPS <<< "$CLEAN_SPEC"
        for group in "${HW_GROUPS[@]}"; do
            
            # Handle missing "*count" by defaulting to 1
            if [[ "$group" == *"*"* ]]; then
                IFS='*' read -r part count <<< "$group"
            else
                part="$group"
                count=1
            fi
            
            part="${part/xs4114/xs-4114}"
            part="${part/dxs4114/dxs-4114}"
            
            read -ra avail_nodes <<< "${PARTITION_NODES[$part]}"
            
            if (( count > ${#avail_nodes[@]} )); then
                echo "Error: Not enough nodes for hardware '$part' (Requested $count, Available ${#avail_nodes[@]})"
                exit 1
            fi
            
            shuffled=($(shuf -e "${avail_nodes[@]}"))
            for ((j=0; j<count; j++)); do
                SELECTED_NODES+=("${shuffled[$j]}")
            done
        done

    elif [[ "$MODE" == "RANDOM" ]]; then
        if [[ "$CLEAN_SPEC" == *","* ]]; then
            IFS=',' read -r min_n max_n <<< "$CLEAN_SPEC"
        else
            min_n="$CLEAN_SPEC"
            max_n="$CLEAN_SPEC"
        fi
        
        if (( min_n > ${#ALL_KNOWN_NODES[@]} )); then
            echo "Error: Not enough nodes available (Requested min $min_n, Available ${#ALL_KNOWN_NODES[@]})"
            exit 1
        fi
        
        (( max_n > ${#ALL_KNOWN_NODES[@]} )) && max_n=${#ALL_KNOWN_NODES[@]}
        if (( max_n < min_n )); then max_n=$min_n; fi
        
        local RAND_COUNT=$(( RANDOM % (max_n - min_n + 1) + min_n ))
        
        shuffled=($(shuf -e "${ALL_KNOWN_NODES[@]}"))
        for ((j=0; j<RAND_COUNT; j++)); do
            SELECTED_NODES+=("${shuffled[$j]}")
        done
    fi

    if [[ "$MODE" != "NODE" ]]; then
        FORMATTED_SPEC="["
        for ((j=0; j<${#SELECTED_NODES[@]}; j++)); do
            FORMATTED_SPEC+="${SELECTED_NODES[$j]}"
            if (( j < ${#SELECTED_NODES[@]} - 1 )); then
                FORMATTED_SPEC+="&"
            fi
        done
        FORMATTED_SPEC+="]"
    fi

    TOTAL_CORES=0
    TOTAL_NODES=${#SELECTED_NODES[@]}
    ALLOC_NODES=()
    MPI_HOST_STR="" 

    for node in "${SELECTED_NODES[@]}"; do
        local cores=${NODE_TO_CORES[$node]}
        if [[ -z "$cores" ]]; then
            echo "Error: Unknown node ID '$node'"
            exit 1
        fi
        TOTAL_CORES=$((TOTAL_CORES + cores))
        ALLOC_NODES+=("soctf-pdc-${node}")
        
        if [ -n "$MPI_HOST_STR" ]; then MPI_HOST_STR+=","; fi
        MPI_HOST_STR+="soctf-pdc-${node}:${cores}"
    done

    SAVE_IFS=$IFS
    IFS=','
    SLURM_NODELIST="${ALLOC_NODES[*]}"
    IFS=$SAVE_IFS
}

# ==========================================
# 4. AUTO-GENERATION PIPELINE
# ==========================================
if [ ! -f "$MASTER_FILE" ]; then
    echo "Warning: $MASTER_FILE not found! Triggering auto-generation."
    if [ ! -f "gen_master.cpp" ]; then echo "Error: gen_master.cpp missing."; exit 1; fi
    srun -N 1 -n 1 --time=00:05:00 g++ -O3 gen_master.cpp -o gen_master || exit 1
    srun -N 1 -n 1 --time=00:10:00 ./gen_master 200000 500 || exit 1
fi

MASTER_LINES=$(wc -l < "$MASTER_FILE" | awk '{print $1}')

generate_data() {
    local N=$1; local Q=$2; local A=$3; local K=$4; local seed=$5
    local targets_file="/tmp/targets_$$_${N}_${Q}.txt"
    local total_lines=$((N + Q))

    awk -v n="$total_lines" -v max="$MASTER_LINES" -v seed="$seed" '
    BEGIN { srand(seed); for(i=0; i<n; i++) print int(rand() * max) + 1; }' | sort -n > "$targets_file"

    awk -v target_file="$targets_file" -v N="$N" -v Q="$Q" -v A="$A" -v K="$K" -v seed="$seed" '
    BEGIN {
        srand(seed); print N, Q, A
        t_idx = 1
        while ((getline t < target_file) > 0) { targets[t_idx++] = t }
        total_targets = t_idx - 1; close(target_file)
        current_t = 1; printed = 0
    }
    {
        while (current_t <= total_targets && NR == targets[current_t]) {
            printed++
            if (printed <= N) {
                printf "%d", int(rand() * 100)
                for(i=2; i<=A+1; i++) printf " %s", $i; print ""
            } else {
                printf "Q %d", int(rand() * 2 * K) + 1
                for(i=2; i<=A+1; i++) printf " %s", $i; print ""
            }
            current_t++
        }
        if (current_t > total_targets) exit
    }' "$MASTER_FILE"

    rm -f "$targets_file"
}

# ==========================================
# 5. SLURM EXECUTION SWEEP
# ==========================================
echo "=================================================="
echo " Allocation Mode : $MODE ($SPEC)"
echo " Details         : Dynamic Retry on Busy (Bypassed for -N)"
echo " Saving Output   : $OUTPUT_FILE"
echo "=================================================="

TEMP_FILE="temp_dataset_$$.in"
SKIP_HEADER=true
LAST_N=""; LAST_Q=""; LAST_A=""; LAST_K=""

while IFS=, read -r POINTS QUERIES DIMS AVG_K EXEC_ARGS || [ -n "$POINTS" ]; do
    if [ "$SKIP_HEADER" = true ]; then
        SKIP_HEADER=false
        continue
    fi
    
    EXEC_ARGS=$(echo "$EXEC_ARGS" | tr -d '\r')

    echo ">>> Config: N=$POINTS, Q=$QUERIES, A=$DIMS, K=$AVG_K, ARGS=[$EXEC_ARGS]" | tee -a "$OUTPUT_FILE"

    if [ "$POINTS" != "$LAST_N" ] || [ "$QUERIES" != "$LAST_Q" ] || [ "$DIMS" != "$LAST_A" ] || [ "$AVG_K" != "$LAST_K" ]; then
        echo "    -> Generating new dataset..." | tee -a "$OUTPUT_FILE"
        generate_data "$POINTS" "$QUERIES" "$DIMS" "$AVG_K" "$BASE_SEED" > "$TEMP_FILE"
        LAST_N="$POINTS"; LAST_Q="$QUERIES"; LAST_A="$DIMS"; LAST_K="$AVG_K"
    else
        echo "    -> Reusing existing dataset..." | tee -a "$OUTPUT_FILE"
    fi

    for (( i=1; i<=REPEATS; i++ )); do
        
        # --- EXPLICIT NODE MODE (-N) ---
        # Queue normally. No immediate fail, no retry loop.
        if [[ "$MODE" == "NODE" ]]; then
            allocate_nodes # Parse once
            
            MSG="    -> Trial $i/$REPEATS | Nodes: $FORMATTED_SPEC | Cores: $TOTAL_CORES"
            echo "$MSG" | tee -a "$OUTPUT_FILE"
            
            salloc --nodelist="$SLURM_NODELIST" --ntasks="$TOTAL_CORES" -N "$TOTAL_NODES" --exclusive --time=00:15:00 \
                mpirun --host "$MPI_HOST_STR" --bind-to hwthread "$MPI_PROGRAM" $EXEC_ARGS < "$TEMP_FILE" 1> /dev/null 2>> "$OUTPUT_FILE"
        
        # --- DYNAMIC MODES (-H or -R) ---
        # Fail instantly if nodes are busy, pick new nodes, and try again.
        else
            ALLOC_SUCCESS=false
            ATTEMPT=1
            
            while [ "$ALLOC_SUCCESS" = false ]; do
                
                allocate_nodes # Get a fresh set of random nodes
                
                MSG="    -> Trial $i/$REPEATS (Attempt $ATTEMPT) | Nodes: $FORMATTED_SPEC | Cores: $TOTAL_CORES"
                echo "$MSG" | tee -a "$OUTPUT_FILE"
                
                TMP_ERR=$(mktemp)
                
                salloc --immediate=2 --nodelist="$SLURM_NODELIST" --ntasks="$TOTAL_CORES" -N "$TOTAL_NODES" --exclusive --time=00:15:00 \
                    mpirun --host "$MPI_HOST_STR" --bind-to hwthread "$MPI_PROGRAM" $EXEC_ARGS < "$TEMP_FILE" 1> /dev/null 2> "$TMP_ERR"
                
                EXIT_CODE=$?
                
                cat "$TMP_ERR" >> "$OUTPUT_FILE"
                
                if [ $EXIT_CODE -eq 0 ]; then
                    ALLOC_SUCCESS=true
                else
                    WARN_MSG="    -> [!] Nodes busy or allocation failed (Exit $EXIT_CODE). Regenerating..."
                    echo "$WARN_MSG" | tee -a "$OUTPUT_FILE"
                    
                    ATTEMPT=$((ATTEMPT + 1))
                    sleep 3
                fi
                rm -f "$TMP_ERR"
            done
        fi
            
        echo "" >> "$OUTPUT_FILE" 
    done
done < "$MANIFEST_FILE"

echo "Sweep complete. Output saved to $OUTPUT_FILE"
rm -f "$TEMP_FILE"