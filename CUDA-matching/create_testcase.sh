#!/bin/bash

# https://gemini.google.com/share/1842a83a5005

# --- Check for exactly 7 arguments ---
if [ "$#" -ne 7 ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage: $0 <num_sigs> <avg_sig_len> <num_samples> <avg_samp_len> <max_viruses> <percent_virus> <n_ratio>"
    echo "Example: $0 210 6500 400 1500000 10 0.01 0.1"
    exit 1
fi

# --- Assign Positional Arguments ---
NUM_SIGS=$1
AVG_SIG_LEN=$2
NUM_SAMPLES=$3
AVG_SAMP_LEN=$4
MAX_VIRUSES=$5
PERCENT_VIRUS=$6
N_RATIO=$7

# Fixed parameters (Constants)
MIN_PHRED=0
MAX_PHRED=93

# Signature Lengths 
MIN_SIG_LEN=$(echo "scale=0; 1 + $AVG_SIG_LEN * 0.46153 / 1" | bc)
MAX_SIG_LEN=$(echo "scale=0; $AVG_SIG_LEN * 1.53847 / 1" | bc)

# Sample Lengths 
MIN_SAMP_LEN=$(echo "scale=0; 1 + $AVG_SAMP_LEN * 0.6666666 / 1" | bc)
MAX_SAMP_LEN=$(echo "scale=0; $AVG_SAMP_LEN * 1.3333334 / 1" | bc)

# --- Sample Distribution Math ---
# We use printf to handle potential float precision issues before passing to bc
NUM_WITH_VIRUS=$(echo "$NUM_SAMPLES * $PERCENT_VIRUS" | bc | awk '{print int($1)}')
NUM_NO_VIRUS=$(( NUM_SAMPLES - NUM_WITH_VIRUS ))

# Ensure the inputs directory exists
mkdir -p inputs

# File naming variables
SIG_FASTA="inputs/${NUM_SIGS}-${AVG_SIG_LEN}-${N_RATIO}.fasta"
SAMP_FASTQ="inputs/${NUM_SIGS}-${AVG_SIG_LEN}-${N_RATIO}-${NUM_SAMPLES}-${AVG_SAMP_LEN}-${MAX_VIRUSES}-${PERCENT_VIRUS}.fastq"

# --- Execution ---

if [ -f "$SIG_FASTA" ]; then
    echo "🦠⏭️: $SIG_FASTA"
else
    srun --time=00:00:20 ./gen_sig "$NUM_SIGS" "$MIN_SIG_LEN" "$MAX_SIG_LEN" "$N_RATIO" > "$SIG_FASTA"

    if [ $? -eq 0 ]; then
        echo "🦠✅: $SIG_FASTA"
    else
        echo "🦠❌ Error generating signatures"
        rm $SIG_FASTA
        exit 1
    fi
fi


if [ -f "$SAMP_FASTQ" ]; then
    echo "🧬⏭️: $SAMP_FASTQ"
else
    srun --time=00:20:00 ./gen_sample "$SIG_FASTA" "$NUM_NO_VIRUS" "$NUM_WITH_VIRUS" 1 "$MAX_VIRUSES" "$MIN_SAMP_LEN" "$MAX_SAMP_LEN" "$MIN_PHRED" "$MAX_PHRED" "$N_RATIO" > "$SAMP_FASTQ"

    if [ $? -eq 0 ]; then
        echo "🧬✅: $SAMP_FASTQ"
    else
        echo "🧬❌ Error generating samples"
        rm $SAMP_FASTQ
        exit 1
    fi
fi

echo "🚀 Generation complete."