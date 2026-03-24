#!/bin/bash
#SBATCH --job-name=ribotish_tisdiff
#SBATCH --array=2-16
#SBATCH --error=ribotish_tisdiff-%A-%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=20
#SBATCH --output ribotish_tisdiff-%A-%a.out  # Output files, %A and %a replacement strings for the master job ID and task ID, respectively.

# ================================================================
# Array job to run run_tisdiff.sh on multiple inputs
# ================================================================

# timing on 2 reps vs 2 reps for all pairs of conditions ranges from ~4 hrs to ~6hrs
# if > 2 

set -euo pipefail

INPUT_FILE="/lab/barcheese01/smaffa/coTISja/data/ribotish_tisdiff_manifest.tsv"
REFERENCE_GTF="/lab/barcheese01/aTIS_data/reference/gencode.v49.primary_assembly.annotation.gtf"

# Find the line corresponding to the arrayed job
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$INPUT_FILE")

# Extract the fields from the manifest file
IFS=$'\t' read -r BASELINE_NAME TEST_NAME BASELINE_PREDICT TEST_PREDICT BASELINE_RIBOSEQ_BAM TEST_RIBOSEQ_BAM RNASEQ_MERGED_COUNTS OUTPUT_FILE EXPORTED_COUNTS_FILE <<< "$LINE"

echo "========================================"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "$TEST_NAME vs. $BASELINE_NAME"
echo "Control files at:"
echo "  riboseq predict: $BASELINE_PREDICT"
echo "  input riboseq bams: $BASELINE_RIBOSEQ_BAM"
echo "Test files at:"
echo "  riboseq predict: $TEST_PREDICT"
echo "  input riboseq bams: $TEST_RIBOSEQ_BAM"
echo "Merged counts file at: $RNASEQ_MERGED_COUNTS"
echo "Outputs will be deposited at: $OUTPUT_FILE | $EXPORTED_COUNTS_FILE"
echo "========================================"

# Run the filter script
ribotish tisdiff \
    -1 "$BASELINE_PREDICT" \
    -2 "$TEST_PREDICT" \
    -a "$BASELINE_RIBOSEQ_BAM" \
    -b "$TEST_RIBOSEQ_BAM" \
    --rnaseq "$RNASEQ_MERGED_COUNTS" \
    -g "$REFERENCE_GTF" \
    -o "$OUTPUT_FILE" \
    --verbose \
    --export "$EXPORTED_COUNTS_FILE" \
    --ipth 1 \
    --iqth 1 

echo ""
echo "Done with task $SLURM_ARRAY_TASK_ID"
