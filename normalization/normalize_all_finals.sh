#!/bin/bash

# Script to normalize all final prediction files from inference checkpoints
# Usage: ./normalize_all_finals.sh

# Set the base directories
INFERENCE_DIR="/media/adam/Samsung/Documents/PAN/for paper - all models/poleval-gender/solution/task_proofreading/02_inference"
NORMALIZER_SCRIPT="/media/adam/Samsung/Documents/PAN/for paper - all models/poleval-gender/normalization/normaliser_ipis.py"

# Check if normalizer script exists
if [ ! -f "$NORMALIZER_SCRIPT" ]; then
    echo "Error: Normalizer script not found at $NORMALIZER_SCRIPT"
    exit 1
fi

# Counter for processed files
count=0
errors=0
skipped=0

# Count total files first (excluding already normalized files)
total_files=$(find "$INFERENCE_DIR"/inference_checkpoints_* -name "predictions_final_*.jsonl" ! -name "*_NORMALIZED.jsonl" | wc -l)

echo "=========================================="
echo "Starting normalization of final predictions"
echo "Total files to process: $total_files"
echo "=========================================="
echo ""

# Find all predictions_final_*.jsonl files (excluding already normalized files)
while IFS= read -r input_file; do
    current=$((count + errors + skipped + 1))
    # Get the directory of the input file
    input_dir=$(dirname "$input_file")
    
    # Get the base filename without path
    input_basename=$(basename "$input_file")
    
    # Create output filename by replacing .jsonl with _NORMALIZED.jsonl
    output_basename="${input_basename%.jsonl}_NORMALIZED.jsonl"
    output_file="$input_dir/$output_basename"
    
    # Skip if already normalized
    if [ -f "$output_file" ]; then
        ((skipped++))
        clear
        echo "=========================================="
        echo "Progress: $current/$total_files"
        echo "Processed: $count | Errors: $errors | Skipped: $skipped"
        echo "=========================================="
        echo ""
        echo "Skipping (already exists): $input_basename"
        echo "   Output: $output_file"
        sleep 0.25
        continue
    fi
    
    #clear
    echo "=========================================="
    echo "Progress: $current/$total_files"
    echo "Processed: $count | Errors: $errors | Skipped: $skipped"
    echo "=========================================="
    echo ""
    echo "📝 Processing: $input_basename"
    echo "   Input:  $input_file"
    echo "   Output: $output_file"
    echo ""
    
    # Run the normalizer
    if python "$NORMALIZER_SCRIPT" "$input_file" "$output_file"; then
        echo ""
        echo "   Success!"
        ((count++))
        sleep 0.5
    else
        echo ""
        echo "   Error normalizing file"
        ((errors++))
        sleep 2
    fi
    
done < <(find "$INFERENCE_DIR"/inference_checkpoints_* -name "predictions_final_*.jsonl" ! -name "*_NORMALIZED.jsonl" | sort)

#clear
echo "=========================================="
echo "Normalization complete!"
echo "=========================================="
echo "Total files found: $total_files"
echo "Successfully processed: $count files"
echo "Skipped (already normalized): $skipped files"
echo "Errors: $errors files"
echo ""
