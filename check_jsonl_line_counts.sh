#!/bin/bash

# Set the folder path (change this as needed)
FOLDER="data/meta4xnli_ptbr/en2es"
VERBOSE=1

# Loop through all .jsonl files in the folder
for file in "$FOLDER"/*.jsonl; do
    filename=$(basename -- "$file")

    # Extract the expected number of lines from the filename
    # Format: <prefix>_<expected_lines>.jsonl
    if [[ "$filename" =~ _([0-9]+)\.jsonl$ ]]; then
        expected_lines="${BASH_REMATCH[1]}"
        
        # Count actual number of lines in the file
        actual_lines=$(wc -l < "$file")

        if [[ ! "$actual_lines" -eq "$expected_lines" ]]; then
            
            echo "❌ $filename: MISMATCH - Expected $expected_lines, Found $actual_lines"
        elif [ $VERBOSE -eq 1 ]; then
            echo "✅ $filename: OK ($actual_lines lines)"
        fi
    else
        echo "⚠️  $filename: Filename does not match expected pattern"
    fi
done
