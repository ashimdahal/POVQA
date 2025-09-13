#!/bin/bash

# =================================================================
# Batch Runner for MovieVQA Experiments
#
# This script automates the process of running the Python script
# across multiple preprocessing methods (e.g., your blurring techniques).
# =================================================================

# --- [ USER CONFIGURATION ] ---
# Please edit the variables in this section to match your setup.

# 1. List of preprocessing methods to test.
#    These should EXACTLY match the folder names in your
#    `out_preprocessed/<movie>/` directories.
METHODS=(
    "blend_blur_with_last_frame"
    "weighted_average"
    "weighted_average_exponential"
    "weighted_average_ramp"
    # Add any other custom method names you have here, each in quotes
)

# 2. Path to your project's root directory.
#    (e.g., "/home/user/my_movie_project")
ROOT_DIR="./"

# 3. Hugging Face model ID you want to use for the experiments.
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# 4. Common experimental parameters.
MAX_FRAMES=59
# --- [ END OF CONFIGURATION ] ---


# Create the output directory if it doesn't exist
mkdir -p "$ROOT_DIR/runs"

# Get the total number of methods to run for progress tracking
TOTAL_METHODS=${#METHODS[@]}
CURRENT_METHOD_NUM=1

echo "================================================="
echo "Starting Batch Experiment Run"
echo "Model: $MODEL_NAME"
echo "Frames per Question: $MAX_FRAMES"
echo "Total Methods to Test: $TOTAL_METHODS"
echo "================================================="
echo ""

# Loop through each method in the METHODS array
for METHOD in "${METHODS[@]}"
do
    echo "--- [ Running Experiment ($CURRENT_METHOD_NUM/$TOTAL_METHODS): $METHOD ] ---"

    # Construct a descriptive output filename for this specific run
    # Example: runs/qwen7b_interleaved_60frames_weighted_average.jsonl
    OUTPUT_FILE="$ROOT_DIR/runs/qwen7b_interleaved_${MAX_FRAMES}frames_${METHOD}.jsonl"

    echo "Output will be saved to: $OUTPUT_FILE"
    echo ""

    # The main command to execute your Python script with the correct parameters
    python ./scripts/chain_of_thoughts/generate_synthetic_movies.py \
      --root_dir "$ROOT_DIR" \
      --output_file "$OUTPUT_FILE" \
      --model_name_or_path "$MODEL_NAME" \
      --pooling "method" \
      --method "$METHOD" \
      --frame_selection "uniform" \
      --max_frames "$MAX_FRAMES" \
      --use_4bit \
      --interleave \
      --segs_per_frame 1 \
      --seg_radius 2.0

    # Check the exit code of the last command to see if it was successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "--- [ SUCCESS: Finished $METHOD ] ---"
        echo ""
    else
        echo ""
        echo "--- [ ERROR: The script failed for method: $METHOD ] ---"
        echo "Please check the output above for errors."
        echo "Aborting the rest of the runs."
        exit 1 # Exit the script immediately if any command fails
    fi

    # Increment the counter for the next loop
    ((CURRENT_METHOD_NUM++))
done

echo "================================================="
echo "All experiments have completed successfully!"
echo "Check the '$ROOT_DIR/runs/' directory for your results."
echo "================================================="
