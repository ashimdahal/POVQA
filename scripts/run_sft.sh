#!/usr/bin/env bash
set -euo pipefail

# =================================================================
# SFT Runner for MovieVQA QLoRA Fine-Tuning (GENERAL, SELF-CONTAINED)
# - Edit the config block below; then run from the project root:
#   ./scripts/run_sft_all_methods.sh
# =================================================================

##########################
# User Configuration
##########################
# Paths
ROOT_DIR="$(pwd)"                      # Assumes you run from the project root
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
EXPERIMENT_NAME="sft-qwen7b-interleaved-16f"
OUTPUT_BASE="$ROOT_DIR/models/$EXPERIMENT_NAME"

# Methods to iterate over (edit as needed)
METHODS=(
  "blend_blur_with_last_frame"
  "weighted_average"
  "weighted_average_exponential"
  "weighted_average_ramp"
)

# Data shaping
MAX_FRAMES=16
# CRITICAL: This now caps subtitles at 64. Change to 999 to use ALL available subtitles.
MAX_SEGMENTS=2048
FRAME_SELECTION="uniform" # near_ts | uniform | all
SPLIT_RATIO=0.9           # movie-level split: 90% train / 10% eval

# Booleans (1=yes, 0=no)
APPEND_KEYFRAME=1         # Append the user's paused frame to the context
KEYFRAME_HINT=1           # Add a text note about the paused keyframe
INTERLEAVE=1              # Interleave images with their nearest subtitle snippet
ADAPT_PROJECTOR=1         # Also apply LoRA to the visual projector (recommended)
BF16=1                    # Use bf16 if supported
GRADIENT_CHECKPOINTING=1  # Enable to save VRAM

# Training hyperparameters
NUM_EPOCHS=2
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LEARNING_RATE=5e-5
WARMUP_RATIO=0.08
LORA_R=32
LORA_ALPHA=32
LORA_DROPOUT=0.05
SEED=42
MAX_LENGTH=4096

##########################
# Script Logic
##########################
echo "--- Starting SFT runs (all methods) ---"
echo "Root dir       : $ROOT_DIR"
echo "Model          : $MODEL_NAME"
echo "Output base    : $OUTPUT_BASE"
echo "Methods to run : ${METHODS[*]}"
echo "----------------------------------------"

# Helper function to build boolean flags for the python script
build_flags() {
    [[ "$INTERLEAVE" -eq 1 ]] && echo "--interleave"
    [[ "$APPEND_KEYFRAME" -eq 1 ]] && echo "--append_keyframe"
    [[ "$KEYFRAME_HINT" -eq 1 ]] && echo "--keyframe_hint"
    [[ "$ADAPT_PROJECTOR" -eq 1 ]] && echo "--lora_target_mm_projector"
    [[ "$BF16" -eq 1 ]] && echo "--bf16"
    [[ "$GRADIENT_CHECKPOINTING" -eq 1 ]] && echo "--gradient_checkpointing"
}

for METHOD_NAME in "${METHODS[@]}"; do
  RUN_DIR="$OUTPUT_BASE/$METHOD_NAME"
  mkdir -p "$RUN_DIR"
  echo -e "\n>>> Starting Method: $METHOD_NAME"
  echo "    Output will be saved to: $RUN_DIR"

  # CRITICAL FIX: Run the training script AS A MODULE using `python -m`.
  # This makes the relative imports (`from ..`) work correctly.
  python -m scripts.train.sft_train \
    --root_dir "$ROOT_DIR" \
    --output_dir "$RUN_DIR" \
    --model_name_or_path "$MODEL_NAME" \
    --seed "$SEED" \
    --pooling "method" \
    --method "$METHOD_NAME" \
    --frame_selection "$FRAME_SELECTION" \
    --max_frames_train "$MAX_FRAMES" \
    --max_segments_train "$MAX_SEGMENTS" \
    --split_ratio "$SPLIT_RATIO" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --logging_steps 10 \
    --max_length "$MAX_LENGTH" \
    $(build_flags) # Add all boolean flags here

  status=$?
  if [[ $status -eq 0 ]]; then
    echo "--- SUCCESS: Finished method '$METHOD_NAME' ---"
  else
    echo "--- FAILED: Method '$METHOD_NAME' exited with status $status ---" >&2
    # Optional: exit the entire script if one method fails
    # exit $status
  fi
done

echo -e "\nAll method runs complete. Trained adapters are located under: $OUTPUT_BASE"
