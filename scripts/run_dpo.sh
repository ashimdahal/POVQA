#!/usr/bin/env bash
set -euo pipefail

# =================================================================
# DPO Runner for MovieVQA QLoRA (GENERAL, SELF-CONTAINED)
# Run from project root: ./scripts/run_dpo_all_methods.sh
# =================================================================

##########################
# User Configuration
##########################
ROOT_DIR="$(pwd)"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"   # policy base AND reference base
SFT_EXPERIMENT_NAME="sft-qwen7b-interleaved-16f"
SFT_BASE="$ROOT_DIR/models/$SFT_EXPERIMENT_NAME"

DPO_EXPERIMENT_NAME="dpo-qwen7b-interleaved-16f"
OUTPUT_BASE="$ROOT_DIR/models/$DPO_EXPERIMENT_NAME"

METHODS=(
  "blend_blur_with_last_frame"
  "weighted_average"
  "weighted_average_exponential"
  "weighted_average_ramp"
)

# Data shaping (match eval/infer)
MAX_FRAMES=16
MAX_SEGMENTS=2048
FRAME_SELECTION="uniform"   # near_ts | uniform | all
SPLIT_RATIO=0.9

# Booleans
APPEND_KEYFRAME=1
KEYFRAME_HINT=1
INTERLEAVE=1
ADAPT_PROJECTOR=1
BF16=1
GRADIENT_CHECKPOINTING=1
LENGTH_NORMALIZE=1
REF_FROM_SFT=1   # use SFT adapter as the FROZEN reference

# Training hyperparameters (DPO)
NUM_EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LEARNING_RATE=5e-6
WARMUP_RATIO=0.03
LORA_R=32
LORA_ALPHA=32
LORA_DROPOUT=0.05
BETA=0.3
SEED=42
MAX_LENGTH=4096
MAX_GRAD_NORM=1.0
LOGGING_STEPS=$GRAD_ACCUM_STEPS 
##########################
# Script Logic
##########################
echo "--- Starting DPO runs (all methods) ---"
echo "Root dir       : $ROOT_DIR"
echo "Backbone model : $MODEL_NAME"
echo "SFT base       : $SFT_BASE"
echo "DPO out base   : $OUTPUT_BASE"
echo "Methods        : ${METHODS[*]}"
echo "----------------------------------------"

build_flags() {
  [[ "$INTERLEAVE" -eq 1 ]] && echo "--interleave"
  [[ "$APPEND_KEYFRAME" -eq 1 ]] && echo "--append_keyframe"
  [[ "$KEYFRAME_HINT" -eq 1 ]] && echo "--keyframe_hint"
  [[ "$ADAPT_PROJECTOR" -eq 1 ]] && echo "--lora_target_mm_projector"
  [[ "$BF16" -eq 1 ]] && echo "--bf16"
  [[ "$GRADIENT_CHECKPOINTING" -eq 1 ]] && echo "--gradient_checkpointing"
  [[ "$LENGTH_NORMALIZE" -eq 1 ]] && echo "--length_normalize"
  # We explicitly optimize the Final Answer tokens and use correctness-only negatives:
  echo "--correctness_only"
}

for METHOD_NAME in "${METHODS[@]}"; do
  RUN_DIR="$OUTPUT_BASE/$METHOD_NAME"
  SFT_ADAPTER_DIR="$SFT_BASE/$METHOD_NAME"

  mkdir -p "$RUN_DIR"
  echo -e "\n>>> DPO Method: $METHOD_NAME"
  echo "    SFT adapter         : $SFT_ADAPTER_DIR"
  echo "    DPO output (policy) : $RUN_DIR"

  # Reference wiring:
  # - Always use the BASE as the ref backbone
  # - If REF_FROM_SFT=1, attach the method's SFT adapter to the FROZEN ref
  REF_BASE="$MODEL_NAME"
  REF_ADAPTER=""
  if [[ "$REF_FROM_SFT" -eq 1 ]]; then
    REF_ADAPTER="$SFT_ADAPTER_DIR"
    echo "    Reference = BASE + SFT adapter (frozen)"
  else
    echo "    Reference = BASE only (frozen)"
  fi

  # Run as a module so relative imports work.
  python -m scripts.train.dpo_train \
    --root_dir "$ROOT_DIR" \
    --output_dir "$RUN_DIR" \
    --model_name_or_path "$MODEL_NAME" \
    --ref_model_name_or_path "$REF_BASE" \
    ${REF_ADAPTER:+--ref_peft_adapter "$REF_ADAPTER"} \
    --sft_adapter "$SFT_ADAPTER_DIR" \
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
    --beta "$BETA" \
    --max_length "$MAX_LENGTH" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    $(build_flags)

  status=$?
  if [[ $status -eq 0 ]]; then
    echo "--- SUCCESS: DPO finished for '$METHOD_NAME' ---"
  else
    echo "--- FAILED: DPO for '$METHOD_NAME' exited with status $status ---" >&2
    # exit $status   # uncomment to stop at first failure
  fi
done

echo -e "\nAll DPO runs complete. Updated adapters are under: $OUTPUT_BASE"
