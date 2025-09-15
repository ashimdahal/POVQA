#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ============================
# Evaluate base model AND every LoRA adapter
# on 59 method frames + 1 keyframe (+hint)
# over the SAME EVAL SPLIT used in training.
# ============================

# ---- USER CONFIG ----
ROOT_DIR="./"                                   # has annotations/ and out_preprocessed/
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Your 5 preprocessing methods (folder names in out_preprocessed/<movie>/)
METHODS=(
  "blend_blur_with_last_frame"
  "weighted_average"
  "weighted_average_exponential"
  "weighted_average_ramp"
)

# LoRA adapters root (each subdir is one adapter)
ADAPTER_ROOT="models/sft-qwen7b-interleaved-16f"

# Output roots
OUT_BASE="${ROOT_DIR}/runs/base-qwen7b_59f_plus_keyframe"
OUT_LORA="${ROOT_DIR}/runs/sft-qwen7b-interleaved-16f_59f_plus_keyframe"

# Frame/text shaping
MAX_FRAMES=59
FRAME_SELECTION="uniform"
SEGS_PER_FRAME=1
SEG_RADIUS=2.0
KEYFRAME_FLAGS=(--append_keyframe --keyframe_hint)

# Generation
USE_4BIT="--use_4bit"
MAX_NEW_TOKENS=4096
TEMPERATURE=0.0
DO_SAMPLE=""   # set to "--do_sample" if you want sampling

# Eval split (MIRROR TRAINING)
SPLIT="eval"        # all | train | eval
SPLIT_RATIO=0.9
SEED=42

# Optional quick test limiter
LIMIT=""            # e.g., "--limit 50"

# ============================

mkdir -p "${OUT_BASE}" "${OUT_LORA}"

echo "================================================="
echo "Eval plan: 59 frames + keyframe (+hint)"
echo "Split: ${SPLIT} (ratio=${SPLIT_RATIO}, seed=${SEED})"
echo "Methods: ${METHODS[*]}"
echo "Base out : ${OUT_BASE}"
echo "LoRA out : ${OUT_LORA}"
echo "================================================="
echo ""

# ----------------------------
# 1) BASE MODEL over all methods
# ----------------------------
echo ">>> BASE MODEL <<<"
for METHOD in "${METHODS[@]}"; do
  echo "--- Base on method: ${METHOD} ---"
  OUT_DIR_METHOD="${OUT_BASE}/${METHOD}"
  mkdir -p "${OUT_DIR_METHOD}"
  OUT_FILE="${OUT_DIR_METHOD}/base_qwen7b_${METHOD}_59f_plus_kf.jsonl"

  python ./scripts/chain_of_thoughts/generate_synthetic_movies.py \
    --root_dir "${ROOT_DIR}" \
    --output_file "${OUT_FILE}" \
    --model_name_or_path "${BASE_MODEL}" \
    --pooling "method" \
    --method "${METHOD}" \
    --frame_selection "${FRAME_SELECTION}" \
    --max_frames "${MAX_FRAMES}" \
    --interleave \
    --segs_per_frame "${SEGS_PER_FRAME}" \
    --seg_radius "${SEG_RADIUS}" \
    "${KEYFRAME_FLAGS[@]}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    ${DO_SAMPLE} \
    ${USE_4BIT} \
    --split "${SPLIT}" \
    --split_ratio "${SPLIT_RATIO}" \
    --seed "${SEED}" \
    ${LIMIT}

  echo "Summary: ${OUT_FILE%.jsonl}.summary.json"
  echo ""
done

# ---------------------------------------
# 2) LoRA ADAPTERS: each adapter × methods
# ---------------------------------------
echo ">>> LoRA ADAPTERS <<<"
# Collect adapter dirs dynamically
mapfile -t ADAPTERS < <(find "${ADAPTER_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

if (( ${#ADAPTERS[@]} == 0 )); then
  echo "[WARN] No adapters found in ${ADAPTER_ROOT}. Skipping LoRA section."
else
  for AD in "${ADAPTERS[@]}"; do
    echo "== Adapter: ${AD} =="
    AD_PATH="${ADAPTER_ROOT}/${AD}"

    for METHOD in "${METHODS[@]}"; do
      echo "--- ${AD} on method: ${METHOD} ---"
      OUT_DIR_METHOD="${OUT_LORA}/${AD}/${METHOD}"
      mkdir -p "${OUT_DIR_METHOD}"
      OUT_FILE="${OUT_DIR_METHOD}/sft_${AD}_${METHOD}_59f_plus_kf.jsonl"

      python ./scripts/chain_of_thoughts/generate_synthetic_movies.py \
        --root_dir "${ROOT_DIR}" \
        --output_file "${OUT_FILE}" \
        --model_name_or_path "${BASE_MODEL}" \
        --peft_adapter "${AD_PATH}" \
        --pooling "method" \
        --method "${METHOD}" \
        --frame_selection "${FRAME_SELECTION}" \
        --max_frames "${MAX_FRAMES}" \
        --interleave \
        --segs_per_frame "${SEGS_PER_FRAME}" \
        --seg_radius "${SEG_RADIUS}" \
        "${KEYFRAME_FLAGS[@]}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        ${DO_SAMPLE} \
        ${USE_4BIT} \
        --split "${SPLIT}" \
        --split_ratio "${SPLIT_RATIO}" \
        --seed "${SEED}" \
        ${LIMIT}

      echo "Summary: ${OUT_FILE%.jsonl}.summary.json"
      echo ""
    done
  done
fi

echo "================================================="
echo "DONE. Outputs:"
echo "  Base: ${OUT_BASE}/<method>/*.jsonl (+ .summary.json)"
echo "  LoRA: ${OUT_LORA}/<adapter>/<method>/*.jsonl (+ .summary.json)"
echo "Merge away for your mega table!"
echo "================================================="
