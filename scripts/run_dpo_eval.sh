#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ============================
# Evaluate DPO LoRA adapters
# on 59 method frames + 1 keyframe (+hint)
# over the SAME EVAL SPLIT used in training.
# (No base-model eval here.)
# ============================

# ---- USER CONFIG ----
ROOT_DIR="./"                                   # has annotations/ and out_preprocessed/
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Your 4 preprocessing methods (present in your DPO dir)
METHODS=(
  "blend_blur_with_last_frame"
  "weighted_average"
  "weighted_average_exponential"
  "weighted_average_ramp"
)

# DPO adapters root (each subdir is one adapter trained with DPO)
ADAPTER_ROOT="models/dpo-qwen7b-interleaved-16f"

# Output root
OUT_DPO="${ROOT_DIR}/runs/dpo-qwen7b-interleaved-16f_59f_plus_keyframe"

# Frame/text shaping
MAX_FRAMES=59
FRAME_SELECTION="uniform"
SEGS_PER_FRAME=1
SEG_RADIUS=2.0
KEYFRAME_FLAGS=(--append_keyframe --keyframe_hint)

# Generation
USE_4BIT="--use_4bit"
MAX_NEW_TOKENS=4096
TEMPERATURE=0
DO_SAMPLE=""   # set to "--do_sample" if you want sampling

# Eval split (MIRROR TRAINING)
SPLIT="eval"        # all | train | eval
SPLIT_RATIO=0.9
SEED=42

# Optional quick test limiter
LIMIT=""            # e.g., "--limit 50"

# ============================

mkdir -p "${OUT_DPO}"

echo "================================================="
echo "DPO Eval plan: 59 frames + keyframe (+hint)"
echo "Split: ${SPLIT} (ratio=${SPLIT_RATIO}, seed=${SEED})"
echo "Methods: ${METHODS[*]}"
echo "DPO adapters root: ${ADAPTER_ROOT}"
echo "Out: ${OUT_DPO}"
echo "================================================="
echo ""

# ---------------------------------------
# DPO ADAPTERS: each adapter × methods
# ---------------------------------------
echo ">>> DPO ADAPTERS <<<"

# Collect adapter dirs dynamically
mapfile -t ADAPTERS < <(find "${ADAPTER_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort || true)

if (( ${#ADAPTERS[@]} == 0 )); then
  echo "[ERROR] No adapters found in ${ADAPTER_ROOT}."
  echo "Make sure your DPO outputs exist like:"
  echo "  ${ADAPTER_ROOT}/blend_blur_with_last_frame"
  echo "  ${ADAPTER_ROOT}/weighted_average"
  echo "  ${ADAPTER_ROOT}/weighted_average_exponential"
  echo "  ${ADAPTER_ROOT}/weighted_average_ramp"
  exit 1
fi

echo "Found adapters: ${ADAPTERS[*]}"
echo ""

for AD in "${ADAPTERS[@]}"; do
  echo "== Adapter: ${AD} =="
  AD_PATH="${ADAPTER_ROOT}/${AD}"

  for METHOD in "${METHODS[@]}"; do
    echo "--- ${AD} on method: ${METHOD} ---"
    OUT_DIR_METHOD="${OUT_DPO}/${AD}/${METHOD}"
    mkdir -p "${OUT_DIR_METHOD}"
    OUT_FILE="${OUT_DIR_METHOD}/dpo_${AD}_${METHOD}_59f_plus_kf.jsonl"

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

echo "================================================="
echo "DONE. Outputs:"
echo "  DPO: ${OUT_DPO}/<adapter>/<method>/*.jsonl (+ .summary.json)"
echo "Merge these for your DPO mega table."
echo "================================================="
