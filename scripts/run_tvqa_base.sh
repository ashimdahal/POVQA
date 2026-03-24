#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ============================================================
# TVQA BASE MODEL (no adapters), VAL-ONLY 5k subset
# ============================================================

# ---- Paths (edit if needed) ----
TVQA_RAW_DIR="input_data"                                     # has: frames_hq/, tvqa_subtitles/, tvqa_qa_release/
PROCESSED_DIR="processed_tvqa"                                # processed clips live here (already created earlier)
VAL_JSONL_SRC="${TVQA_RAW_DIR}/tvqa_qa_release/tvqa_val.jsonl"
mkdir -p "${PROCESSED_DIR}"

# ---- Subset params ----
SUBSET_N=5000
SUBSET_SEED=42
VAL_JSONL="${PROCESSED_DIR}/val_${SUBSET_N}_seed${SUBSET_SEED}.jsonl"

# ---- Eval entrypoint ----
EVAL_PY="scripts/chain_of_thoughts/generate_synthetic_tvqa.py"

# ---- Imports path ----
export PYTHONPATH="${PYTHONPATH:-.}:."

# ---- Methods to eval ----
METHODS=(blend_blur_with_last_frame) 

# ---- Output root (BASE) ----
OUT_BASE="runs/base_tvqa"
mkdir -p "${OUT_BASE}"

# ---- Model / generation constants ----
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

MAX_FRAMES=3
FRAME_SELECTION="uniform"

USE_4BIT="--use_4bit"
MAX_NEW_TOKENS=512
TEMPERATURE=0
DO_SAMPLE=""
LIMIT=""
PAR_EVAL_JOBS=1  # increase if your GPU can handle multiple loads

echo "================================================="
echo "TVQA BASE (no adapters) on stratified ${SUBSET_N} val subset (seed=${SUBSET_SEED})"
echo "Raw: ${TVQA_RAW_DIR}"
echo "Val src: ${VAL_JSONL_SRC}"
echo "Val subset: ${VAL_JSONL}"
echo "Methods: ${METHODS[*]}   Frames: ${MAX_FRAMES}   Selection: ${FRAME_SELECTION}"
echo "Output root: ${OUT_BASE}"
echo "================================================="

export VAL_JSONL_SRC
export VAL_JSONL
export SUBSET_SEED
export SUBSET_N

# ------------------------------------------------
# [0] Build a stratified 5k subset (by show_name)
# ------------------------------------------------
python - <<'PY'
import os, json, random, collections, pathlib
src = os.environ["VAL_JSONL_SRC"]
dst = os.environ["VAL_JSONL"]
random.seed(int(os.environ.get("SUBSET_SEED","42")))
N = int(os.environ.get("SUBSET_N","5000"))

rows = [json.loads(l) for l in open(src, "r", encoding="utf-8") if l.strip()]
buckets = collections.defaultdict(list)
for r in rows:
    buckets[r["show_name"]].append(r)

tot = len(rows)
alloc = {}
remaining = N
for show, items in buckets.items():
    q = round(N * len(items) / tot)
    k = min(int(q), len(items))
    alloc[show] = k
    remaining -= k

shows = list(buckets.keys())
i = 0
while remaining > 0 and shows:
    s = shows[i % len(shows)]
    if alloc[s] < len(buckets[s]):
        alloc[s] += 1
        remaining -= 1
    i += 1

subset = []
stats = {}
for s, items in buckets.items():
    k = alloc.get(s, 0)
    pick = random.sample(items, k) if k > 0 else []
    subset.extend(pick)
    stats[s] = {"pool": len(items), "take": k}

random.shuffle(subset)

pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r) + "\n")

with open(dst + ".stats.json", "w", encoding="utf-8") as f:
    json.dump({"total_pool": tot, "target_n": N, "by_show": stats}, f, indent=2)
with open(dst + ".qids.txt", "w", encoding="utf-8") as f:
    for r in subset:
        f.write(str(r.get("qid")) + "\n")

print(f"[subset] wrote {len(subset)} → {dst}")
PY

# ------------------------------------------------
# [1] EVALUATE BASE MODEL ONLY (no adapters)
# ------------------------------------------------
echo ">>> BASE MODEL <<<"

launch_eval_base() {
  local METHOD="$1"
  local OUT_DIR_METHOD="${OUT_BASE}/${METHOD}"
  mkdir -p "${OUT_DIR_METHOD}"
  local OUT_FILE="${OUT_DIR_METHOD}/tvqa_base_${METHOD}_val${SUBSET_N}_59f_${FRAME_SELECTION}.jsonl"

  python "${EVAL_PY}" \
    --tvqa_root "${PROCESSED_DIR}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_file "${OUT_FILE}" \
    --model_name_or_path "${BASE_MODEL}" \
    --method "${METHOD}" \
    --interleave \
    --max_frames "${MAX_FRAMES}" \
    --frame_selection "${FRAME_SELECTION}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    ${DO_SAMPLE} \
    ${USE_4BIT} \
    ${LIMIT}

  echo "Summary: ${OUT_FILE%.jsonl}.summary.json"
}

eval_pids=()
for METHOD in "${METHODS[@]}"; do
  echo "--- BASE × ${METHOD} (eval) ---"
  launch_eval_base "${METHOD}" & eval_pids+=($!)
  if [ ${#eval_pids[@]} -ge ${PAR_EVAL_JOBS} ]; then
    wait -n
    tmp=()
    for pid in "${eval_pids[@]}"; do kill -0 "$pid" 2>/dev/null && tmp+=("$pid") || true; done
    eval_pids=("${tmp[@]}")
  fi
done
wait

# ------------------------------------------------
# [2] Tiny accuracy vector for quick sanity
# ------------------------------------------------
python - <<'PY'
import os, json, glob
roots = glob.glob("runs/base_tvqa/*/*.summary.json")
cols = ["blend_blur_with_last_frame","weighted_average","weighted_average_exponential","weighted_average_ramp"]
accs = {}
for p in roots:
    # runs/base_tvqa/<eval_method>/tvqa_base_<method>_val5000_59f_uniform.summary.json
    parts = p.split("/")
    if len(parts) < 3: 
        continue
    eval_m = parts[2]
    try:
        with open(p, "r") as f:
            s = json.load(f)
        accs[eval_m] = s["metrics"]["Accuracy"]
    except Exception:
        pass

print("\n== BASE accuracy by eval method on val5k ==")
for c in cols:
    v = accs.get(c, float('nan'))
    print(f"{c}\t{v:.3f}")
PY

echo "================================================="
echo "DONE. Outputs:"
echo "  ${OUT_BASE}/<eval_method>/*.jsonl (+ .summary.json)"
echo "Subset IDs: ${VAL_JSONL}.qids.txt   | Stats: ${VAL_JSONL}.stats.json"
echo "================================================="
