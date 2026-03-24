#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ============================================================
# TVQA DPO-only, VAL-ONLY 5k subset: no preprocessing here
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
METHODS=(blend_blur_with_last_frame weighted_average weighted_average_exponential weighted_average_ramp)

# ---- DPO adapters root (ONLY DPO) ----
ADAPTER_ROOT="models/dpo-qwen7b-interleaved-16f"

# ---- Output root ----
OUT_DPO="runs/tvqa_dpo_val5k_59f_uniform"
mkdir -p "${OUT_DPO}"

# ---- Model / generation constants (match your movie runner) ----
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

MAX_FRAMES=59
FRAME_SELECTION="uniform"

USE_4BIT="--use_4bit"
MAX_NEW_TOKENS=512   # TIP: set to 128 for ~10–20x faster + much lower VRAM
TEMPERATURE=0
DO_SAMPLE=""          # set to "--do_sample" if you want sampling

# Optional limiter for quick smoke tests (e.g., "--limit 250")
LIMIT=""

# ---- GPU concurrency ----
PAR_EVAL_JOBS=1       # keep 1 unless your GPU has room for multiple Qwen loads

echo "================================================="
echo "TVQA DPO-only on stratified ${SUBSET_N} val subset (seed=${SUBSET_SEED})"
echo "Raw: ${TVQA_RAW_DIR}"
echo "Val src: ${VAL_JSONL_SRC}"
echo "Val subset: ${VAL_JSONL}"
echo "Adapters root: ${ADAPTER_ROOT}"
echo "Methods: ${METHODS[*]}   Frames: ${MAX_FRAMES}   Selection: ${FRAME_SELECTION}"
echo "================================================="

export VAL_JSONL_SRC
export VAL_JSONL
export SUBSET_SEED
export SUBSET_N
# ------------------------------------------------
# [0] Build a stratified 5k subset (by show_name)
#     - writes: ${VAL_JSONL}
#     - writes: ${VAL_JSONL}.stats.json and .qids.txt
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
# proportional allocation
alloc = {}
remaining = N
for show, items in buckets.items():
    q = round(N * len(items) / tot)
    k = min(int(q), len(items))
    alloc[show] = k
    remaining -= k

# distribute leftover due to rounding
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

# write subset
pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r) + "\n")

# write stats + qids
with open(dst + ".stats.json", "w", encoding="utf-8") as f:
    json.dump({"total_pool": tot, "target_n": N, "by_show": stats}, f, indent=2)
with open(dst + ".qids.txt", "w", encoding="utf-8") as f:
    for r in subset:
        f.write(str(r.get("qid")) + "\n")

print(f"[subset] wrote {len(subset)} → {dst}")
PY

# ------------------------------------------------
# [1] EVALUATE DPO adapters only (val5k)
# ------------------------------------------------
echo ">>> DPO ADAPTERS <<<"

# Discover DPO adapters dynamically
mapfile -t ADAPTERS < <(find "${ADAPTER_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort || true)
if (( ${#ADAPTERS[@]} == 0 )); then
  echo "[ERROR] No adapters found in ${ADAPTER_ROOT}."
  echo "Expected: blend_blur_with_last_frame weighted_average weighted_average_exponential weighted_average_ramp"
  exit 1
fi
echo "Found DPO adapters: ${ADAPTERS[*]}"
echo ""

launch_eval() {
  local AD="$1"
  local METHOD="$2"
  local AD_PATH="${ADAPTER_ROOT}/${AD}"
  local OUT_DIR_METHOD="${OUT_DPO}/${AD}/${METHOD}"
  mkdir -p "${OUT_DIR_METHOD}"
  local OUT_FILE="${OUT_DIR_METHOD}/tvqa_dpo_${AD}_${METHOD}_val${SUBSET_N}_59f_${FRAME_SELECTION}.jsonl"

  python "${EVAL_PY}" \
    --tvqa_root "${PROCESSED_DIR}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_file "${OUT_FILE}" \
    --model_name_or_path "${BASE_MODEL}" \
    --peft_adapter "${AD_PATH}" \
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

# throttle GPU eval jobs
eval_pids=()
for AD in "${ADAPTERS[@]}"; do
  echo "== Adapter: ${AD} =="
  for METHOD in "${METHODS[@]}"; do
    echo "--- ${AD} (train) × ${METHOD} (eval) ---"
    launch_eval "$AD" "$METHOD" & eval_pids+=($!)
    if [ ${#eval_pids[@]} -ge ${PAR_EVAL_JOBS} ]; then
      wait -n
      tmp=()
      for pid in "${eval_pids[@]}"; do kill -0 "$pid" 2>/dev/null && tmp+=("$pid") || true; done
      eval_pids=("${tmp[@]}")
    fi
  done
done
wait

# ------------------------------------------------
# [2] Tiny accuracy grid for quick sanity
# ------------------------------------------------
python - <<'PY'
import os, json, glob
roots = glob.glob("runs/tvqa_dpo_val5k_59f_uniform/*/*/*.summary.json")
cols = ["blend_blur_with_last_frame","weighted_average","weighted_average_exponential","weighted_average_ramp"]
grid = {}  # train_method -> {eval_method: acc}
for p in roots:
    _,_,family,train_m,eval_m,_ = p.split("/", 6)  # runs/tvqa_dpo_val5k_59f_uniform/<train>/<eval>/*.summary.json
    with open(p, "r") as f:
        s = json.load(f)
    acc = s["metrics"]["Accuracy"]
    grid.setdefault(train_m, {})[eval_m] = acc

print("\n== Accuracy grid (train × eval) on val5k ==")
print("train\\eval\t" + "\t".join(cols))
for tr in cols:
    row = [f"{grid.get(tr, {}).get(c, float('nan')):.3f}" for c in cols]
    print(tr + "\t" + "\t".join(row))
PY

echo "================================================="
echo "DONE. Outputs:"
echo "  ${OUT_DPO}/<adapter>/<eval_method>/*.jsonl (+ .summary.json)"
echo "Subset IDs: ${VAL_JSONL}.qids.txt   | Stats: ${VAL_JSONL}.stats.json"
echo "================================================="
