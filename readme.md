# POVQA: Preference-Optimized Video Question Answering (ReasonVQA & TVQA)

> Accepted to the MAR Workshop at CVPR 2026.

<p align="center">
  <a href="https://arxiv.org/abs/2510.01009"><strong>Paper</strong></a> |
  <a href="https://povqa.github.io"><strong>Project Website</strong></a>
</p>

This repository contains the code and runnable scripts for **POVQA**, a preference-optimized framework for video QA that combines **temporal pooling** with **rationale supervision**. The repo is set up for fully reproducible runs of:

* **Supervised Fine-Tuning (SFT)** with QLoRA on interleaved frames + subtitles
* **Direct Preference Optimization (DPO)** on SFT-initialized policies
* **Cross-method evaluation** on ReasonVQA and a **5k stratified subset of TVQA**

The scripts are **self-contained**: they discover adapters on disk, mirror train/eval splits, and summarize outputs into JSON suitable for LaTeX table generation.

---

## Overview

* **Backbone**: `Qwen/Qwen2.5-VL-7B-Instruct` in **4-bit** (QLoRA)
* **Temporal evidence**: 4 pooling strategies

  * `blend_blur_with_last_frame (BBLF)`
  * `weighted_average (WA)`
  * `weighted_average_exponential (WAE)`
  * `weighted_average_ramp (WAR)`
* **Context shaping** (ReasonVQA): up to **59 frames + 1 keyframe (+hint)** at eval; **16 frames** at train; **interleaved** with nearest subtitles
* **Projector adaptation**: optional LoRA on multimodal projector (`--lora_target_mm_projector`)
* **SFT**: trains method-specific LoRA adapters under `models/sft-qwen7b-interleaved-16f/<method>`
* **DPO**: initializes policy from SFT adapter, uses **frozen reference** (base + SFT adapter), outputs under `models/dpo-qwen7b-interleaved-16f/<method>`
* **Evaluation**: base model + all adapters over all methods; TVQA **val-only stratified 5k** (`val_5000_seed42.jsonl`)
* **Outputs**: `.jsonl` generations + `.summary.json` metrics per run; LaTeX table generators provided

---

## Repository Layout (relevant to runs)

```
scripts/
├─ run_sft.sh                 # Train SFT adapters for all methods
├─ run_sft_eval.sh            # Evaluate base + SFT adapters across methods
├─ run_dpo.sh                 # DPO training for all methods (policy init = SFT)
├─ run_dpo_eval.sh            # Evaluate DPO adapters across methods
├─ run_tvqa_eval.sh           # TVQA val-only, DPO-only, stratified 5k subset

├─ chain_of_thoughts/
│  ├─ generate_synthetic_movies.py  # ReasonVQA generator + evaluator
│  ├─ generate_synthetic_tvqa.py    # TVQA generator + evaluator

├─ preprocessing/
│  ├─ video_preprocessing.py        # ReasonVQA preprocessing utilities
│  ├─ tvqa_processing.py            # TVQA processing utilities

├─ train/
│  ├─ sft_train.py                  # SFT entrypoint (QLoRA)
│  ├─ dpo_train.py                  # DPO entrypoint (policy + frozen ref)

└─ visualize/
   ├─ generate_latex_table_from_metrics.py
   ├─ generate_latex_ablation.py
   ├─ generate_latex_delta.py
   ├─ qualitative.py
   └─ qualitative_tvqa.py
```

---

## Data & Expected Folders

* **ReasonVQA (in-house)**

  * `annotations/` — JSON/JSONL annotations (Q/A + reasoning, timestamps)
  * `out_preprocessed/<movie>/<method>/frame_*.png` — 59 frames per clip + keyframe
  * Subtitles aligned at sentence/phrase granularity (nearest to frames)

* **TVQA (public)**

  * `input_data/`

    * `frames_hq/` (pre-extracted frame folders)
    * `tvqa_subtitles/` (ASR/subtitles per clip)
    * `tvqa_qa_release/` (contains `tvqa_val.jsonl`)
  * `processed_tvqa/` (created by `run_tvqa_eval.sh`):

    * `val_5000_seed42.jsonl` (+ `.stats.json`, `.qids.txt`) after stratified sampling

**Note on splits:** For ReasonVQA training, a **movie-level split** is created via `--split_ratio 0.9` with `--seed 42`, mirrored in eval to avoid leakage.

---

## Temporal Pooling & Context Shaping

We evaluate 4 frame pooling methods:

1. **BBLF**: blend a temporal blur with the **last frame** (key pose anchoring).
2. **WA**: uniform weighted average of frames.
3. **WAE**: exponential decay weights (recent frames emphasized).
4. **WAR**: linear ramp weights (recent frames emphasized).

**Key design choices**

* **Interleaving (`--interleave`)**: we insert each selected frame followed by its **nearest** subtitle snippet—improves temporal coherence for LVLMs.
* **Keyframe**: we **append the paused/annotated keyframe** (`--append_keyframe`) and insert a **textual hint** (`--keyframe_hint`) that grounds the question in the user’s paused moment.
* **Projector LoRA**: enabling `--lora_target_mm_projector` adapts the visual projector jointly with the language adapter.

---

## Environment & Hardware

* Python 3.10; PyTorch w/ CUDA; HF `transformers`/`peft`/`accelerate` stack
* **QLoRA (4-bit)** to fit **Qwen2.5-VL-7B-Instruct** on a single high-VRAM GPU
* Typical training runs used **BF16**, **gradient checkpointing**, and **grad accumulation** to manage memory
* For evaluation speed, you may set `MAX_NEW_TOKENS` lower (e.g., 128–512) for TVQA

---

## Reproducibility Settings

* Global seed: `--seed 42`
* ReasonVQA **train/eval** split mirrored across SFT/DPO and their eval scripts
* TVQA: deterministic **5k stratified** subset by `show_name` (`seed=42`) with allocation stats saved to disk

---

## How to Run (Minimal End-to-End)

### 0) Verify data layout

* ReasonVQA:

  * `annotations/` and `out_preprocessed/…` exist and match the four method names.
* TVQA:

  * `input_data/tvqa_qa_release/tvqa_val.jsonl` exists
  * (Optional) `frames_hq/` + `tvqa_subtitles/` for richer context

### 1) Train SFT adapters (ReasonVQA)

```bash
./scripts/run_sft.sh
```

* Produces: `models/sft-qwen7b-interleaved-16f/<method>/`
* Key flags inside:

  * `MAX_FRAMES=16`, `MAX_SEGMENTS=2048` (subtitle cap), `--interleave`, `--append_keyframe`, `--keyframe_hint`, `--lora_target_mm_projector`, `--bf16`, `--gradient_checkpointing`
* **Implementation note**: we call Python as a **module** (`python -m scripts.train.sft_train`) to make relative imports robust.

### 2) Evaluate base + SFT adapters (ReasonVQA)

```bash
./scripts/run_sft_eval.sh
```

* Base model (no adapter) + every SFT adapter over all four methods
* Eval context: **59 frames + keyframe (+hint)**, `--use_4bit`, `--interleave`
* Outputs:

  * Base: `runs/base-qwen7b_59f_plus_keyframe/<method>/*.jsonl(.summary.json)`
  * SFT:  `runs/sft-qwen7b-interleaved-16f_59f_plus_keyframe/<adapter>/<method>/*.jsonl(.summary.json)`

### 3) Train DPO policies (ReasonVQA)

```bash
./scripts/run_dpo.sh
```

* Produces: `models/dpo-qwen7b-interleaved-16f/<method>/`
* **Reference model**: **frozen** base **+ SFT adapter** (method-matched)

  * `--ref_model_name_or_path` = base; `--ref_peft_adapter` = SFT adapter dir
* **Policy init**: `--sft_adapter` points to the same SFT adapter for warm-start
* Key DPO settings: `BETA=0.3`, `LEARNING_RATE=5e-6`, `NUM_EPOCHS=1`, correctness-only negatives (`--correctness_only`)

### 4) Evaluate DPO policies (ReasonVQA)

```bash
./scripts/run_dpo_eval.sh
```

* Discovers adapters dynamically under `models/dpo-qwen7b-interleaved-16f/`
* Eval context: **59 frames + keyframe (+hint)**, `--use_4bit`, `--interleave`
* Outputs: `runs/dpo-qwen7b-interleaved-16f_59f_plus_keyframe/<adapter>/<method>/*.jsonl(.summary.json)`

### 5) Evaluate DPO on TVQA (val-only 5k stratified)

```bash
./scripts/run_tvqa_eval.sh
```

* Step \[0]: builds `processed_tvqa/val_5000_seed42.jsonl` with stratified sampling by show, plus stats
* Step \[1]: evaluates **DPO adapters only** over four methods (unified eval context)
* Step \[2]: prints a **tiny accuracy grid** (train × eval method) for quick sanity check
* Outputs: `runs/tvqa_dpo_val5k_59f_uniform/<adapter>/<method>/*.jsonl(.summary.json)`

---

## Metrics, Summaries & Tables

Every generation script writes:

* `*.jsonl` — one JSON per example with prompts, model output, (optionally) chain-of-thought/rationale, final answers
* `*.summary.json` — aggregated metrics for the file (EM/F1/BLEU/ROUGE-L/embedding metrics, and **Accuracy** for TVQA)

**LaTeX helpers** (under `visualize/`):

* `generate_latex_table_from_metrics.py` — load multiple `.summary.json` and produce **leaderboards** (bold maxima, method buckets, etc.)
* `generate_latex_ablation.py`, `generate_latex_delta.py` — ablation-style tables and delta tables
* `qualitative.py`, `qualitative_tvqa.py` — produces qualitative panels (model outputs, human refs, options block inline in the figure caption/section, etc.)

> Tip: The table generators expect keys like `EmbedCos` and `EmbedCos_Reasoning`. The scripts already align to those names.

---

## Key Flags (What They Do)

* `--interleave`
  Interleave each selected frame with its nearest subtitle snippet for temporally grounded context.

* `--append_keyframe` + `--keyframe_hint`
  Append the user’s paused frame and insert a brief textual hint referencing it (helps localize the question + answer).

* `--lora_target_mm_projector`
  Apply LoRA to the **multimodal projector** in addition to the language layers (improves fusion).

* `--length_normalize` (DPO)
  Normalizes sequence-length effects during preference loss.

* `--use_4bit`
  Load the backbone with 4-bit quantization to fit on a single GPU.

* `--max_frames` vs `--max_frames_train`
  Train with fewer frames (e.g., 16) for efficiency; evaluate with richer context (e.g., 59 + keyframe).

* `--max_segments(_train)`
  Cap the number of subtitle fragments (we default to 2048; set higher to allow more text).

---

## Repro Notes & Determinism

* **Splits:** ReasonVQA `--split_ratio 0.9`, `--seed 42` are used everywhere and mirrored at eval.
* **TVQA:** The 5k subset is **stratified by show\_name** with deterministic seeding. Allocation stats + selected QIDs are written beside the subset file.
* **Relative imports:** We always invoke training modules via `python -m` to avoid import errors.

---

## Troubleshooting & Tips

* **It runs but is slow**:

  * Reduce `MAX_NEW_TOKENS` (e.g., 128–512) on eval scripts; keep `TEMPERATURE=0` for determinism.
  * Use `--limit 100` during smoke tests to validate the pipeline quickly.

* **LoRA loading / reference wiring**:

  * DPO uses **base as ref backbone**, and if `REF_FROM_SFT=1`, we attach the **frozen SFT adapter** to the reference.
  * Policy starts from the **SFT adapter** (passed via `--sft_adapter`) and is updated by DPO.

* **Out of memory**:

  * Keep `--use_4bit`, `--bf16`, and `--gradient_checkpointing` on; reduce `BATCH_SIZE` or increase `GRAD_ACCUM_STEPS`.

* **Accidentally interrupted training**:

  * Scripts write into method-scoped directories. Relaunching will **reuse existing folders**; delete or rename a run dir to start clean.

---

## Results Surfaces (what to look for)

* **Cross-method tables** (base vs SFT vs DPO) under `runs/*_59f_plus_keyframe/…`
* **TVQA sanity grid** (train × eval methods) printed at the end of `run_tvqa_eval.sh`
* **Qualitative figures** built by `visualize/qualitative*.py` that include an options table **inside the human reference section** for a more compact layout

---

## Ethical & Data Notes

* ReasonVQA content is curated for **research**; subtitles are aligned and truncated for fair-use academic evaluation.
* TVQA val subset uses the official release; our pipeline **does not** modify QA content, only frames/subtitles selection for context shaping.

---

## Citation

If you find this repo helpful, please cite the **arXiv version** for now. We can update this section once the workshop/proceedings citation is available.

```
@article{dahal2025povqa,
  title   = {POVQA: Preference-Optimized Video Question Answering with Rationales for Data Efficiency},
  author  = {Dahal, Ashim and Ghimire, Ankit and Murad, Saydul Akbar and Rahimi, Nick},
  journal = {arXiv preprint arXiv:2510.01009},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.01009}
}
```

Paper: https://arxiv.org/abs/2510.01009

Project website: https://povqa.github.io

---

## FAQ

**Q: Why 16 frames for training but 59 + keyframe at eval?**
A: To keep training efficient while evaluating with richer temporal evidence. Empirically, SFT/DPO generalize to denser eval contexts.

**Q: What exactly is optimized in DPO?**
A: We optimize **final-answer tokens** with **correctness-only negatives**. Rationale text is used as supervision in SFT; DPO focuses the policy on choosing correct final answers.

**Q: Where are method-best scores pulled from for tables?**
A: `visualize/generate_latex_table_from_metrics.py` scans `runs/sft-*` and `runs/dpo-*` and reports **per-method maxima** across SFT or DPO, depending on which is higher for that method.

---

## Reproduction Checklist

* [ ] Confirm `annotations/` and `out_preprocessed/` exist for ReasonVQA
* [ ] Confirm `input_data/tvqa_qa_release/tvqa_val.jsonl` exists for TVQA
* [ ] `./scripts/run_sft.sh` → produces SFT adapters
* [ ] `./scripts/run_sft_eval.sh` → base + SFT results under `runs/`
* [ ] `./scripts/run_dpo.sh` → produces DPO adapters (policy)
* [ ] `./scripts/run_dpo_eval.sh` → DPO results under `runs/`
* [ ] `./scripts/run_tvqa_eval.sh` → DPO on TVQA val5k + accuracy grid
* [ ] `visualize/*.py` → LaTeX tables & qualitative figures

---

If anything is unclear or you’d like an **ablation-first** quickstart (e.g., BBLF only), here’s a one-liner you can run from repo root to train + eval just that method:

```bash
METHODS=(blend_blur_with_last_frame) ./scripts/run_sft.sh && \
METHODS=(blend_blur_with_last_frame) ./scripts/run_sft_eval.sh && \
METHODS=(blend_blur_with_last_frame) ./scripts/run_dpo.sh && \
METHODS=(blend_blur_with_last_frame) ./scripts/run_dpo_eval.sh
```

Thanks for reviewing!
