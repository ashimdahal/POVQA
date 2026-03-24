#!/usr/bin/env python
"""
fine_tune_qwen2_5_vl_qlora.py
QLoRA SFT for Qwen/Qwen2.5-VL-7B on a Movie-VQA style dataset with two-line outputs.

Overview
--------
This script performs **Supervised Fine-Tuning (SFT)** of `Qwen/Qwen2.5-VL-7B-Instruct` using
**QLoRA** (LoRA on top of 4-bit quantization). It teaches the model to produce an answer in a
strict **two-line format**:

    Reasoning: <brief, grounded rationale citing frames/segments>
    Final Answer: <≤ 6 words, no punctuation>

The training samples are built from your existing preprocessing/artifacts (frames + aligned
subtitle segments) and your helper functions. Only the **assistant turn** (those two lines) is
supervised; the **user turn** (images + prompt text) is label-masked.

What this script does
---------------------
1) **Loads Qwen2.5-VL in 4-bit** (bitsandbytes, nf4) with `device_map="auto"` and optional bf16.
2) **Freezes vision components by default** to keep memory stable and focus on language-side
   reasoning (optionally you can adapt the multimodal projector via a flag).
3) **Applies LoRA** to common transformer linear layers (q/k/v/o, up/down/gate; discovered
   dynamically). Uses `prepare_model_for_kbit_training` to make QLoRA safe.
4) **Builds training conversations** per item using your helpers:
   - Selects N frames (`--max_frames_train`) and up to M subtitle segments (`--max_segments_train`)
   - Constructs either a *classic* prompt (all images then text) or an *interleaved* prompt
     (image → nearest subtitle snippet per frame).
   - Appends the **gold two-line assistant response** from annotations.
5) **Masks labels** so the loss is computed **only on the assistant two lines**; user/prefix tokens
   are set to `-100`.
6) **Trains with HuggingFace Trainer** (works with Accelerate / torchrun), keeping
   `remove_unused_columns=False` which is required for vision-language models.
7) **Saves a PEFT adapter** (LoRA weights) and the processor/tokenizer to `--output_dir`.

Expected data layout (same as your inference script)
----------------------------------------------------
<root_dir>/
  ├─ annotations/<movie>.json           # items with: timestamp, contextTimestamp, question, answer, reasoning, index
  └─ out_preprocessed/<movie>/
      ├─ KEY_FRAMES/
      ├─ <method folder>/               # e.g., blend_blur_with_last_frame, weighted_* (with saved chunks/images)
      ├─ metadata_text_centric.json
      └─ metadata_text_centric_<method>.json

Training objective (what is optimized)
--------------------------------------
Minimize cross-entropy only on the assistant block:

    Reasoning: {gold_reasoning or short fallback}
    Final Answer: {gold_answer}

This directly rewards concise, grounded reasoning and short final answers. Your separate evaluator
can still enforce post-rules (≤ 6 words / no punctuation) at validation time.

LoRA + Quantization details
---------------------------
- **Quantization**: 4-bit nf4 with double quant; compute dtype bf16 (if supported) or fp16.
- **Target modules**: detected linear layers in the transformer (q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj, qkv_proj). Optional flag to include the **mm projector**.
- **Vision tower**: frozen by default (lower VRAM, preserves visual features). Enable projector LoRA
  later if you want modest cross-modal alignment updates.

Memory & throughput tips (practical defaults)
---------------------------------------------
- You do **not** need 60 images per sample for SFT. Recommended to start with:
  - `--max_frames_train 3–6` (default 4) and `--max_segments_train 3–6` (default 4).
  - Prefer `--frame_selection near_ts` or `--interleave` to keep the context compact and salient.
- Use `--per_device_train_batch_size 1` with `--gradient_accumulation_steps 8` on 24–48 GB GPUs.
- If OOM: reduce `--max_frames_train` to 3, lower `--max_length` (e.g., 1536), and keep projector
  LoRA off.

Key arguments (subset)
----------------------
--root_dir                    Project root with `annotations/` and `out_preprocessed/`.
--output_dir                  Directory to save the LoRA adapter and processor.
--model_name_or_path          HF id or local path (default: Qwen/Qwen2.5-VL-7B-Instruct).
--pooling {keyframe,method}   Which frame source to use (matches your pipeline).
--method                      Method folder name if pooling=method (e.g., weighted_average_exponential).
--frame_selection             {near_ts,uniform,all} frame picking strategy.
--max_frames_train            Frames per sample (train-time). Default 4.
--max_segments_train          Subtitle segments per sample. Default 4.
--interleave                  If set, interleave each frame with its nearest subtitle snippet(s).
--append_keyframe             Optionally append the keyframe image.
--keyframe_hint               Optionally mention the paused keyframe in the prompt.
--lora_r, --lora_alpha,
--lora_dropout                LoRA hyperparameters. Defaults: r=16, α=16, p=0.05.
--lora_target_mm_projector    Also adapt the multimodal projector (optional, higher VRAM).
--learning_rate               Default 1e-4 for LoRA.
--num_train_epochs            Default 2 (tune to your dataset size).
--per_device_train_batch_size Default 1; pair with --gradient_accumulation_steps.
--bf16 / --gradient_checkpointing  Mixed precision & memory trade-offs.
--limit                       Use only the first N samples (debugging).

Usage examples
--------------
Single GPU (bf16 if available):
    python fine_tune_qwen2_5_vl_qlora.py \
      --root_dir /path/to/project \
      --output_dir runs/qlora_qwen2_5vl_sft_wexp_8f1kf \
      --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
      --pooling method \
      --method weighted_average_exponential \
      --frame_selection near_ts \
      --max_frames_train 8 \
      --append_keyframe \
      --keyframe_hint \
      --interleave \
      --max_segments_train 8 \
      --segs_per_frame 1 \
      --lora_target_mm_projector \
      --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --gradient_checkpointing \
      --bf16 \
      --max_length 2048

Accelerate (multi-GPU):
    accelerate launch --mixed_precision bf16 --num_processes 4 fine_tune_qwen2_5_vl_qlora.py \
      --root_dir /path/to/project --output_dir runs/qlora_qwen2_5vl_sft_accel \
      --pooling method --method weighted_average_exponential \
      --frame_selection near_ts --max_frames_train 4 --max_segments_train 4 \
      --interleave --gradient_checkpointing --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 2

Outputs
-------
- **PEFT adapter** (LoRA weights) saved to `--output_dir` (load with `PeftModel.from_pretrained` or
  `AutoPeftModelForCausalLM` workflows).
- **Processor/tokenizer** saved alongside (captures pad/eos fixes for safe inference).
- Trainer logs/checkpoints per `TrainingArguments`.

Notes & pitfalls
----------------
- Keep `remove_unused_columns=False` (vision-language models need the extra processor fields).
- We set pad token to eos if missing to prevent generation errors.
- During label masking, the script computes the boundary by re-rendering the user prefix with
  `add_generation_prompt=True`. This ensures only assistant tokens contribute to the loss.
- For best generalization, align your *training* prompt shape (interleaved vs. classic, frame/segment
  counts) with how you plan to run evaluation/inference.
"""

import os, json, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    Qwen2_5_VLForConditionalGeneration,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from scripts.chain_of_thoughts.generate_synthetic_movies import (
    hms_to_sec, parse_context_ts,
    resolve_metadata_file, extract_segments_for_window,
    collect_method_frames_with_times, collect_keyframe_by_index_with_time,
    build_two_step_messages_with_subs, build_two_step_messages_interleaved,
)
# -------------
# Utilities
# -------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def suggested_target_modules(model: nn.Module) -> List[str]:
    """Collect common proj/up/down module names present in Qwen2.5 blocks."""
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj"}
    present = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            for c in candidates:
                if n.endswith(c):
                    present.add(c)
    return sorted(present) if present else [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]

# -------------
# Dataset
# -------------

class MovieVQASFTDataset(Dataset):
    """
    Builds a single-turn chat per sample:
      [user (images + context text)] + [assistant (gold "Reasoning:\nFinal Answer:")]
    Labels mask user tokens as -100, supervise only assistant tokens.

    New: movie-level split via (split, split_ratio). Fixed seed 42.
    """

    def __init__(
        self,
        root_dir: str,
        processor: AutoProcessor,
        # NEW split args
        split: str = "train",  # {"train","eval"}
        split_ratio: float = 0.9,
        # Existing shaping args
        pooling: str = "method",
        method: str = "weighted_average_exponential",
        frame_selection: str = "near_ts",
        max_frames_train: int = 4,
        max_segments_train: int = 4,
        append_keyframe: bool = True,
        keyframe_hint: bool = False,
        interleave: bool = True,
        segs_per_frame: int = 1,
        seg_radius: float = 2.0,
        max_length: int = 2048,
        limit: int = None,
    ):
        assert split in {"train", "eval"}, "split must be 'train' or 'eval'"
        self.root = Path(root_dir)
        self.ann_dir = self.root / "annotations"
        self.out_root = self.root / "out_preprocessed"
        self.processor = processor

        self.split = split
        self.split_ratio = split_ratio

        self.pooling = pooling
        self.method = method
        self.frame_selection = frame_selection
        self.max_frames_train = max_frames_train
        self.max_segments_train = max_segments_train
        self.append_keyframe = append_keyframe
        self.keyframe_hint = keyframe_hint
        self.interleave = interleave
        self.segs_per_frame = segs_per_frame
        self.seg_radius = seg_radius
        self.max_length = max_length

        # --- REPRODUCIBLE MOVIE-LEVEL SPLIT ---
        all_ann_files = sorted(self.ann_dir.glob("*.json"))
        rnd = random.Random(42)
        rnd.shuffle(all_ann_files)
        split_idx = int(len(all_ann_files) * split_ratio)
        if split == "train":
            ann_files_for_split = all_ann_files[:split_idx]
            print(f"[INFO] Created TRAIN split with {len(ann_files_for_split)} movies.")
        else:
            ann_files_for_split = all_ann_files[split_idx:]
            print(f"[INFO] Created EVAL split with {len(ann_files_for_split)} movies.")

        # --- LOAD SAMPLES FOR SELECTED MOVIES ---
        self.samples: List[Tuple[str, Dict[str, Any]]] = []
        for ann_file in ann_files_for_split:
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception:
                continue
            movie = ann_file.stem
            movie_root = self.out_root / movie
            if not movie_root.exists():
                continue
            for it in items:
                # For eval, keep only items with both reasoning and answer (optional quality gate)
                if split == "eval" and not (it.get("reasoning") and it.get("answer")):
                    continue
                self.samples.append((movie, it))
                if limit and len(self.samples) >= limit:
                    break
            if limit and len(self.samples) >= limit:
                break

        print(f"[INFO] Loaded {len(self.samples)} samples for the {split} split.")

    def __len__(self):
        return len(self.samples)

    def _build_conv_and_images(self, movie: str, it: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        movie_root = self.out_root / movie
        keyframes_dir = movie_root / "KEY_FRAMES"

        # Resolve metadata (subtitles + method mapping)
        try:
            metadata_json = resolve_metadata_file(
                movie_root,
                self.method if self.pooling == "method" else None,
            )
        except Exception:
            metadata_json = movie_root / "metadata_text_centric.json"

        # Parse fields
        q = it["question"].strip()
        gold_ans = it["answer"].strip()
        gold_reason = (it.get("reasoning") or "").strip()
        ts_sec = hms_to_sec(it["timestamp"].strip())
        ctx_start, ctx_end = parse_context_ts(it["contextTimestamp"])
        idx = int(it["index"])

        # Subtitle segments
        segs = []
        if metadata_json.exists():
            segs = extract_segments_for_window(
                metadata_json, ctx_start, ctx_end, max_segments=self.max_segments_train
            )

        # Frames
        frame_paths: List[Path] = []
        frame_times: List[float] = []
        if self.pooling == "keyframe":
            frame_paths, frame_times = collect_keyframe_by_index_with_time(keyframes_dir, idx)
            if (not frame_paths) and self.method:
                if metadata_json.exists():
                    frame_paths, frame_times = collect_method_frames_with_times(
                        movie_root,
                        self.method,
                        metadata_json,
                        ts_sec,
                        ctx_start,
                        ctx_end,
                        max_frames=1,
                    )
        elif self.pooling == "method":
            if not self.method:
                raise ValueError("When pooling=method you must pass --method")
            frame_paths, frame_times = collect_method_frames_with_times(
                movie_root,
                self.method,
                metadata_json,
                ts_sec,
                ctx_start,
                ctx_end,
                max_frames=self.max_frames_train,
                frame_selection=self.frame_selection,
            )
        else:
            raise ValueError("pooling must be one of {keyframe, method}")

        keyframe_hint_text = ""
        if self.append_keyframe:
            kfp, kft = collect_keyframe_by_index_with_time(keyframes_dir, idx)
            if kfp:
                seen = {str(p) for p in frame_paths}
                if str(kfp[0]) not in seen:
                    frame_paths.append(kfp[0])
                    frame_times.append(kft[0] if (kft and kft[0] is not None) else None)
            if self.keyframe_hint and kfp:
                approx_t = (
                    f"{(kft[0] if (kft and kft[0] is not None) else 0.0):.1f}s"
                )
                keyframe_hint_text = (
                    f"Note: The user asked the question while paused at the keyframe {kfp[0].name} (t≈{approx_t})."
                )

        # Build user message + load images
        if not frame_paths:
            user_msgs = build_two_step_messages_with_subs(
                frame_paths=[],
                frame_times=[],
                question=q,
                subtitle_segments=[],
                keyframe_hint_text=keyframe_hint_text,
            )
            imgs: List[Image.Image] = []
        else:
            if self.interleave:
                user_msgs = build_two_step_messages_interleaved(
                    frame_paths=frame_paths,
                    frame_times=frame_times,
                    question=q,
                    subtitle_segments=segs,
                    keyframe_hint_text=keyframe_hint_text,
                    segs_per_frame=self.segs_per_frame,
                    seg_radius=self.seg_radius,
                )
            else:
                user_msgs = build_two_step_messages_with_subs(
                    frame_paths=frame_paths,
                    frame_times=frame_times,
                    question=q,
                    subtitle_segments=segs,
                    keyframe_hint_text=keyframe_hint_text,
                )
            imgs = []
            for item in user_msgs[0]["content"]:
                if item.get("type") == "image":
                    try:
                        imgs.append(Image.open(item["image"]).convert("RGB"))
                    except Exception:
                        pass

        # Assistant gold two-line answer
        gold_reason = gold_reason or "[Short rationale not provided]"
        two_line = f"Reasoning: {gold_reason}\nFinal Answer: {gold_ans}"
        conv = user_msgs + [{"role": "assistant", "content": [{"type": "text", "text": two_line}]}]
        return conv, imgs

    def __getitem__(self, i):
        movie, it = self.samples[i]
        conv, imgs = self._build_conv_and_images(movie, it)

        # 1) Build full conversation text and user-prefix text (same as before)
        text_full = self.processor.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False
        )
        text_user_prefix = self.processor.apply_chat_template(
            [conv[0]],
            tokenize=False,
            add_generation_prompt=True
        )

        # 2) Tokenize WITH the processor for BOTH prefix and full, passing the SAME images
        enc_full = self.processor(
            text=text_full, images=imgs,
            return_tensors="pt", padding=False, truncation=True, max_length=self.max_length
        )
        enc_pref = self.processor(
            text=text_user_prefix, images=imgs,
            return_tensors="pt", padding=False, truncation=True, max_length=self.max_length
        )

        # 3) Use processor-generated ids; DO NOT retokenize with plain tokenizer
        input_ids = enc_full["input_ids"][0]               # shape: [L]
        prefix_len = enc_pref["input_ids"].shape[1]        # number of tokens to mask

        # 4) Build labels: mask user/prefix; learn only assistant two lines
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        # 5) Return enc_full + labels (keep all vision tensors intact)
        enc_full["labels"] = labels.unsqueeze(0)
        return enc_full

# -------------
# Collator (pad per-batch via processor)
# -------------

class DataCollatorVL:
    def __init__(self, processor: AutoProcessor, pad_to_multiple_of: int = 8):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        keys = set().union(*[f.keys() for f in features])
        for k in keys:
            if k in ("input_ids", "attention_mask", "labels"):
                tensors = [f[k].squeeze(0) for f in features]
                pad_val = (
                    self.processor.tokenizer.pad_token_id if k != "labels" else -100
                )
                batch[k] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=pad_val
                )
                if self.pad_to_multiple_of:
                    L = batch[k].shape[1]
                    pad_len = (self.pad_to_multiple_of - (L % self.pad_to_multiple_of)) % self.pad_to_multiple_of
                    if pad_len:
                        pad = torch.full(
                            (batch[k].shape[0], pad_len), pad_val, dtype=batch[k].dtype
                        )
                        batch[k] = torch.cat([batch[k], pad], dim=1)
            else:
                vals = [f[k] for f in features if k in f]
                if len(vals) == 0:
                    continue
                batch[k] = torch.cat(vals, dim=0)
        return batch

# -------------
# Main
# -------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True, help="Project root with annotations/ and out_preprocessed/")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

    # Frame/text shaping for TRAINING (smaller than inference!)
    ap.add_argument("--pooling", type=str, choices=["keyframe", "method"], default="method")
    ap.add_argument("--method", type=str, default="weighted_average_exponential")
    ap.add_argument("--frame_selection", type=str, choices=["near_ts", "uniform", "all"], default="near_ts")
    ap.add_argument("--max_frames_train", type=int, default=4)
    ap.add_argument("--max_segments_train", type=int, default=4)
    ap.add_argument("--append_keyframe", action="store_true")
    ap.add_argument("--keyframe_hint", action="store_true")
    ap.add_argument("--interleave", action="store_true")
    ap.add_argument("--segs_per_frame", type=int, default=1)
    ap.add_argument("--seg_radius", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None)

    # Token/sequence
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)

    # QLoRA & training hyperparams
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_mm_projector",
        action="store_true",
        help="Also adapt the vision-language projector if present.",
    )

    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--bf16", action="store_true", help="Use bf16 if available.")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # Split control
    ap.add_argument("--split_ratio", type=float, default=0.9, help="Fraction of movies for training (movie-level split).")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 4-bit quant
    compute_dtype = (
        torch.bfloat16
        if (args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Tokenizer padding safety
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
        if getattr(model, "config", None) and model.config.pad_token_id is None:
            model.config.pad_token_id = tok.eos_token_id

    # Disable cache + optionally enable gradient ckpt
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze vision towers by default (keeps memory lower & focuses reasoning)
    for n, p in model.named_parameters():
        if any(k in n for k in ["vision_tower", "visual", "mm_vision", "image", "multi_modal"]):
            p.requires_grad = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    tmods = suggested_target_modules(model)
    if args.lora_target_mm_projector:
        tmods = list(sorted(set(tmods + ["mm_projector", "mm_projector.0", "mm_projector.2", "vision_tower.proj"])))

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tmods,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # =====================
    # Datasets + Collator
    # =====================
    train_ds = MovieVQASFTDataset(
        root_dir=args.root_dir,
        processor=processor,
        split="train",
        split_ratio=args.split_ratio,
        pooling=args.pooling,
        method=args.method,
        frame_selection=args.frame_selection,
        max_frames_train=args.max_frames_train,
        max_segments_train=args.max_segments_train,
        append_keyframe=args.append_keyframe,
        keyframe_hint=args.keyframe_hint,
        interleave=args.interleave,
        segs_per_frame=args.segs_per_frame,
        seg_radius=args.seg_radius,
        max_length=args.max_length,
        limit=args.limit,
    )
    eval_ds = MovieVQASFTDataset(
        root_dir=args.root_dir,
        processor=processor,
        split="eval",
        split_ratio=args.split_ratio,
        pooling=args.pooling,
        method=args.method,
        frame_selection=args.frame_selection,
        max_frames_train=args.max_frames_train,
        max_segments_train=args.max_segments_train,
        append_keyframe=args.append_keyframe,
        keyframe_hint=args.keyframe_hint,
        interleave=args.interleave,
        segs_per_frame=args.segs_per_frame,
        seg_radius=args.seg_radius,
        max_length=args.max_length,
        limit=args.limit,
    )

    collator = DataCollatorVL(processor)

    # =====================
    # Training arguments
    # =====================
    tr_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="epoch",      # evaluate each epoch
        save_strategy="epoch",            # save each epoch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,  # IMPORTANT for VL models
        per_device_eval_batch_size=1,
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # Debug prints
    dl = trainer.get_train_dataloader()
    print("Total samples in train_ds:", len(trainer.train_dataset))
    print("Total samples in eval_ds:", len(eval_ds))
    print("World size:", trainer.args.world_size)
    print("Per-device batch size:", trainer.args.per_device_train_batch_size)
    print("Batches/epoch (this process):", len(dl))
    print(
        "Optimizer updates/epoch (with grad acc):",
        (len(dl) + trainer.args.gradient_accumulation_steps - 1) // trainer.args.gradient_accumulation_steps,
    )

    trainer.train()

    # Save PEFT adapter and processor
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
