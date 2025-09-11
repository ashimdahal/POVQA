#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QLoRA SFT for Qwen2.5-VL-7B (movie VQA two-line format)
- 4-bit nf4 quantization (bitsandbytes)
- LoRA on LM blocks (optionally mm projector)
- Supervised only on assistant's final two lines:
    "Reasoning: ..."
    "Final Answer: ..."
Assumes your preprocessing + helpers are available for import.

python fine_tune_qwen2_5_vl_qlora.py \
  --root_dir /path/to/project \
  --output_dir runs/qlora_qwen2_5vl_sft \
  --pooling method \
  --method weighted_average_exponential \
  --frame_selection near_ts \
  --max_frames_train 4 \
  --max_segments_train 4 \
  --interleave \
  --bf16 \
  --gradient_checkpointing \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 2
"""

import os, json, math, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

from ..chain_of_thoughts.generate_synthetic_movies import (
    hms_to_sec, parse_context_ts,
    resolve_metadata_file, extract_segments_for_window,
    collect_method_frames_with_times, collect_keyframe_by_index_with_time,
    build_two_step_messages_with_subs, build_two_step_messages_interleaved
)

# -------------
# Utilities
# -------------

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def suggested_target_modules(model: nn.Module) -> List[str]:
    """
    Dynamically collect common proj/up/down module names present in Qwen2.5 blocks.
    """
    candidates = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","qkv_proj"}
    present = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            for c in candidates:
                if n.endswith(c):
                    present.add(c)
    # fall back to a broad set if nothing matched (unlikely)
    return sorted(present) if present else ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# -------------
# Dataset
# -------------

class MovieVQASFTDataset(Dataset):
    """
    Builds a single-turn chat per sample:
      [user (images + context text)] + [assistant (gold "Reasoning:\nFinal Answer:")]
    Labels mask user tokens as -100, supervise only assistant tokens.
    """
    def __init__(
        self,
        root_dir: str,
        processor: AutoProcessor,
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
        self.root = Path(root_dir)
        self.ann_dir = self.root / "annotations"
        self.out_root = self.root / "out_preprocessed"
        self.processor = processor

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

        self.samples = []  # (movie, item dict)
        for ann_file in sorted(self.ann_dir.glob("*.json")):
            with open(ann_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            movie = ann_file.stem
            movie_root = self.out_root / movie
            if not movie_root.exists():
                continue
            for it in items:
                self.samples.append((movie, it))
                if limit and len(self.samples) >= limit:
                    break
            if limit and len(self.samples) >= limit:
                break

    def __len__(self):
        return len(self.samples)

    def _build_conv_and_images(self, movie: str, it: Dict[str, Any]) -> Tuple[List[Dict[str,Any]], List[Image.Image]]:
        movie_root = self.out_root / movie
        keyframes_dir = movie_root / "KEY_FRAMES"

        # Resolve metadata (subtitles + method mapping)
        try:
            metadata_json = resolve_metadata_file(
                movie_root,
                self.method if self.pooling == "method" else None
            )
        except Exception:
            metadata_json = movie_root / "metadata_text_centric.json"

        # Parse fields
        q = it["question"].strip()
        gold_ans = it["answer"].strip()
        gold_reason = (it.get("reasoning") or "").strip()
        ts_sec     = hms_to_sec(it["timestamp"].strip())
        ctx_start, ctx_end = parse_context_ts(it["contextTimestamp"])
        idx = int(it["index"])

        # Subtitle segments
        segs = []
        if metadata_json.exists():
            segs = extract_segments_for_window(
                metadata_json, ctx_start, ctx_end,
                max_segments=self.max_segments_train
            )

        # Frames
        frame_paths, frame_times = [], []
        if self.pooling == "keyframe":
            frame_paths, frame_times = collect_keyframe_by_index_with_time(keyframes_dir, idx)
            if (not frame_paths) and self.method:
                if metadata_json.exists():
                    frame_paths, frame_times = collect_method_frames_with_times(
                        movie_root, self.method, metadata_json, ts_sec, ctx_start, ctx_end,
                        max_frames=1
                    )
        elif self.pooling == "method":
            if not self.method:
                raise ValueError("pooling=method requires --method")
            frame_paths, frame_times = collect_method_frames_with_times(
                movie_root, self.method, metadata_json, ts_sec, ctx_start, ctx_end,
                max_frames=self.max_frames_train, frame_selection=self.frame_selection
            )
        else:
            raise ValueError("pooling must be keyframe or method")

        keyframe_hint_text = ""
        if self.append_keyframe:
            kfp, kft = collect_keyframe_by_index_with_time(keyframes_dir, idx)
            if kfp:
                s = set(str(p) for p in frame_paths)
                if str(kfp[0]) not in s:
                    frame_paths.append(kfp[0]); frame_times.append(kft[0] if kft else None)
            if self.keyframe_hint and kfp:
                approx_t = f"{(kft[0] if (kft and kft[0] is not None) else 0.0):.1f}s"
                keyframe_hint_text = f"Note: The user asked the question while paused at the keyframe {kfp[0].name} (t≈{approx_t})."

        if not frame_paths:
            # Build degenerate user message to keep sample valid
            user_msgs = build_two_step_messages_with_subs(
                frame_paths=[], frame_times=[],
                question=q, subtitle_segments=[],
                keyframe_hint_text=keyframe_hint_text
            )
            imgs = []
        else:
            if self.interleave:
                user_msgs = build_two_step_messages_interleaved(
                    frame_paths=frame_paths, frame_times=frame_times,
                    question=q, subtitle_segments=segs,
                    keyframe_hint_text=keyframe_hint_text,
                    segs_per_frame=self.segs_per_frame, seg_radius=self.seg_radius
                )
            else:
                user_msgs = build_two_step_messages_with_subs(
                    frame_paths=frame_paths, frame_times=frame_times,
                    question=q, subtitle_segments=segs,
                    keyframe_hint_text=keyframe_hint_text
                )
            # Load images now
        imgs = []
        for item in user_msgs[0]["content"]:
            if item.get("type") == "image":
                try:
                    imgs.append(Image.open(item["image"]).convert("RGB"))
                except Exception:
                    pass

        # Append assistant gold two-line answer
        gold_reason = gold_reason or "[Short rationale not provided]"
        two_line = f"Reasoning: {gold_reason}\nFinal Answer: {gold_ans}"
        conv = user_msgs + [{"role":"assistant","content":[{"type":"text","text": two_line}]}]
        return conv, imgs

    def __getitem__(self, i):
        movie, it = self.samples[i]
        conv, imgs = self._build_conv_and_images(movie, it)

        # Full conversation text (user + assistant)
        text_full = self.processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        # User-only with generation prompt (to get prefix length)
        text_user_prompt = self.processor.apply_chat_template(
            [conv[0]], tokenize=False, add_generation_prompt=True
        )

        # Tokenize individually to compute the label mask boundary
        tok = self.processor.tokenizer

        full_ids = tok(text_full, add_special_tokens=False).input_ids
        pref_ids = tok(text_user_prompt, add_special_tokens=False).input_ids

        # Safety: clamp if any mismatch
        user_len = min(len(full_ids)-1, len(pref_ids)) if len(full_ids) > 0 else 0
        # Build labels: -100 for user/prefix, supervise only assistant tail
        labels = [-100]*user_len + full_ids[user_len:]
        assert len(labels) == len(full_ids)

        # Now run full processor to get MM tensors (this reproduces input_ids again, but ensures pixel tensors)
        enc = self.processor(
            text=text_full, images=imgs,
            return_tensors="pt", padding=False, truncation=True, max_length=self.max_length
        )

        # Replace input_ids with the 'full_ids' we already computed (ensures same)
        enc["input_ids"] = torch.tensor([full_ids], dtype=torch.long)
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        enc["labels"] = torch.tensor([labels], dtype=torch.long)
        # Some processors add extra image meta tensors; keep as-is
        return enc

# -------------
# Collator (pad per-batch via processor)
# -------------

class DataCollatorVL:
    def __init__(self, processor: AutoProcessor, pad_to_multiple_of: int = 8):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str,Any]]) -> Dict[str, torch.Tensor]:
        # Gather lists of element-wise tensors (already 1-batch shaped)
        batch = {}
        keys = set().union(*[f.keys() for f in features])
        for k in keys:
            if k in ("input_ids","attention_mask","labels"):
                tensors = [f[k].squeeze(0) for f in features]
                batch[k] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True,
                    padding_value = (self.processor.tokenizer.pad_token_id if k != "labels" else -100)
                )
                # Optional: pad to multiple of 8 for tensor cores
                if self.pad_to_multiple_of:
                    L = batch[k].shape[1]
                    pad_len = (self.pad_to_multiple_of - (L % self.pad_to_multiple_of)) % self.pad_to_multiple_of
                    if pad_len:
                        pad_val = self.processor.tokenizer.pad_token_id if k != "labels" else -100
                        pad = torch.full((batch[k].shape[0], pad_len), pad_val, dtype=batch[k].dtype)
                        batch[k] = torch.cat([batch[k], pad], dim=1)

            else:
                # image / pixel / grid tensors (already stacked by processor)
                # Expect same shapes across items; stack along batch
                vals = [f[k] for f in features if k in f]
                if len(vals) == 0: continue
                # Each is shape (1, ...); stack on dim 0
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
    ap.add_argument("--pooling", type=str, choices=["keyframe","method"], default="method")
    ap.add_argument("--method", type=str, default="weighted_average_exponential")
    ap.add_argument("--frame_selection", type=str, choices=["near_ts","uniform","all"], default="near_ts")
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
    ap.add_argument("--lora_target_mm_projector", action="store_true",
                    help="Also adapt the vision-language projector if present.")

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

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 4-bit quant
    compute_dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
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
        # Try to catch common projector lin layers
        tmods = list(sorted(set(tmods + ["mm_projector","mm_projector.0","mm_projector.2","vision_tower.proj"])))

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

    # Dataset + Collator
    train_ds = MovieVQASFTDataset(
        root_dir=args.root_dir,
        processor=processor,
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

    # Training args
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
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,  # IMPORTANT for VL models
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()
    # Save PEFT adapter
    trainer.model.save_pretrained(args.output_dir)
    # Also save processor (tokenizer config changes)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
