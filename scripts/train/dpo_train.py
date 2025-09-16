#!/usr/bin/env python
"""
preference_optimize_dpo_qwen2_5_vl.py
Direct Preference Optimization (DPO) for Qwen/Qwen2.5-VL-7B with QLoRA.

Key behavior in this version:
- Reference model = BASE + SFT LoRA (frozen), loaded via load_qwen_vl (processor comes from adapter).
- Policy model = BASE + SFT LoRA (trainable).
- DPO objective computed ONLY over Final Answer tokens when --focus_final_only (default: ON).
- Negatives default to correctness-only (wrong answer but same format).

Usage (per method):
python -m scripts.train.dpo_train \
  --root_dir . \
  --output_dir models/dpo-qwen7b-interleaved-16f/weighted_average \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --ref_model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --sft_adapter models/sft-qwen7b-interleaved-16f/weighted_average \
  --ref_peft_adapter models/sft-qwen7b-interleaved-16f/weighted_average \
  --pooling method --method weighted_average \
  --frame_selection uniform --max_frames_train 16 --max_segments_train 2048 \
  --interleave --append_keyframe --keyframe_hint \
  --num_train_epochs 2 --beta 0.2 --length_normalize \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --bf16 --gradient_checkpointing --correctness_only

"""

import os, json, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Reuse helpers (frames/subtitles/prompt builders + model loader)
from scripts.chain_of_thoughts.generate_synthetic_movies import (
    hms_to_sec, parse_context_ts,
    resolve_metadata_file, extract_segments_for_window,
    collect_method_frames_with_times, collect_keyframe_by_index_with_time,
    build_two_step_messages_with_subs, build_two_step_messages_interleaved,
    load_qwen_vl,  # <--- use this to attach PEFT to REF & source processor from adapter
)

# -----------------
# Utilities
# -----------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def suggested_target_modules(model: nn.Module) -> List[str]:
    candidates = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","qkv_proj"}
    present = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            for c in candidates:
                if n.endswith(c):
                    present.add(c)
    return sorted(present) if present else [
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ]

def two_line_text(reason: str, answer: str) -> str:
    reason = (reason or "[Short rationale not provided]").strip()
    answer = answer.strip()
    return f"Reasoning: {reason}\nFinal Answer: {answer}"

# --------- negatives ---------

def make_rejected_variants_correctness_only(
    question: str,
    gold_reason: str,
    gold_answer: str,
    distractor_answer: str,
) -> List[str]:
    # Wrong answer variants that keep the two-line shape.
    bad1 = two_line_text(gold_reason, distractor_answer)
    bad2_reason = (
        (gold_reason or "Visual cues appear ambiguous in the provided frames.")
        + " Given these cues, an alternative conclusion seems more likely."
    )
    bad2 = two_line_text(bad2_reason, distractor_answer)
    bad3 = two_line_text(
        "Reasoning references details not visible in the selected frames or subtitles.",
        distractor_answer,
    )
    return [bad1, bad2, bad3]

def make_rejected_variants(gold_reason: str, gold_answer: str, distractor_answer: str) -> List[str]:
    # Wrong answer + format violations (legacy)
    bad1 = two_line_text(gold_reason, distractor_answer)
    long_ans = gold_answer + " therefore this is definitely correct."
    bad2 = two_line_text(gold_reason + " The evidence is absolutely conclusive.", long_ans)
    bad3 = f"Final Answer: {gold_answer}\nReasoning: {gold_reason}"
    return [bad1, bad2, bad3]

# -----------------
# Dataset (pairs)
# -----------------

class DpoMovieVQAPairs(Dataset):
    """
    Yields dict with:
      - user_msgs: chat messages (images + text)
      - imgs: list of PIL.Image
      - chosen: gold two-line string
      - rejected: negative two-line string
    """

    def __init__(
        self,
        root_dir: str,
        processor,  # AutoProcessor (from adapter when available)
        split: str = "train",
        split_ratio: float = 0.9,
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
        seed: int = 42,
        correctness_only: bool = True,
    ):
        assert split in {"train", "eval"}
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
        self.correctness_only = correctness_only

        self.rnd = random.Random(seed)

        # reproducible movie split
        all_ann_files = sorted(self.ann_dir.glob("*.json"))
        self.rnd.shuffle(all_ann_files)
        split_idx = int(len(all_ann_files) * split_ratio)
        split_files = all_ann_files[:split_idx] if split == "train" else all_ann_files[split_idx:]
        print(f"[INFO] {split.upper()} split with {len(split_files)} movies.")

        # distractor answer pool
        all_answers: List[str] = []
        for af in all_ann_files:
            try:
                with open(af, "r", encoding="utf-8") as f:
                    for it in json.load(f):
                        if it.get("answer"):
                            all_answers.append(it["answer"].strip())
            except Exception:
                pass
        if not all_answers:
            all_answers = ["unknown", "not sure", "cannot determine"]
        self.all_answers = all_answers

        # samples
        self.samples: List[Tuple[str, Dict[str, Any]]] = []
        for ann_file in split_files:
            try:
                items = json.load(open(ann_file, "r", encoding="utf-8"))
            except Exception:
                continue
            movie = ann_file.stem
            movie_root = self.out_root / movie
            if not movie_root.exists():
                continue
            for it in items:
                if split == "eval" and not (it.get("reasoning") and it.get("answer")):
                    continue
                self.samples.append((movie, it))
                if limit and len(self.samples) >= limit:
                    break
            if limit and len(self.samples) >= limit:
                break

        print(f"[INFO] Loaded {len(self.samples)} {split} samples.")

    def __len__(self):
        return len(self.samples)

    def _build_user_and_images(self, movie: str, it: Dict[str, Any]):
        movie_root = self.out_root / movie
        keyframes_dir = movie_root / "KEY_FRAMES"

        # metadata
        try:
            metadata_json = resolve_metadata_file(
                movie_root,
                self.method if self.pooling == "method" else None,
            )
        except Exception:
            metadata_json = movie_root / "metadata_text_centric.json"

        q = it["question"].strip()
        ts_sec = hms_to_sec(it["timestamp"].strip())
        ctx_start, ctx_end = parse_context_ts(it["contextTimestamp"])
        idx = int(it["index"])

        # subtitles
        segs = []
        if metadata_json.exists():
            segs = extract_segments_for_window(
                metadata_json, ctx_start, ctx_end, max_segments=self.max_segments_train
            )

        # frames
        frame_paths: List[Path] = []
        frame_times: List[float] = []
        if self.pooling == "keyframe":
            frame_paths, frame_times = collect_keyframe_by_index_with_time(keyframes_dir, idx)
            if (not frame_paths) and self.method and metadata_json.exists():
                frame_paths, frame_times = collect_method_frames_with_times(
                    movie_root, self.method, metadata_json, ts_sec, ctx_start, ctx_end,
                    max_frames=1,
                )
        elif self.pooling == "method":
            frame_paths, frame_times = collect_method_frames_with_times(
                movie_root, self.method, metadata_json, ts_sec, ctx_start, ctx_end,
                max_frames=self.max_frames_train, frame_selection=self.frame_selection,
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
                approx_t = f"{(kft[0] if (kft and kft[0] is not None) else 0.0):.1f}s"
                keyframe_hint_text = (
                    f"Note: The user asked the question while paused at the keyframe {kfp[0].name} (t≈{approx_t})."
                )

        # user message + images
        if not frame_paths:
            user_msgs = build_two_step_messages_with_subs(
                frame_paths=[], frame_times=[], question=q,
                subtitle_segments=[], keyframe_hint_text=keyframe_hint_text,
            )
            imgs: List[Image.Image] = []
        else:
            if self.interleave:
                user_msgs = build_two_step_messages_interleaved(
                    frame_paths=frame_paths, frame_times=frame_times,
                    question=q, subtitle_segments=segs,
                    keyframe_hint_text=keyframe_hint_text,
                    segs_per_frame=self.segs_per_frame, seg_radius=self.seg_radius,
                )
            else:
                user_msgs = build_two_step_messages_with_subs(
                    frame_paths=frame_paths, frame_times=frame_times,
                    question=q, subtitle_segments=segs,
                    keyframe_hint_text=keyframe_hint_text,
                )
            imgs = []
            for item in user_msgs[0]["content"]:
                if item.get("type") == "image":
                    try:
                        imgs.append(Image.open(item["image"]).convert("RGB"))
                    except Exception:
                        pass

        return user_msgs, imgs

    def __getitem__(self, i):
        movie, it = self.samples[i]
        user_msgs, imgs = self._build_user_and_images(movie, it)

        gold_reason = (it.get("reasoning") or "").strip()
        gold_answer = it["answer"].strip()
        chosen = two_line_text(gold_reason, gold_answer)

        # --- token-level guard to ensure distractor != gold ---
        tok = self.processor.tokenizer
        def tok_norm(s: str) -> list[int]:
            return tok(s.strip().lower(), add_special_tokens=False)["input_ids"]

        gold_ids = tok_norm(gold_answer)

        distractor = gold_answer
        tries = 0
        # try harder to find a genuinely different answer by token IDs
        while tries < 20:
            cand = self.rnd.choice(self.all_answers).strip()
            if cand and tok_norm(cand) != gold_ids:
                distractor = cand
                break
            tries += 1
        # hard fallback if we somehow never found a different one
        if tok_norm(distractor) == gold_ids:
            distractor = "unknown"

        q = it.get("question", "").strip()

        if self.correctness_only:
            rejected_candidates = make_rejected_variants_correctness_only(
                q, gold_reason, gold_answer, distractor
            )
        else:
            rejected_candidates = make_rejected_variants(
                gold_reason, gold_answer, distractor
            )

        rejected = self.rnd.choice(rejected_candidates)
        return {"user_msgs": user_msgs, "imgs": imgs, "chosen": chosen, "rejected": rejected}

# -----------------
# Collator
# -----------------

class DpoCollatorVL:
    def __init__(self, processor, max_length: int = 2048, pad_to_multiple_of: int = 8,
                 focus_final_only: bool = True):
        self.processor = processor
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.focus_final_only = focus_final_only

    def _find_subseq(self, haystack: list[int], needle: list[int]) -> int:
        L = len(needle)
        if L == 0: return -1
        for i in range(0, len(haystack) - L + 1):
            if haystack[i:i+L] == needle:
                return i
        return -1

    def _encode_pair(self, user_msgs, imgs, reply_text: str):
        # full conversation (user + assistant reply)
        conv = user_msgs + [{"role": "assistant", "content": [{"type": "text", "text": reply_text}]}]
        text_full = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        # user prefix only
        text_prefix = self.processor.apply_chat_template([user_msgs[0]], tokenize=False, add_generation_prompt=True)

        enc_full = self.processor(
            text=text_full, images=imgs, return_tensors="pt",
            padding=False, truncation=True, max_length=self.max_length
        )
        enc_pref = self.processor(
            text=text_prefix, images=imgs, return_tensors="pt",
            padding=False, truncation=True, max_length=self.max_length
        )

        input_ids = enc_full["input_ids"][0]
        prefix_len = enc_pref["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prefix_len] = -100  # mask the prompt (all user content)

        if self.focus_final_only:
            # Tokenize the assistant reply alone (no special tokens)
            reply_ids = self.processor.tokenizer(
                reply_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].tolist()

            # Robustly find the "Final Answer:" marker (tokenized variants)
            def ids(s: str) -> list[int]:
                return self.processor.tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()

            candidates = [
                "Final Answer:",
                "Final Answer :",
                "final answer:",
                "final answer :",
                "Final answer:",
                "Final answer :",
            ]
            pos = -1
            marker_len = 0
            for m in candidates:
                mid = ids(m)
                j = self._find_subseq(reply_ids, mid)
                if j != -1:
                    pos = j
                    marker_len = len(mid)
                    break

            if pos != -1:
                ans_start_in_reply = pos + marker_len
                # Mask the assistant reasoning portion (everything up to the answer start)
                labels[prefix_len : prefix_len + ans_start_in_reply] = -100
                # Keep supervision on the actual answer tokens (and anything after)
            # else: fallback to supervising the full assistant reply (already covered)

        out = {k: v for k, v in enc_full.items()}
        out["labels"] = labels.unsqueeze(0)
        out["prefix_len"] = torch.tensor([prefix_len], dtype=torch.long)
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_list, rejected_list = [], []
        for f in features:
            enc_c = self._encode_pair(f["user_msgs"], f["imgs"], f["chosen"])
            enc_r = self._encode_pair(f["user_msgs"], f["imgs"], f["rejected"])
            chosen_list.append(enc_c)
            rejected_list.append(enc_r)

        def pad_stack(batch_list: List[Dict[str, torch.Tensor]], prefix: str) -> Dict[str, torch.Tensor]:
            keys = set().union(*[b.keys() for b in batch_list])
            out: Dict[str, torch.Tensor] = {}
            for k in keys:
                if k == "prefix_len":
                    out[f"{prefix}_prefix_len"] = torch.cat([b[k] for b in batch_list], dim=0)
                    continue
                if k in ("input_ids", "attention_mask", "labels"):
                    pad_val = (
                        self.processor.tokenizer.pad_token_id if k != "labels" else -100
                    )
                    tensors = [b[k].squeeze(0) for b in batch_list]
                    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_val)
                    if self.pad_to_multiple_of:
                        L = padded.shape[1]
                        pad_len = (self.pad_to_multiple_of - (L % self.pad_to_multiple_of)) % self.pad_to_multiple_of
                        if pad_len:
                            pad = torch.full((padded.shape[0], pad_len), pad_val, dtype=padded.dtype)
                            padded = torch.cat([padded, pad], dim=1)
                    out[f"{prefix}_{k}"] = padded
                else:
                    vals = [b[k] for b in batch_list if k in b]
                    if not vals:
                        continue
                    out[f"{prefix}_{k}"] = torch.cat(vals, dim=0)
            return out

        out_c = pad_stack(chosen_list, "chosen")
        out_r = pad_stack(rejected_list, "rejected")
        out_c.update(out_r)
        return out_c

# -----------------
# DPO Trainer
# -----------------

class DPOTrainer(Trainer):
    def __init__(self, ref_model: nn.Module, beta: float = 0.2, length_normalize: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.length_normalize = length_normalize
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    @staticmethod
    def _sequence_logprob(logits: torch.Tensor, labels: torch.Tensor, length_normalize: bool) -> torch.Tensor:
        # logits: [B, L, V], labels: [B, L] with -100 masked
        log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
        label_mask = (labels != -100)              # [B, L]
        gather = log_probs.gather(dim=-1, index=labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # [B, L]
        gather = gather * label_mask
        token_sums = gather.sum(dim=1)                                # [B]
        lengths = label_mask.sum(dim=1).clamp(min=1)                  # [B]
        return token_sums / lengths if length_normalize else token_sums

    def _fw(self, model, batch: Dict[str, torch.Tensor], prefix: str):
        keys = [k for k in batch.keys() if k.startswith(prefix + "_")]
        kwargs = {}
        for k in keys:
            base = k[len(prefix) + 1:]
            if base in ("prefix_len", "labels"):
                continue
            kwargs[base] = batch[k].to(self.args.device)
        outputs = model(**kwargs)
        logits = outputs.logits
        labels = batch[f"{prefix}_labels"].to(self.args.device)
        return self._sequence_logprob(logits, labels, self.length_normalize)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # DPO: -log σ(β * [(logπ_c - logπ_r) - (logπ_ref_c - logπ_ref_r)])
        logp_c = self._fw(model, inputs, "chosen")
        logp_r = self._fw(model, inputs, "rejected")
        with torch.no_grad():
            logp_c_ref = self._fw(self.ref_model, inputs, "chosen")
            logp_r_ref = self._fw(self.ref_model, inputs, "rejected")
        pref_logits = (logp_c - logp_r) - (logp_c_ref - logp_r_ref)
        loss = -F.logsigmoid(self.beta * pref_logits).mean()

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({
                "loss_dpo": float(loss.detach().cpu().item()),
                "policy_advantage_mean": float((logp_c - logp_r).mean().detach().cpu().item()),
                "ref_advantage_mean": float((logp_c_ref - logp_r_ref).mean().detach().cpu().item()),
            })

        if return_outputs:
            return loss, {
                "eval_loss": float(loss.detach().cpu().item()),
                "eval_logp_chosen": float(logp_c.mean().detach().cpu().item()),
                "eval_logp_rejected": float(logp_r.mean().detach().cpu().item()),
            }
        return loss

# -----------------
# Main
# -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--ref_model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="Frozen reference backbone (base).")

    # Adapter wiring
    ap.add_argument("--sft_adapter", type=str, default=None,
                    help="Trainable LoRA to continue from (policy). Also used as default ref adapter.")
    ap.add_argument("--ref_peft_adapter", type=str, default=None,
                    help="Attach this LoRA to the FROZEN reference model (if unset, falls back to --sft_adapter).")

    # Shaping (same semantics as SFT)
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

    # Token/sequence
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split_ratio", type=float, default=0.9)

    # LoRA config (only used if sft_adapter is NOT provided)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_mm_projector", action="store_true")

    # DPO knobs
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--length_normalize", action="store_true")

    # Train args
    ap.add_argument("--learning_rate", type=float, default=2e-5)  # stabler than 5e-5 for DPO QLoRA
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Negatives + focus
    ap.add_argument("--correctness_only", action="store_true",
                    help="Use wrong-answer-only negatives (no format penalties).")
    ap.add_argument("--focus_final_only", action="store_true",
                    help="Compute DPO loss only on Final Answer tokens.")
    # defaults ON for your use case
    ap.set_defaults(correctness_only=True, focus_final_only=True)

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 4-bit quant (policy); ref will be loaded via helper (also 4-bit)
    compute_dtype = (
        torch.bfloat16 if (args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # -------- Reference (FROZEN): base + SFT adapter; processor from adapter --------
    ref_adapter = args.ref_peft_adapter or args.sft_adapter
    ref_model, processor = load_qwen_vl(
        model_name_or_path=args.ref_model_name_or_path,
        use_4bit=True,
        device_map="auto",
        peft_adapter_path=ref_adapter,
        peft_merge=False,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    if getattr(ref_model, "config", None) is not None:
        ref_model.config.use_cache = False
    if args.gradient_checkpointing:
        ref_model.gradient_checkpointing_enable()

    tok = getattr(processor, "tokenizer", None)

    # -------- Policy (TRAINABLE): base (+ SFT adapter) --------
    policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_cfg, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True,
    )
    if tok is not None and tok.pad_token_id is None and tok.eos_token_id is not None:
        if getattr(policy, "config", None) and policy.config.pad_token_id is None:
            policy.config.pad_token_id = tok.eos_token_id
    policy.config.use_cache = False
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()

    # Freeze vision towers by default
    for n, p in policy.named_parameters():
        if any(k in n for k in ["vision_tower","visual","mm_vision","image","multi_modal"]):
            p.requires_grad = False

    # Prepare for k-bit training
    policy = prepare_model_for_kbit_training(policy)

    # Load/Init LoRA
    if args.sft_adapter:
        policy = PeftModel.from_pretrained(policy, args.sft_adapter, is_trainable=True)
        print(f"[INFO] POLICY: continued from SFT LoRA @ {args.sft_adapter}")
    else:
        tmods = suggested_target_modules(policy)
        if args.lora_target_mm_projector:
            tmods = list(sorted(set(tmods + ["mm_projector","mm_projector.0","mm_projector.2","vision_tower.proj"])))
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=tmods, bias="none", task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora_cfg)
        print(f"[INFO] POLICY: initialized fresh LoRA with targets: {tmods}")
    try:
        policy.print_trainable_parameters()
    except Exception:
        pass

    # Datasets
    train_ds = DpoMovieVQAPairs(
        root_dir=args.root_dir, processor=processor, split="train", split_ratio=args.split_ratio,
        pooling=args.pooling, method=args.method, frame_selection=args.frame_selection,
        max_frames_train=args.max_frames_train, max_segments_train=args.max_segments_train,
        append_keyframe=args.append_keyframe, keyframe_hint=args.keyframe_hint,
        interleave=args.interleave, segs_per_frame=args.segs_per_frame, seg_radius=args.seg_radius,
        max_length=args.max_length, limit=args.limit, seed=args.seed,
        correctness_only=args.correctness_only,
    )
    eval_ds = DpoMovieVQAPairs(
        root_dir=args.root_dir, processor=processor, split="eval", split_ratio=args.split_ratio,
        pooling=args.pooling, method=args.method, frame_selection=args.frame_selection,
        max_frames_train=args.max_frames_train, max_segments_train=args.max_segments_train,
        append_keyframe=args.append_keyframe, keyframe_hint=args.keyframe_hint,
        interleave=args.interleave, segs_per_frame=args.segs_per_frame, seg_radius=args.seg_radius,
        max_length=args.max_length, limit=args.limit, seed=args.seed,
        correctness_only=args.correctness_only,
    )
    collator = DpoCollatorVL(processor, max_length=args.max_length, focus_final_only=args.focus_final_only)

    # Train args
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # IMPORTANT for VL models
        per_device_eval_batch_size=1,
        report_to="none",
        save_strategy="no",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        overwrite_output_dir=True,
        max_grad_norm=args.max_grad_norm,
    )

    trainer = DPOTrainer(
        ref_model=ref_model,
        beta=args.beta,
        length_normalize=args.length_normalize,
        model=policy,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # Debug prints
    dl = trainer.get_train_dataloader()
    print("Total train pairs:", len(trainer.train_dataset))
    print("Total eval pairs:", len(eval_ds))
    print("World size:", trainer.args.world_size)
    print("Per-device batch size:", trainer.args.per_device_train_batch_size)
    print("Batches/epoch (this process):", len(dl))
    print("Optimizer updates/epoch (with grad acc):",
          (len(dl) + trainer.args.gradient_accumulation_steps - 1) // trainer.args.gradient_accumulation_steps)

    trainer.train()

    # Save LoRA adapter + processor
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("[DONE] Saved DPO-updated LoRA to:", args.output_dir)


if __name__ == "__main__":
    main()
