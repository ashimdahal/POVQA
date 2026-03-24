#!/usr/bin/env python3

"""
TVQA evaluation with Qwen2.5-VL (vision+text, two-step) using processed frames + subtitle metadata.

Folder layout (yours):
processed_tvqa/
  └─ <vid_name>/                         # e.g., castle_s01e04_seg02_clip_01
       ├─ blend_blur_with_last_frame/    # frames as frame00001.jpg, ...
       └─ metadata_tvqa_text_centric.json

val.jsonl example line:
{"a0":"...", "a1":"...", "a2":"...", "a3":"...", "a4":"...",
 "answer_idx":4, "q":"...", "qid":137290, "show_name":"House M.D.",
 "ts":"13.37-20.8", "vid_name":"house_s03e14_seg02_clip_23"}

Key features
------------
- Interleaving option (images interleaved with the nearest subtitle snippets).
- MC prompt forces model to respond with exactly two lines:
    Reasoning: ...
    Final Answer: <A/B/C/D/E or short option text>
- Robust parser: letter (A–E) or option text (fuzzy fallback).
- Metrics: EM, F1, BLEU1, BLEU4_BP, ROUGE_L, EmbedCos (Final Answer vs gold option text),
  plus EmbedCos_Reasoning (0.0 if no gold reasoning in TVQA). Also Accuracy over choices.
- Summary JSON includes your required keys: "EmbedCos" and "EmbedCos_Reasoning".

Usage (examples)
----------------
# 1) SFT adapter (interleaved, 3 frames near mid of ts window)
python run_tvqa_eval.py \
  --tvqa_root processed_tvqa \
  --val_jsonl processed_tvqa/val.jsonl \
  --output_file runs/tvqa_qwen7b_sft_interleaved3.jsonl \
  --use_4bit \
  --peft_adapter models/sft-qwen7b-interleaved-16f/blend_blur_with_last_frame \
  --interleave --max_frames 3

# 2) SFT + DPO adapter
python run_tvqa_eval.py \
  --tvqa_root processed_tvqa \
  --val_jsonl processed_tvqa/val.jsonl \
  --output_file runs/tvqa_qwen7b_sftdpo_interleaved3.jsonl \
  --use_4bit \
  --peft_adapter models/dpo-qwen7b-interleaved-16f/blend_blur_with_last_frame \
  --interleave --max_frames 3
"""

import os, re, json, math, argparse, warnings
import time
from tqdm import tqdm
import difflib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Text + metric helpers
# -------------------------
_punc_re = re.compile(r"[^\w\s]")
_space_re = re.compile(r"\s+")

def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = _punc_re.sub("", s)
    s = _space_re.sub(" ", s)
    return s

def tokens(s: str) -> List[str]:
    return norm_text(s).split()

def em(pred: str, gold: str) -> float:
    return 1.0 if norm_text(pred) == norm_text(gold) else 0.0

def f1(pred: str, gold: str) -> float:
    p_toks, g_toks = tokens(pred), tokens(gold)
    if not p_toks and not g_toks: return 1.0
    if not p_toks or not g_toks: return 0.0
    pc, gc = Counter(p_toks), Counter(g_toks)
    overlap = sum((pc & gc).values())
    if overlap == 0: return 0.0
    prec = overlap / max(1, len(p_toks))
    reca = overlap / max(1, len(g_toks))
    return 2 * prec * reca / (prec + reca + 1e-12)

def ngrams(seq: List[str], n: int) -> Counter:
    return Counter(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))

def bleu_scores(pred: str, gold: str) -> Tuple[float, float, float]:
    p = tokens(pred); g = tokens(gold)
    if len(p) == 0 and len(g) == 0: return 1.0, 1.0, 1.0
    if len(p) == 0 or len(g) == 0: return 0.0, 0.0, 0.0
    # BLEU-1
    p1 = sum((Counter(p) & Counter(g)).values()) / max(1, len(p))
    # BLEU-4
    precisions = []
    for n in [1,2,3,4]:
        if len(p) < n or len(g) < n:
            precisions.append(0.0); continue
        pn = ngrams(p, n); gn = ngrams(g, n)
        match = sum((pn & gn).values())
        precisions.append(match / max(1, sum(pn.values())))
    bleu4 = 0.0 if min(precisions)==0.0 else math.exp(sum(math.log(x+1e-9) for x in precisions)/4)
    bp = 1.0 if len(p) > len(g) else math.exp(1 - len(g)/max(1,len(p)))
    return p1, bleu4, bp*bleu4

def rouge_l(pred: str, gold: str) -> float:
    p = tokens(pred); g = tokens(gold)
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    dp = [[0]*(len(g)+1) for _ in range(len(p)+1)]
    for i in range(1, len(p)+1):
        for j in range(1, len(g)+1):
            if p[i-1]==g[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    prec = lcs / max(1,len(p))
    reca = lcs / max(1,len(g))
    if prec==0 or reca==0: return 0.0
    return 2*prec*reca/(prec+reca)

class EmbSim:
    """Sentence-embedding cosine; safe fallback to 0 if not installed."""
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"[EmbSim] Disabled (sentence-transformers not available: {e})")
    def score(self, a: str, b: str) -> float:
        if self.model is None: return 0.0
        import torch
        ea, eb = self.model.encode([a,b], convert_to_tensor=True, normalize_embeddings=True)
        return float(torch.matmul(ea, eb.T).item())

# -------------------------
# Time + metadata helpers
# -------------------------
def parse_ts_range(s: str) -> Tuple[float, float]:
    # "start-end" in seconds (possibly floats like "13.37-20.8")
    a, b = s.split("-")
    return float(a.strip()), float(b.strip())

def resolve_tvqa_metadata(vid_root: Path) -> Path:
    p = vid_root / "metadata_tvqa_text_centric.json"
    if p.exists(): return p
    raise FileNotFoundError(f"No metadata_tvqa_text_centric.json in {vid_root}")

def collect_method_frames_with_times(
    vid_root: Path,
    method: str,
    metadata_json: Path,
    ts_center: float,
    ctx_start: float,
    ctx_end: float,
    max_frames: int,
    frame_selection: str = "near_ts",  # "near_ts" | "uniform" | "all"
) -> Tuple[List[Path], List[float]]:
    frames_dir = vid_root / method
    with open(metadata_json, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Gather all candidate (time, path) inside the context window
    candidates: List[Tuple[float, Path]] = []
    for seg in segments:
        s = seg.get("text_start_time_sec"); e = seg.get("text_end_time_sec")
        if s is None or e is None: continue
        if max(ctx_start, s) < min(ctx_end, e):
            for ch in seg.get("corresponding_chunks", []):
                fname = ch.get("saved_chunk_filename"); cst = ch.get("chunk_start_time_sec")
                if not fname or cst is None: continue
                p = frames_dir / fname
                if p.exists():
                    candidates.append((float(cst), p))

    if not candidates:
        return [], []

    # Sort by time
    candidates.sort(key=lambda x: x[0])
    times = [t for (t, _) in candidates]
    paths = [p for (_, p) in candidates]

    if frame_selection == "all":
        if max_frames and max_frames > 0:
            return paths[:max_frames], times[:max_frames]
        return paths, times

    if frame_selection == "uniform":
        if not max_frames or len(paths) <= max_frames:
            return paths, times
        step = (len(paths) - 1) / (max_frames - 1)
        idxs = [round(i * step) for i in range(max_frames)]
        uniq = []
        for k in idxs:
            if not uniq or k != uniq[-1]:
                uniq.append(k)
        return [paths[i] for i in uniq], [times[i] for i in uniq]

    # default: near_ts
    dists = [(abs(t - ts_center), i) for i, t in enumerate(times)]
    dists.sort(key=lambda x: (x[0], x[1]))
    if max_frames and max_frames > 0:
        chosen = [i for (_, i) in dists[:max_frames]]
    else:
        chosen = [i for (_, i) in dists]
    chosen.sort()
    return [paths[i] for i in chosen], [times[i] for i in chosen]

def extract_segments_for_window(
    metadata_json: Path,
    context_start: float,
    context_end: float,
    max_segments: int = 8,
) -> List[Tuple[float, float, str]]:
    with open(metadata_json, "r", encoding="utf-8") as f:
        segments = json.load(f)
    cand: List[Tuple[float, float, str]] = []
    for seg in segments:
        s = seg.get("text_start_time_sec"); e = seg.get("text_end_time_sec")
        if s is None or e is None: continue
        if max(context_start, s) < min(context_end, e):
            txt = seg.get("text", "").strip() or "[No speech]"
            cand.append((float(s), float(e), txt))
    cand.sort(key=lambda x: x[0])
    if len(cand) <= max_segments:
        return cand
    keep_head = max_segments - 1
    return cand[:keep_head] + [cand[-1]]

def _segments_near_time(segments, t, k=1, radius=2.0):
    if t is None or not segments:
        return []
    overlaps = [(abs(((s+e)/2.0) - t), s, e, txt) for (s,e,txt) in segments if (s <= t <= e)]
    if overlaps:
        overlaps.sort(key=lambda x: x[0])
        return [(s,e,txt) for (_,s,e,txt) in overlaps[:k]]
    near = [(abs(((s+e)/2.0) - t), s, e, txt) for (s,e,txt) in segments]
    near.sort(key=lambda x: x[0])
    picked = []
    for d,s,e,txt in near:
        if d <= radius and len(picked) < k:
            picked.append((s,e,txt))
    return picked

# -------------------------
# Prompt templates (MC)
# -------------------------
TEMPLATE_MC_HEADER = """\
You are given movie frames and aligned subtitle snippets for a short time window.
Answer the MULTIPLE-CHOICE question using only this context. Cite frames/segments in Reasoning.

QUESTION:
{QUESTION}

OPTIONS:
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}

OUTPUT FORMAT (exactly two lines):
Reasoning: <brief, grounded to frames/segments>
Final Answer: <A/B/C/D/E or the short option text>

STRICT RULES:
1) Ground reasoning with frame indices and/or segment time ranges.
2) Visual first for actions/objects; text first for dialogue facts.
3) No outside knowledge.
4) Be concise; no punctuation in the option letter itself.
"""

def make_interleaved_mc_messages(
    frame_paths: List[Path],
    frame_times: List[float],
    question: str,
    options: List[str],  # length 5
    subtitle_segments: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    header = TEMPLATE_MC_HEADER.format(
        QUESTION=question,
        A=options[0], B=options[1], C=options[2], D=options[3], E=options[4],
    )
    content.append({"type": "text", "text": header})
    for i, p in enumerate(frame_paths):
        t = frame_times[i] if i < len(frame_times) else None
        content.append({"type": "image", "image": str(p)})
        lines = [f"Frame #{i+1}: {p.name}" + (f" (t≈{t:.1f}s)" if t is not None else "")]
        near = _segments_near_time(subtitle_segments, t, k=1, radius=2.0)
        if near:
            for (s,e,txt) in near:
                lines.append(f"Segment ({s:.2f}s–{e:.2f}s): {txt}")
        else:
            lines.append("[No nearby subtitle segment]")
        content.append({"type": "text", "text": "\n".join(lines)})
    return [{"role": "user", "content": content}]

def make_block_mc_messages(
    frame_paths: List[Path],
    frame_times: List[float],
    question: str,
    options: List[str],
    subtitle_segments: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    def make_sub_block(segs):
        if not segs:
            return "[No subtitle segments in window]"
        return "\n".join([f"Segment ({s:.2f}s–{e:.2f}s): {t}" for (s,e,t) in segs])
    def make_frame_list(paths, times):
        rows=[]
        for i, p in enumerate(paths):
            if i < len(times) and times[i] is not None:
                rows.append(f"Frame #{i+1}: {p.name} (t≈{times[i]:.1f}s)")
            else:
                rows.append(f"Frame #{i+1}: {p.name}")
        return "\n".join(rows)

    content = [{"type":"image","image":str(p)} for p in frame_paths]
    text_block = (
        "You are given a short sequence of movie frames and aligned subtitle segments.\n"
        "Answer the MULTIPLE-CHOICE question using only this context.\n\n"
        "CONTEXT – SUBTITLES:\n" + make_sub_block(subtitle_segments) + "\n\n"
        "CONTEXT – FRAMES:\n" + make_frame_list(frame_paths, frame_times) + "\n\n"
        f"QUESTION:\n{question}\n\n"
        "OPTIONS:\n"
        f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\nE) {options[4]}\n\n"
        "OUTPUT FORMAT (exactly two lines):\n"
        "Reasoning: <brief, grounded>\n"
        "Final Answer: <A/B/C/D/E or the short option text>\n"
    )
    content.append({"type":"text","text":text_block})
    return [{"role":"user","content":content}]

# -------------------------
# Model loading + generation
# -------------------------
def load_qwen_vl(model_name_or_path: str, use_4bit=True, device_map="auto",
                 peft_adapter_path: str | None = None, peft_merge: bool = False):
    print(f"Loading {model_name_or_path} (4bit={use_4bit}) ...")
    quantization_config = None
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if use_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)

    proc_src = peft_adapter_path or model_name_or_path
    processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)

    kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=True, device_map=device_map)
    if quantization_config: kwargs["quantization_config"] = quantization_config
    else: kwargs["torch_dtype"] = torch_dtype

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)

    if peft_adapter_path:
        print(f"[PEFT] Attaching adapter: {peft_adapter_path}")
        model = PeftModel.from_pretrained(model, peft_adapter_path)
        if peft_merge:
            print("[PEFT] merge_and_unload()")
            model = model.merge_and_unload()

    model.eval()

    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
        if getattr(model, "config", None) and model.config.pad_token_id is None:
            model.config.pad_token_id = tok.eos_token_id
    return model, processor

@torch.inference_mode()
def generate_two_step(model, processor, messages, max_new_tokens=256, temperature=0.0, do_sample=False):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs = []
    for item in messages[0]["content"]:
        if item.get("type") == "image":
            try:
                imgs.append(Image.open(item["image"]).convert("RGB"))
            except Exception:
                pass
    inputs = processor(text=text, images=imgs, return_tensors="pt", padding=True).to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        if tok.pad_token_id is not None: gen_kwargs["pad_token_id"] = tok.pad_token_id
        if tok.eos_token_id is not None: gen_kwargs["eos_token_id"] = tok.eos_token_id
    out = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, input_len:]
    txt = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return txt

def parse_two_step_output(text: str) -> Tuple[str, str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    reasoning, answer = "", ""
    for line in lines:
        low = line.lower()
        if low.startswith("reasoning:") and not reasoning:
            reasoning = line.split(":", 1)[1].strip()
        if low.startswith("final answer:") and not answer:
            answer = line.split(":", 1)[1].strip()
    if not answer:
        m = re.search(r"final answer\s*:\s*(.+)", text, flags=re.I)
        if m: answer = m.group(1).strip()
    if not reasoning:
        m = re.search(r"reasoning\s*:\s*(.+?)(?:\n|$)", text, flags=re.I|re.S)
        if m: reasoning = m.group(1).strip()
    # Keep answer as short label/text; strip punctuation and cap tokens (<= 6 words)
    if answer:
        answer = " ".join(answer.split()[:6])
        answer = re.sub(r"[^\w\s]", "", answer).strip()
    return reasoning, answer

# -------------------------
# MC mapping
# -------------------------
LETTER2IDX = {"a":0,"b":1,"c":2,"d":3,"e":4}
IDX2LETTER = {v:k.upper() for k,v in LETTER2IDX.items()}

def map_answer_to_idx(pred_answer: str, options: List[str]) -> Tuple[int, str]:
    """
    Return (pred_idx, normalized_pred_text).
    Accepts 'A'..'E' or the option text. Uses fuzzy fallback if needed.
    """
    ans = pred_answer.strip()
    if not ans:
        return -1, ans
    # Letter?
    m = re.match(r"^\s*([A-Ea-e])\b", ans)
    if m:
        i = LETTER2IDX[m.group(1).lower()]
        return i, options[i]
    # Exact text?
    norm_ans = norm_text(ans)
    norms = [norm_text(x) for x in options]
    if norm_ans in norms:
        i = norms.index(norm_ans)
        return i, options[i]
    # Substring hit?
    for i, opt in enumerate(norms):
        if opt and opt in norm_ans or norm_ans in opt:
            return i, options[i]
    # Fuzzy
    sims = [(difflib.SequenceMatcher(None, norm_ans, opt).ratio(), i) for i, opt in enumerate(norms)]
    sims.sort(reverse=True)
    best_sim, best_i = sims[0]
    if best_sim >= 0.6:
        return best_i, options[best_i]
    return -1, ans

# -------------------------
# Main
# -------------------------
def main(args):
    tvqa_root = Path(args.tvqa_root)
    with open(args.val_jsonl, "r", encoding="utf-8") as f:
        val_lines = [json.loads(line) for line in f if line.strip()]

    model, processor = load_qwen_vl(
        args.model_name_or_path,
        use_4bit=args.use_4bit,
        device_map="auto",
        peft_adapter_path=args.peft_adapter,
        peft_merge=args.peft_merge,
    )
    embsim = EmbSim(args.embed_model)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w", encoding="utf-8")

    # metrics
    ems=[]; f1s=[]; b1s=[]; b4bps=[]; rls=[]; ecs=[]
    ecs_reasoning=[]; rls_reasoning=[]
    accs=[]

    processed = 0
    method = args.method

    for ex in tqdm(val_lines, desc=f"TVQA eval [{method}]", unit="qa"):
        try:
            vid = ex["vid_name"]
            question = ex["q"].strip()
            options = [ex[f"a{i}"].strip() for i in range(5)]
            gold_idx = int(ex["answer_idx"])
            gold_text = options[gold_idx]
            ts_start, ts_end = parse_ts_range(ex["ts"])
            ts_center = 0.5*(ts_start + ts_end)

            vid_root = tvqa_root / vid
            if not vid_root.exists():
                print(f"[Skip] Missing video folder: {vid}")
                continue

            md = resolve_tvqa_metadata(vid_root)

            # subtitle segments in window
            segs = extract_segments_for_window(md, ts_start, ts_end, max_segments=args.max_segments)

            # frames in window
            frame_paths, frame_times = collect_method_frames_with_times(
                vid_root, method, md, ts_center, ts_start, ts_end,
                max_frames=args.max_frames,
                frame_selection=args.frame_selection
            )

            if not frame_paths:
                pred_reason, pred_answer = "", ""
                pred_idx = -1; pred_text = ""
            else:
                if args.interleave:
                    messages = make_interleaved_mc_messages(
                        frame_paths=frame_paths, frame_times=frame_times,
                        question=question, options=options, subtitle_segments=segs
                    )
                else:
                    messages = make_block_mc_messages(
                        frame_paths=frame_paths, frame_times=frame_times,
                        question=question, options=options, subtitle_segments=segs
                    )

                raw = generate_two_step(
                    model, processor, messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample
                )
                pred_reason, pred_answer = parse_two_step_output(raw)
                pred_idx, pred_text = map_answer_to_idx(pred_answer, options)

            # metrics vs gold option text
            ems.append(em(pred_text, gold_text))
            f1s.append(f1(pred_text, gold_text))
            b1, _b4, b4bp = bleu_scores(pred_text, gold_text)
            b1s.append(b1); b4bps.append(b4bp)
            rl = rouge_l(pred_text, gold_text); rls.append(rl)
            ecs.append(embsim.score(pred_text, gold_text))

            # No gold reasoning in TVQA; keep keys present
            ecs_reasoning.append(0.0)
            rls_reasoning.append(0.0)

            accs.append(1.0 if pred_idx == gold_idx else 0.0)

            rec = {
                "qid": ex.get("qid"),
                "show_name": ex.get("show_name"),
                "vid_name": vid,
                "ts": ex.get("ts"),
                "question": question,
                "options": options,
                "gold_idx": gold_idx,
                "gold_text": gold_text,
                "pred_reasoning": pred_reason,
                "pred_answer_raw": pred_answer,
                "pred_idx": pred_idx,
                "pred_letter": IDX2LETTER.get(pred_idx, ""),
                "pred_text": pred_text,
                "correct": bool(pred_idx == gold_idx),
                "method": method,
                "frames_used": [str(p) for p in frame_paths],
                "frame_times": frame_times,
                "subtitle_segments": [{"start": s, "end": e, "text": t} for (s,e,t) in segs],
            }
            fout.write(json.dumps(rec) + "\n")
            processed += 1
            if args.limit and processed >= args.limit:
                break

        except Exception as e:
            print(f"[Error] {ex.get('vid_name')} qid={ex.get('qid')} -> {e}")

    fout.close()

    def avg(xs): return float(sum(xs)/max(1,len(xs)))
    summary = {
        "runs": processed,
        "dataset": "TVQA",
        "method": method,
        "max_frames": args.max_frames,
        "max_segments": args.max_segments,
        "metrics": {
            "Accuracy": avg(accs),
            "EM": avg(ems),
            "F1": avg(f1s),
            "BLEU1": avg(b1s),
            "BLEU4_BP": avg(b4bps),
            "ROUGE_L": avg(rls),
            "EmbedCos": avg(ecs),
            "ROUGE_L_Reasoning": avg(rls_reasoning),
            "EmbedCos_Reasoning": avg(ecs_reasoning)
        },
        "output_file": str(out_path)
    }
    summ = out_path.with_suffix(".summary.json")
    with open(summ, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser("TVQA eval (two-step LVLM, frames+subs, multiple-choice)")
    ap.add_argument("--tvqa_root", type=str, required=True, help="processed_tvqa/ folder root")
    ap.add_argument("--val_jsonl", type=str, required=True, help="Path to TVQA val.jsonl")
    ap.add_argument("--output_file", type=str, required=True)

    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--use_4bit", action="store_true")

    ap.add_argument("--method", type=str, default="blend_blur_with_last_frame",
                    help="Frame folder inside each vid dir (default: blend_blur_with_last_frame)")
    ap.add_argument("--max_frames", type=int, default=3)
    ap.add_argument("--max_segments", type=int, default=8)
    ap.add_argument("--frame_selection", type=str, choices=["near_ts","uniform","all"], default="near_ts")

    ap.add_argument("--interleave", action="store_true",
                    help="Interleave frames with nearest subtitle snippet cards")
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--do_sample", action="store_true")

    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--peft_adapter", type=str, default=None,
                    help="Path to LoRA adapter (SFT or SFT+DPO)")
    ap.add_argument("--peft_merge", action="store_true")

    args = ap.parse_args()
    main(args)
