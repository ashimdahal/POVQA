"""
generate_synthetic_movies.py

Vision+Text (subtitles) two-step LVLM baseline for your movie VQA dataset.
It constructs prompts from **frames + aligned subtitle segments** and enforces a
strict two-line output:
  • "Reasoning:"  — short, grounded explanation that cites frames/segments
  • "Final Answer:" — ≤ 6 words, no punctuation

What this script does
---------------------
1) **Reads annotations** from `annotations/<movie>.json` with fields:
   `timestamp`, `contextTimestamp`, `question`, `answer`, `reasoning`, `index`.
2) **Selects frames** for each item:
   - **KEY_FRAMES/** by `index`, or
   - **method folder** (e.g., `blend_blur_with_last_frame`, `weighted_*`) using
     `metadata_text_centric*.json` to pick frames inside the `contextTimestamp` window.  
   *(If your build includes frame-selection flags, you can pick the N closest to
   the question time, sample uniformly, or include all frames—see examples.)*
3) **Extracts subtitle segments** overlapping the `contextTimestamp` from the
   matching `metadata_text_centric*.json`, compacts to a readable block.
4) **Builds a chat message**: images first, then a single text prompt that embeds
   the subtitle block, a frame legend (indices & ~times), and the question.
   *(Optionally, your build can append the keyframe image and include a one-line
   hint that the user was paused at that keyframe.)*
5) **Generates** two-step output with an LVLM (default: `Qwen/Qwen2.5-VL-7B-Instruct`,
   4-bit optional), then **parses** `Reasoning:` and `Final Answer:`.
6) **Writes** `predictions.jsonl` (one line per QA item) and a `*.summary.json`
   with aggregate metrics.

Assumed project layout
----------------------
<root_dir>/
  ├─ annotations/<movie>.json                       # QA items
  └─ out_preprocessed/<movie>/
      ├─ KEY_FRAMES/                                # keyframes by index
      ├─ blend_blur_with_last_frame/  (or weighted_*)
      ├─ metadata_text_centric.json
      ├─ metadata_text_centric_<method>.json        # per-method variant (optional)
      └─ run_summary.json

Metrics
-------
- **EM** (exact match, normalized)
- **Token F1**
- **BLEU-1**, **BLEU-4** (brevity-penalized)
- **ROUGE-L**
- **Embedding cosine** (via `sentence-transformers`, optional)

Usage examples
--------------
# 1) KEY_FRAMES only (+ subtitles)
python run_movie_vqa_two_step_with_subs.py \
  --root_dir /path/to/project \
  --pooling keyframe \
  --use_4bit \
  --output_file runs/qwen7b_keyframe_subs.jsonl

# 2) Motion-blur method (1 frame) + subtitles
python run_movie_vqa_two_step_with_subs.py \
  --root_dir /path/to/project \
  --pooling method \
  --method blend_blur_with_last_frame \
  --max_frames 1 \
  --use_4bit \
  --output_file runs/qwen7b_blur1_subs.jsonl

# 3) Short temporal window (3 frames) from weighted_average_exponential + subtitles
python run_movie_vqa_two_step_with_subs.py \
  --root_dir /path/to/project \
  --pooling method \
  --method weighted_average_exponential \
  --max_frames 3 \
  --use_4bit \
  --output_file runs/qwen7b_wexp3_subs.jsonl

# (Optional, if enabled in your build)
#   • Select frames uniformly across the window or include all frames
#   • Append the keyframe at the end and add a one-line hint
# Examples:
#   --frame_selection uniform --max_frames 5
#   --frame_selection all --max_frames 12
#   --append_keyframe --keyframe_hint

Key arguments
-------------
--root_dir              Path to project root containing `annotations/` and `out_preprocessed/`.
--output_file           Where to write predictions JSONL; a `*.summary.json` is written alongside.
--model_name_or_path    HF model id or local path (default: Qwen/Qwen2.5-VL-7B-Instruct).
--use_4bit              Load with bitsandbytes 4-bit quantization (if available).
--embed_model           Sentence-transformers model for embedding cosine (optional).
--limit                 Process only the first N items (debugging).
# If present in your build:
--pooling {keyframe,method}
--method  {blend_blur_with_last_frame, weighted_*, ...}
--max_frames INT
--max_segments INT
--frame_selection {near_ts,uniform,all}
--append_keyframe, --keyframe_hint

"""

import os, re, json, math, argparse, warnings
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel 

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =======================
# Time helpers
# =======================

def hms_to_sec(hms: str) -> float:
    parts = [p.strip() for p in hms.strip().split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return int(h)*3600 + int(m)*60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m)*60 + float(s)
    raise ValueError(f"Bad timestamp '{hms}'")

def parse_context_ts(s: str) -> Tuple[float, float]:
    # "HH:MM:SS - HH:MM:SS"
    a, b = s.split("-")
    return hms_to_sec(a.strip()), hms_to_sec(b.strip())

def parse_hms_from_keyframe_name(name: str) -> float:
    # e.g., KEY_00-03-54_0000.jpg -> 00-03-54
    m = re.search(r"KEY_(\d{2})-(\d{2})-(\d{2})_", name)
    if not m:
        return None
    H, M, S = map(int, m.groups())
    return H*3600 + M*60 + S

# =======================
# Metrics
# =======================

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
    return 2*prec*reca/(prec+reca+1e-12)

def ngrams(seq: List[str], n: int) -> Counter:
    return Counter(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))

def bleu_scores(pred: str, gold: str) -> Tuple[float, float, float]:
    # returns BLEU-1, BLEU-4 (no BP), BLEU-4 with brevity penalty
    p = tokens(pred); g = tokens(gold)
    if len(p) == 0 and len(g) == 0: return 1.0, 1.0, 1.0
    if len(p) == 0 or len(g) == 0: return 0.0, 0.0, 0.0
    # BLEU-1
    p1 = sum((Counter(p) & Counter(g)).values()) / max(1, len(p))
    # BLEU-4
    precisions = []
    for n in [1,2,3,4]:
        if len(p) < n or len(g) < n:
            precisions.append(0.0)
            continue
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

# =======================
# Model loader (Qwen-VL)
# =======================

def load_qwen_vl(model_name_or_path: str, use_4bit=True, device_map="auto",
                 peft_adapter_path: str | None = None, peft_merge: bool = False):
    print(f"Loading {model_name_or_path} (4bit={use_4bit}) ...")
    quantization_config = None
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if use_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)

    # Prefer adapter’s processor (tokenizer/chat template) if provided
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

# =======================
# Metadata + frames
# =======================

def resolve_metadata_file(movie_root: Path, method: str) -> Path:
    """
    Returns the metadata json path that corresponds to the selected method.
    Priority:
      1) metadata_text_centric_<method>.json
      2) metadata_text_centric.json
    """
    if method:
        cand = movie_root / f"metadata_text_centric_{method}.json"
        if cand.exists(): return cand
    default = movie_root / "metadata_text_centric.json"
    if default.exists(): return default
    raise FileNotFoundError(f"No metadata_text_centric json for {movie_root.name} (method={method})")

def collect_method_frames_with_times(
    movie_root: Path,
    method: str,
    metadata_json: Path,
    ts_sec: float,
    ctx_start: float,
    ctx_end: float,
    max_frames: int,
    frame_selection: str = "near_ts",  # "near_ts" | "uniform" | "all"
) -> Tuple[List[Path], List[float]]:
    frames_dir = movie_root / method
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

    # sort by time
    candidates.sort(key=lambda x: x[0])
    times = [t for (t, _) in candidates]
    paths = [p for (_, p) in candidates]

    if frame_selection == "all":
        # (Optionally cap to avoid exceeding LVLM image limits)
        if max_frames is not None and max_frames > 0:
            times, paths = times[:max_frames], paths[:max_frames]
        return paths, times

    if frame_selection == "uniform":
        if max_frames <= 0 or len(paths) <= max_frames:
            return paths, times
        step = (len(paths) - 1) / (max_frames - 1)
        idxs = [round(i * step) for i in range(max_frames)]
        uniq = []
        for k in idxs:
            if not uniq or k != uniq[-1]:
                uniq.append(k)
        times_sel = [times[i] for i in uniq]
        paths_sel = [paths[i] for i in uniq]
        return paths_sel, times_sel

    # default: "near_ts" — take closest max_frames to ts_sec
    dists = [(abs(t - ts_sec), i) for i, t in enumerate(times)]
    dists.sort(key=lambda x: (x[0], x[1]))
    chosen = [i for (_, i) in dists[:max_frames]] if max_frames > 0 else [i for (_, i) in dists]
    chosen.sort()
    times_sel = [times[i] for i in chosen]
    paths_sel = [paths[i] for i in chosen]
    return paths_sel, times_sel

def collect_keyframe_by_index_with_time(keyframes_dir: Path, idx: int) -> Tuple[List[Path], List[float]]:
    # KEY_*_{idx:04d}.jpg ; try to parse HH-MM-SS from name for an approximate time
    hits = sorted(keyframes_dir.glob(f"*_{idx:04d}.jpg"))
    if not hits:
        return [], []
    t = parse_hms_from_keyframe_name(hits[0].name)
    return [hits[0]], [t] if t is not None else [None]

# =======================
# Subtitle segments
# =======================

def extract_segments_for_window(
    metadata_json: Path,
    context_start: float,
    context_end: float,
    max_segments: int = 8,
) -> List[Tuple[float, float, str]]:
    """
    Read metadata_text_centric*.json and return up to `max_segments` that
    overlap [context_start, context_end]. Each item is (seg_start, seg_end, text_str).
    Empty text becomes "[No speech]". Keeps time order; compacts if too many.
    """
    with open(metadata_json, "r", encoding="utf-8") as f:
        segments = json.load(f)
    cand: List[Tuple[float, float, str]] = []
    for seg in segments:
        s = seg.get("text_start_time_sec")
        e = seg.get("text_end_time_sec")
        if s is None or e is None:
            continue
        if max(context_start, s) < min(context_end, e):
            txt = seg.get("text", "").strip() or "[No speech]"
            cand.append((float(s), float(e), txt))
    cand.sort(key=lambda x: x[0])
    if len(cand) <= max_segments:
        return cand
    keep_head = max_segments - 1
    return cand[:keep_head] + [cand[-1]]

def make_subtitle_block(segments: List[Tuple[float, float, str]]) -> str:
    if not segments:
        return "[No subtitle segments available in the interval]"
    lines = []
    for (s, e, t) in segments:
        lines.append(f"Segment ({s:.2f}s–{e:.2f}s): {t}")
    return "\n".join(lines)

def make_frame_list(frame_paths: List[Path], frame_times: List[float]) -> str:
    items = []
    for i, p in enumerate(frame_paths):
        base = p.name
        if i < len(frame_times) and frame_times[i] is not None:
            items.append(f"Frame #{i+1}: {base} (t≈{frame_times[i]:.1f}s)")
        else:
            items.append(f"Frame #{i+1}: {base}")
    return "\n".join(items)

# =======================
# Prompting & generation
# =======================

# ---------- Interleaving helpers & alt header ----------

def _segments_near_time(segments, t, k=1, radius=2.0):
    """Return up to k segments that overlap t or are nearest within radius seconds.
    segments: list of (start, end, text). t: float seconds.
    """
    if t is None or not segments:
        return []
    overlaps = [(abs(((s+e)/2.0) - t), s, e, txt) for (s,e,txt) in segments if (s <= t <= e)]
    if overlaps:
        overlaps.sort(key=lambda x: x[0])
        return [(s,e,txt) for (_,s,e,txt) in overlaps[:k]]
    # nearest by center within radius
    near = [(abs(((s+e)/2.0) - t), s, e, txt) for (s,e,txt) in segments]
    near.sort(key=lambda x: x[0])
    picked = []
    for d,s,e,txt in near:
        if d <= radius and len(picked) < k:
            picked.append((s,e,txt))
    return picked

TEMPLATE_INTERLEAVED_HEADER = """\
You are given an interleaved sequence of movie frames with their closest subtitle segments.
Use only this context to answer. Cite frames/segments in Reasoning.

{KEYFRAME_HINT}QUESTION:
{QUESTION}

OUTPUT FORMAT (exactly two lines):
Reasoning: <Give your brief reasoning for choosing the answer, build logic from the text and images>
Final Answer: <Give the short succint answer directly addressing the question>

STRICT RULES:
1) Grounding: reference specific frame indices and/or segment time ranges.
2) Visual first for actions/objects; text first for dialogue facts.
3) No outside knowledge.
4) Be concise; no punctuation in Final Answer.
5) If insufficient, say so briefly in Reasoning and answer succinctly.
"""

TEMPLATE_VISION_TEXT_TWO_STEP = (
    "You are given a short sequence of movie frames and the aligned subtitle segments.\n"
    "Your job is to answer the question using only this provided context.\n\n"
    "{KEYFRAME_HINT}"  
    "CONTEXT – SUBTITLE SEGMENTS (ordered, time-stamped):\n{SUBTITLE_BLOCK}\n\n"
    "CONTEXT – VISUAL FRAMES (ordered):\n{FRAME_LIST}\n\n"
    "QUESTION:\n{QUESTION}\n\n"
    "OUTPUT FORMAT (exactly two lines):\n"
    "Reasoning: <one or two short sentences grounded to frames/segments>\n"
    "Final Answer: <≤ 6 words, no punctuation>\n\n"
    "STRICT RULES:\n"
    "1) Grounding: cite specific segment time ranges and/or frame indices.\n"
    "2) Visual first for actions/objects; text first for dialogue facts.\n"
    "3) No outside knowledge.\n"
    "4) Be concise; no punctuation in Final Answer.\n"
    "5) If insufficient, state so in Reasoning and guess succinctly.\n"
)

def build_two_step_messages_with_subs(
    frame_paths: List[Path],
    frame_times: List[float],
    question: str,
    subtitle_segments: List[Tuple[float, float, str]],
    template: str = TEMPLATE_VISION_TEXT_TWO_STEP,
    keyframe_hint_text: str = ""
) -> List[Dict[str, Any]]:
    """Classic layout: all images first, then one text block with subtitle list, frame legend, and instructions."""
    content = [{"type": "image", "image": str(p)} for p in frame_paths]
    sub_block = make_subtitle_block(subtitle_segments)
    frame_list = make_frame_list(frame_paths, frame_times or [])
    prompt_text = (
        template
        .replace("{SUBTITLE_BLOCK}", sub_block)
        .replace("{FRAME_LIST}", frame_list)
        .replace("{QUESTION}", question)
        .replace("{KEYFRAME_HINT}", (keyframe_hint_text + "\n") if keyframe_hint_text else "")
    )
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def build_two_step_messages_interleaved(
    frame_paths: List[Path],
    frame_times: List[float],
    question: str,
    subtitle_segments: List[Tuple[float, float, str]],
    keyframe_hint_text: str = "",
    segs_per_frame: int = 1,
    seg_radius: float = 2.0,
) -> List[Dict[str, Any]]:
    """Interleave: header text, then for each frame: <image> then a tiny text card with its nearest subtitle(s)."""
    content: List[Dict[str, Any]] = []

    # Header with question + format
    header = (
        TEMPLATE_INTERLEAVED_HEADER
        .replace("{QUESTION}", question)
        .replace("{KEYFRAME_HINT}", (keyframe_hint_text + "\n") if keyframe_hint_text else "")
    )
    content.append({"type": "text", "text": header})

    # For each frame, attach nearest subtitle(s)
    for i, p in enumerate(frame_paths):
        t = frame_times[i] if i < len(frame_times) else None
        content.append({"type": "image", "image": str(p)})
        lines = [f"Frame #{i+1}: {p.name}" + (f" (t≈{t:.1f}s)" if t is not None else "")]
        near = _segments_near_time(subtitle_segments, t, k=segs_per_frame, radius=seg_radius)
        if near:
            for (s,e,txt) in near:
                lines.append(f"Segment ({s:.2f}s–{e:.2f}s): {txt}")
        else:
            lines.append("[No nearby subtitle segment]")
        content.append({"type": "text", "text": "\n".join(lines)})

    return [{"role": "user", "content": content}]

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
    # Extract 'Reasoning:' and 'Final Answer:' lines robustly.
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
    # Enforce <= 6 words, strip punctuation
    if answer:
        answer = " ".join(answer.split()[:6])
        answer = re.sub(r"[^\w\s]", "", answer).strip()
    return reasoning, answer

# =======================
# Main
# =======================

def main(args):
    root = Path(args.root_dir)
    ann_dir = root / "annotations"
    out_root = root / "out_preprocessed"

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

    ems=[]; f1s=[]; b1s=[]; b4bps=[]; rls=[]; ecs=[]
    rls_reasoning = []
    ecs_reasoning = []
    processed = 0

    ann_files_all = sorted(ann_dir.glob("*.json"))
    rng = random.Random(args.seed)
    ann_files_shuf = ann_files_all[:]
    rng.shuffle(ann_files_shuf)
    cut = int(len(ann_files_shuf) * args.split_ratio)

    if args.split == "train":
        ann_files = ann_files_shuf[:cut]
    elif args.split == "eval":
        ann_files = ann_files_shuf[cut:]
    else:
        ann_files = ann_files_all

    print(f"[Info] Using {args.split.upper()} split: {len(ann_files)} movies")
    print(f"[Info] Found {len(ann_files)} movies in annotations/")
    for ann_file in ann_files:
        movie = ann_file.stem
        movie_root = out_root / movie
        if not movie_root.exists():
            print(f"[Skip] No out_preprocessed for {movie}")
            continue

        keyframes_dir = movie_root / "KEY_FRAMES"
        method = args.method  # e.g., blend_blur_with_last_frame
        # choose metadata for subtitles and method frames
        try:
            metadata_json = resolve_metadata_file(movie_root, method if args.pooling == "method" else None)
        except Exception as e:
            print(f"[Warn] {movie}: {e}")
            metadata_json = movie_root / "metadata_text_centric.json"  # best-effort fallback

        with open(ann_file, "r", encoding="utf-8") as f:
            items = json.load(f)

        for it in tqdm(items, desc = f"Processing {movie}"):
            try:
                q = it["question"].strip()
                gold = it["answer"].strip()
                ts_sec = hms_to_sec(it["timestamp"].strip())
                ctx_start, ctx_end = parse_context_ts(it["contextTimestamp"])
                idx = int(it["index"])

                # --- Subtitle segments (overlapping window) ---
                segs = []
                if metadata_json.exists():
                    segs = extract_segments_for_window(
                        metadata_json, ctx_start, ctx_end, max_segments=args.max_segments
                    )

                # --- Frames + times ---
                frame_paths: List[Path] = []
                frame_times: List[float] = []

                if args.pooling == "keyframe":
                    frame_paths, frame_times = collect_keyframe_by_index_with_time(keyframes_dir, idx)
                    if (not frame_paths) and method:
                        # fallback to closest method frame
                        if metadata_json.exists():
                            frame_paths, frame_times = collect_method_frames_with_times(
                                movie_root, method, metadata_json, ts_sec, ctx_start, ctx_end, max_frames=1
                            )
                elif args.pooling == "method":
                    if not method:
                        raise ValueError("When pooling=method you must pass --method")
                    if not metadata_json.exists():
                        raise FileNotFoundError(f"No metadata for method={method} in {movie_root}")

                    frame_paths, frame_times = collect_method_frames_with_times(
                        movie_root, method, metadata_json, ts_sec, ctx_start, ctx_end,
                        max_frames=args.max_frames,
                        frame_selection=args.frame_selection
                    )
                else:
                    raise ValueError("pooling must be one of {keyframe, method}")

                # --- Append keyframe (optional) & Build messages ---
                keyframe_hint_text = ""
                if args.append_keyframe:
                    kf_paths, kf_times = collect_keyframe_by_index_with_time(keyframes_dir, idx)
                    if kf_paths:
                        if str(kf_paths[0]) not in {str(p) for p in frame_paths}:
                            frame_paths.append(kf_paths[0])
                            frame_times.append(kf_times[0] if kf_times and kf_times[0] is not None else None)
                    if args.keyframe_hint and kf_paths:
                        approx_t = f"{kf_times[0]:.1f}s" if kf_times and kf_times[0] is not None else "the shown keyframe time"
                        keyframe_hint_text = f"Note: The user asked the question while paused at the keyframe {kf_paths[0].name} (t≈{approx_t})."

                if not frame_paths:
                    pred_reason, pred_answer = "", ""
                else:
                    if args.interleave:
                        messages = build_two_step_messages_interleaved(
                            frame_paths=frame_paths,
                            frame_times=frame_times,
                            question=q,
                            subtitle_segments=segs,
                            keyframe_hint_text=keyframe_hint_text,
                            segs_per_frame=args.segs_per_frame,
                            seg_radius=args.seg_radius,
                        )
                    else:
                        messages = build_two_step_messages_with_subs(
                            frame_paths=frame_paths,
                            frame_times=frame_times,
                            question=q,
                            subtitle_segments=segs,
                            keyframe_hint_text=keyframe_hint_text,
                        )

                    raw = generate_two_step(
                        model, processor, messages,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=args.do_sample
                    )

                    pred_reason, pred_answer = parse_two_step_output(raw)

                # --- Metrics on Final Answer ---
                emv = em(pred_answer, gold)
                f1v = f1(pred_answer, gold)
                b1, _b4, b4bp = bleu_scores(pred_answer, gold)
                rl = rouge_l(pred_answer, gold)
                ec = embsim.score(pred_answer, gold)

                ems.append(emv); f1s.append(f1v); b1s.append(b1); b4bps.append(b4bp); rls.append(rl); ecs.append(ec)

                gold_reason = ""
                if "reasoning" in it and it["reasoning"]:
                    gold_reason = it["reasoning"].strip()
                    rl_r = rouge_l(pred_reason, gold_reason)
                    ec_r = embsim.score(pred_reason, gold_reason)
                    rls_reasoning.append(rl_r)
                    ecs_reasoning.append(ec_r)

                rec = {
                    "movie": movie,
                    "index": idx,
                    "timestamp": it["timestamp"],
                    "contextTimestamp": it["contextTimestamp"],
                    "question": q,
                    "gold_reasoning": gold_reason,
                    "gold_answer": gold,
                    "pred_reasoning": pred_reason,
                    "pred_answer": pred_answer,
                    "pooling": args.pooling,
                    "method": method if args.pooling=="method" else "KEY_FRAMES",
                    "frames_used": [str(p) for p in frame_paths],
                    "frame_times": frame_times,
                    "subtitle_segments": [
                        {"start": s, "end": e, "text": t} for (s,e,t) in segs
                    ],
                }

                fout.write(json.dumps(rec) + "\n")
                processed += 1
                if args.limit and processed >= args.limit:
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[Error] {movie} idx={it.get('index')} -> {e}")

        if args.limit and processed >= args.limit:
            break

    fout.close()

    def avg(xs): return float(sum(xs)/max(1,len(xs)))
    summary = {
        "runs": processed,
        "pooling": args.pooling,
        "method": args.method if args.pooling=="method" else "KEY_FRAMES",
        "max_frames": args.max_frames,
        "max_segments": args.max_segments,
        "metrics": {
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
    ap = argparse.ArgumentParser("Movie VQA two-step LVLM baseline with subtitles (vision+text)")
    ap.add_argument("--root_dir", type=str, required=True, help="Project root with annotations/ and out_preprocessed/")
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--use_4bit", action="store_true", help="Enable 4-bit loading via bitsandbytes")
    ap.add_argument("--pooling", type=str, choices=["keyframe","method"], default="method",
                    help="'keyframe' uses KEY_FRAMES by index; 'method' uses a method folder + metadata mapping.")
    ap.add_argument("--method", type=str, default="blend_blur_with_last_frame",
                    help="One of {blend_blur_with_last_frame, weighted_average, weighted_average_exponential, weighted_average_ramp} or your custom.")
    ap.add_argument("--max_frames", type=int, default=3, help="If pooling=method, how many frames to pass (1 ⇒ blur-like).")
    ap.add_argument("--max_segments", type=int, default=8, help="Max subtitle segments to include from the window.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output_file", type=str, required=True)
    ap.add_argument("--frame_selection", type=str, choices=["near_ts","uniform","all"], default="near_ts",
                    help="How to pick frames from context window for method pooling.")
    ap.add_argument("--append_keyframe", action="store_true",
                    help="Append the keyframe (by index) after method frames.")
    ap.add_argument("--keyframe_hint", action="store_true",
                    help="Add a one-line note that the user was paused at the keyframe.")
    ap.add_argument("--interleave", action="store_true",
                    help="Interleave frames with their nearest subtitle snippets instead of a single text block.")
    ap.add_argument("--segs_per_frame", type=int, default=1,
                    help="Number of subtitle segments to attach to each frame when interleaving.")
    ap.add_argument("--seg_radius", type=float, default=2.0,
                    help="Max seconds from a frame time to pull a non-overlapping subtitle segment.")

    ap.add_argument("--peft_adapter", type=str, default=None,
                    help="Path to a PEFT/LoRA adapter dir (e.g., models/sft-qwen7b-interleaved-16f/<method>)")
    ap.add_argument("--peft_merge", action="store_true",
                    help="Merge LoRA weights into the base at load time")

    ap.add_argument("--split", type=str, choices=["all","train","eval"], default="all",
                    help="Movie-level split: all/train/eval (same deterministic split as training)")
    ap.add_argument("--split_ratio", type=float, default=0.9,
                    help="Fraction of movies for training part of the split (seeded)")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic movie split")
    args = ap.parse_args()

    main(args)
