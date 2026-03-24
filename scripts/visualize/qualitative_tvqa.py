#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative Figure (TVQA) — HTML generator

• Guarantees exactly --max_frames frames when available (KEY+context+uniform).
• Shows question, time span, show/clip IDs, subtitles, MC options (gold + model).
• Human vs Model reasoning/answers in matched two-column layout.
• Writes a sibling *.imagelist.txt with the selected frame paths.
• Can emit multiple candidates in main mode (--num_main), optionally diversifying by show.

Usage:
  # Single main html (36 frames)
  python qualitative_tvqa.py --records runs_tvqa/*.jsonl --mode main \
    --max_frames 36 --out figs_tvqa/qual_tvqa.html

  # Generate several candidates to choose from
  python qualitative_tvqa.py --records runs_tvqa/*.jsonl --mode main \
    --num_main 6 --max_frames 36 --out figs_tvqa/qual_tvqa.html \
    --prefer_diverse_shows

  # Appendix (one per example)
  python qualitative_tvqa.py --records runs_tvqa/*.jsonl --mode appendix \
    --out_dir figs_tvqa/appendix --max_frames 36
"""

import os, sys, json, argparse, re
from collections import defaultdict, OrderedDict
from typing import List, Tuple
from html import escape

# ========================= IO / parsing =========================

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def join_subs(segs: List[dict]) -> str:
    if not segs: return ""
    txt = " ".join([s.get("text","").strip() for s in segs if s.get("text")])
    return " ".join(txt.split())

def is_key_frame(p: str) -> bool:
    bn = os.path.basename(p)
    return ("KEY_FRAMES" in p) or bn.startswith("KEY_") or "_KEY_" in bn or "KEY_" in bn

def fmt_mmss(sec):
    if sec is None: return ""
    try:
        sec = float(sec)
    except Exception:
        return str(sec)
    m = int(sec//60); s = int(sec%60)
    return f"{m:02d}:{s:02d}"

# ========================= grouping & scoring =========================

def collect_pairs(records_with_src):
    # Group by (vid_name, qid) so multiple method predictions align on the same example
    buck = defaultdict(list)
    for rec, src in records_with_src:
        key = (str(rec.get("vid_name","")), str(rec.get("qid","")))
        buck[key].append((rec, src))
    return buck

def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def score_example(rows, min_reason_len=20, max_reason_len=10_000):
    # Prefer examples with a reasonable model reasoning length and correctness
    best_score, best_row = -1e9, None
    for rec, _ in rows:
        pr = (rec.get("pred_reasoning") or "").strip()
        correct = bool(rec.get("correct", False) or (rec.get("gold_idx") == rec.get("pred_idx")))
        rl = len(pr)
        if rl < min_reason_len or rl > max_reason_len:
            continue
        sc = 10.0
        sc += min(8.0, rl/100.0)              # gently reward longer reasoning (up to ~800 chars)
        if correct: sc += 3.0                  # prefer a clean correct example
        if rec.get("subtitle_segments"): sc += 1.0
        if rec.get("frames_used"): sc += 2.0
        if sc > best_score:
            best_score, best_row = sc, rec
    return best_score, best_row

# ========================= frame sampling (exact count) =========================

def sample_frames(frames_used: List[str], frame_times: List[float], max_frames: int) -> List[Tuple[str,str,bool]]:
    """
    Return exactly max_frames frames whenever possible (N >= max_frames),
    prioritizing the first KEY frame + symmetric local context, then filling
    uniformly from the remaining timeline. No duplicates; key need not be centered.
    """
    if not frames_used:
        return []

    pts = []
    for i, p in enumerate(frames_used):
        t = frame_times[i] if i < len(frame_times) else None
        k = is_key_frame(p)
        pts.append((p, t, i, k))

    # Sort by time if present else by path
    if any(t is not None for _, t, _, _ in pts):
        pts.sort(key=lambda x: (1e18 if x[1] is None else x[1]))
    else:
        pts.sort(key=lambda x: x[0])

    # Deduplicate paths that appear repeated in TVQA exports
    dedup = []
    seen = set()
    for p,t,i,k in pts:
        if (p,t) in seen: continue
        seen.add((p,t))
        dedup.append((p,t,i,k))
    pts = dedup

    N = len(pts)
    if N <= max_frames:
        return [(p, fmt_mmss(t), k) for (p, t, _, k) in pts]

    key_idxs = [i for i, (_,_,_,k) in enumerate(pts) if k]
    anchor = key_idxs[0] if key_idxs else (N // 2)

    before = min(8, anchor)
    after  = min(8, N - 1 - anchor)
    selected = set(range(anchor - before, anchor + after + 1))

    if len(selected) > max_frames:
        block = sorted(selected)
        # evenly pick max_frames from the contiguous block
        import numpy as _np
        keep_idx = set(int(round(x)) for x in _np.linspace(0, len(block)-1, max_frames))
        selected = {block[i] for i in keep_idx}

    remaining = max_frames - len(selected)
    if remaining > 0:
        pool = [i for i in range(N) if i not in selected]
        import numpy as _np
        if len(pool) <= remaining:
            selected.update(pool)
        else:
            idxs = [pool[int(round(x))] for x in _np.linspace(0, len(pool)-1, remaining)]
            selected.update(idxs)

    if len(selected) < max_frames:
        for i in range(N):
            if i not in selected:
                selected.add(i)
                if len(selected) == max_frames:
                    break

    picked = sorted(selected)[:max_frames]
    return [(pts[i][0], fmt_mmss(pts[i][1]), pts[i][3]) for i in picked]

# ========================= HTML =========================

CSS = r"""
:root{ --page-w: 1024px; --gutter: 12px; --gap-after-subs: 16px; --gap-title-to-cols: 10px; --gap-header-to-box: 8px; --col-gap: 14px; --box-pad: 10px; --border: 1px solid rgba(0,0,0,0.35); --label:#0a7c32; --human:#164b9b; --model:#c46a00; --human-bg:rgba(70,130,180,.12); --model-bg:rgba(255,200,0,.16); --ts-bg:rgba(255,255,255,.85); --font: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
*{box-sizing:border-box}
body{margin:0; font-family:var(--font); color:#111}
.figure{width:var(--page-w); margin:0 auto; padding:12px}
.h1{font-size:20px; font-weight:800; margin:0 0 8px 0}
.small{font-size:12px; color:#555}
.italic{font-style:italic}
.frames{display:grid; grid-template-columns:repeat(12,1fr); grid-gap:6px; align-items:stretch}
.frame{position:relative; border:1px solid #ddd; overflow:hidden}
.frame img{display:block; width:100%; height:100%; object-fit:cover}
.frame .ts{position:absolute; top:4px; right:4px; font-size:10px; padding:2px 4px; background:var(--ts-bg); border-radius:3px}
.frame.key{outline:3px solid #d00}
.frame.key::after{content:'KEY'; position:absolute; left:6px; bottom:6px; font-size:10px; padding:2px 4px; background:#d00; color:#fff; border-radius:2px}
.meta{margin:6px 0 0 0}
.subs{margin:4px 0 var(--gap-after-subs) 0; font-size:13px}
.section{margin:0; padding:0}
.section-title{color:var(--label); font-size:16px; font-weight:800; margin:0}
.section-inner{margin-top:var(--gap-title-to-cols)}
.cols{display:grid; grid-template-columns:1fr 1fr; grid-gap:var(--col-gap)}
.cols.multi .models{display:grid; grid-template-columns:repeat(auto-fit, minmax(0,1fr)); grid-gap:var(--col-gap)}
.col-title{font-size:13px; font-weight:800; margin:0}
.human-title{color:var(--human)}
.model-title{color:var(--model)}
.box{margin-top:var(--gap-header-to-box); border:var(--border); padding:var(--box-pad); border-radius:6px; background:#fff}
.box.human{background:var(--human-bg); border-color:#2c5fae}
.box.model{background:var(--model-bg); border-color:#a85e00}
.box p{font-size:13px; line-height:1.35; margin:0}

/* MC options embedded inside the human box */
.box .options{margin-top:8px; font-size:13px}
.box .options table{border-collapse:collapse; width:100%}
.box .options td{border:1px solid #ccc; padding:6px 8px; vertical-align:top}
.box .options td:first-child{font-weight:700; width:36px; text-align:center}
.box .options .gold{background:rgba(10,124,50,.12)}
.box .options .pred{outline:2px dashed rgba(196,106,0,.7)}

.figure:last-child{padding-bottom:8px}
.choice{ margin-top:6px; font-size:12px; color:#333 }
.choice b{ font-weight:800 }
"""
HTML_SHELL = """<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TVQA Qualitative Figure</title><style>{css}</style></head><body>{body}</body></html>"""

def idx_to_letter(i: int) -> str:
    if i is None: return ""
    letters = "ABCDE"
    if 0 <= i < len(letters): return letters[i]
    return ""

def mc_block(options, gold_idx, pred_idx):
    letters = "ABCDE"
    rows = []
    for i, opt in enumerate(options):
        cls = []
        if i == gold_idx: cls.append("gold")
        if pred_idx is not None and i == pred_idx: cls.append("pred")
        c = (' class="' + " ".join(cls) + '"') if cls else ""
        rows.append(f"<tr{c}><td>{letters[i]}</td><td>{escape(opt)}</td></tr>")
    return '<div class="options"><table>' + "\n".join(rows) + "</table></div>"

def fig_html(rec_base, frames, human_reasoning, human_answer, model_reasoning, model_answer):
    q = rec_base.get("question","").strip()
    show = rec_base.get("show_name","").strip()
    clip = rec_base.get("vid_name","").strip()
    ts = rec_base.get("ts","").strip()
    method = (rec_base.get("method") or "").replace("_"," ")
    subs = join_subs(rec_base.get("subtitle_segments") or [])
    options = rec_base.get("options") or []
    gold_idx = rec_base.get("gold_idx")
    pred_idx = rec_base.get("pred_idx")

    out = []
    out.append('<section class="figure">')
    out.append(f'<div class="h1">{escape("Question: "+q)}</div>')
    out.append(f'<div class="small meta"><b>Show:</b> {escape(show)} &nbsp;•&nbsp; <b>Clip:</b> {escape(clip)} &nbsp;•&nbsp; <b>Span:</b> {escape(ts)}</div>')
    out.append('<div class="frames">')
    for src, ts_s, is_key in frames:
        cls = "frame key" if is_key else "frame"
        out.append(f'<div class="{cls}"><img src="../{escape(src)}" alt="frame"/><div class="ts">{escape(ts_s)}</div></div>')
    out.append('</div>')
    if method:
        out.append(f'<div class="small meta italic">Pooling method: {escape(method)}</div>')
    if subs:
        out.append(f'<div class="subs">{escape("Subtitles: "+subs)}</div>')

    # if options:
    #     out.append(mc_block(options, gold_idx, pred_idx))

    # Reasoning
    out.append('<section class="section">')
    out.append('<h2 class="section-title">Reasoning Analysis:</h2>')
    out.append('<div class="section-inner cols multi">')

    # Human col: reasoning + embedded options table
    out.append('<div class="human">')
    out.append('<p class="col-title human-title">Human Reference</p>')
    out.append('<div class="box human">')
    out.append(f'<p>{escape(human_reasoning)}</p>')
    if options:
        out.append(mc_block(options, gold_idx, pred_idx))
    out.append('</div></div>')

    # Model col
    # Model col
    out.append('<div class="models"><div class="model">')
    out.append('<p class="col-title model-title">Model Output</p>')
    out.append(f'<div class="box model"><p>{escape(model_reasoning)}</p></div>')
    # Model’s choice (letter) right under the box
    if options and isinstance(pred_idx, int) and 0 <= pred_idx < len(options):
        out.append(f'<div class="choice">Model’s Choice: <b>{idx_to_letter(pred_idx)}</b></div>')
    out.append('</div></div>')

    out.append('</div></section>')

    out.append('</section>')
    return "\n".join(out)

# ========================= rendering =========================

def render_html_figure(example_rows, out_html_path, max_frames: int):
    # pick the base row (has richest data)
    base, base_score = None, -1
    for rec, _ in example_rows:
        sc = (2 if rec.get("frames_used") else 0) + (1 if rec.get("subtitle_segments") else 0)
        if sc > base_score:
            base_score, base = sc, rec
    if base is None:
        base = example_rows[0][0]

    frames = sample_frames(base.get("frames_used") or [], base.get("frame_times") or [], max_frames)

    human_reasoning = base.get("gold_text","").strip()  # TVQA does not have explicit gold reasoning; use gold_text as reference explanation
    if not human_reasoning:
        # fallback: brief sentence from gold_text
        human_reasoning = base.get("gold_text","").strip()

    human_answer = base.get("gold_text","").strip()
    model_reasoning = (base.get("pred_reasoning") or "").strip()
    model_answer = (base.get("pred_text") or base.get("pred_answer_raw") or "").strip()

    body = fig_html(base, frames, human_reasoning, human_answer, model_reasoning, model_answer)
    html = HTML_SHELL.format(css=CSS, body=body)

    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Write imagelist for quick scp
    imglist_path = os.path.splitext(out_html_path)[0] + ".imagelist.txt"
    with open(imglist_path, "w", encoding="utf-8") as f:
        for p, _, _ in frames:
            f.write(p + "\n")
    return [p for p,_,_ in frames]

# ========================= CLI =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--mode", choices=["main","appendix"], required=True)
    ap.add_argument("--num_main", type=int, default=1)
    ap.add_argument("--min_reason_len", type=int, default=20)
    ap.add_argument("--max_reason_len", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pick_from_top_n", type=int, default=20)
    ap.add_argument("--prefer_diverse_shows", action="store_true")
    ap.add_argument("--max_frames", type=int, default=36)
    ap.add_argument("--out", type=str, default="figs_tvqa/qual_tvqa.html")
    ap.add_argument("--out_dir", type=str, default="figs_tvqa/appendix")
    args = ap.parse_args()

    # Load
    recs_with_src = []
    for p in args.records:
        for r in load_jsonl(p):
            recs_with_src.append((r, p))
    if not recs_with_src:
        print("[ERROR] No records loaded", file=sys.stderr)
        sys.exit(1)

    buckets = collect_pairs(recs_with_src)

    # Score candidates
    cand = []
    for key, rows in buckets.items():
        sc, bestrow = score_example(rows, args.min_reason_len, args.max_reason_len)
        if bestrow is not None:
            cand.append((key, sc))

    import random
    random.seed(args.seed)
    cand.sort(key=lambda x: x[1], reverse=True)

    # Simple diversify by show_name
    def diversify(keys, k):
        if k >= len(keys): return [x for x,_ in keys]
        by_show = defaultdict(list)
        for (vid, qid), sc in keys:
            # fetch one row to read show_name
            show = buckets[(vid, qid)][0][0].get("show_name","")
            by_show[show].append(((vid, qid), sc))
        for sh in by_show: by_show[sh].sort(key=lambda x: x[1], reverse=True)
        picks, ptr = [], {s:0 for s in by_show}
        order = sorted(by_show, key=lambda s: -len(by_show[s]))
        while len(picks) < k:
            progressed = False
            for s in order:
                i = ptr[s]
                if i < len(by_show[s]):
                    picks.append(by_show[s][i][0])
                    ptr[s] += 1
                    progressed = True
                    if len(picks) == k: break
            if not progressed: break
        return picks

    if args.mode == "main":
        if not cand:
            keys = [next(iter(buckets.keys()))]
        else:
            top_pool = cand[:max(1, args.pick_from_top_n)]
            if args.prefer_diverse_shows:
                keys = diversify(top_pool, args.num_main)
            else:
                keys = [k for k,_ in random.sample(top_pool, k=min(args.num_main, len(top_pool)))]
        base_out = args.out
        root, ext = os.path.splitext(base_out)
        outs = [base_out] if args.num_main <= 1 else [f"{root}_{i+1}{ext or '.html'}" for i in range(len(keys))]
        for key, outp in zip(keys, outs):
            frames = render_html_figure(buckets[key], outp, args.max_frames)
            print(f"[OK] {outp}  (frames: {len(frames)})")
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        keys = [k for k,_ in (cand or [(k,0) for k in buckets.keys()])]
        for key in keys:
            vid, qid = key
            out_html = os.path.join(args.out_dir, f"qual_tvqa_{vid}_{qid}.html")
            frames = render_html_figure(buckets[key], out_html, args.max_frames)
        print(f"[OK] Wrote HTML files to {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
