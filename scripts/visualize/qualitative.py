#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative Figure — HTML Drop‑in (v2)

New in v2
---------
* **--max_frames**: cap and smart‑sample frames (prioritize KEY frames + uniform context).
* **--num_main**: in `--mode main`, emit multiple candidate HTMLs so you can pick your favorite.
* Optional **--prefer_diverse_movies** to diversify sampled examples across different movies.

Usage
-----
# one file with most frames
python qual_html_dropin.py \
  --records runs/*.jsonl --mode main \
  --out figs_html/qual_main.html --max_frames 48

# many candidates for review (e.g., 6)
python qual_html_dropin.py \
  --records runs/*.jsonl --mode main \
  --num_main 6 --out figs_html/qual_main.html --max_frames 48

# appendix: one file per (movie,index)
python qual_html_dropin.py \
  --records runs/*.jsonl --mode appendix \
  --out_dir figs_html/appendix --max_frames 48

Convert HTML → PDF/PNG via wkhtmltopdf/WeasyPrint/Chromium as before.
"""

import os, sys, json, argparse, math, re
from collections import defaultdict, OrderedDict
from typing import List, Tuple
from html import escape

# ---------- IO utils ----------

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def join_subs(segs: List[dict]) -> str:
    if not segs: return ''
    txt = ' '.join([s.get('text','').strip() for s in segs if s.get('text')])
    return ' '.join(txt.split())

def is_key_frame(p: str) -> bool:
    bn = os.path.basename(p)
    return ('KEY_FRAMES' in p) or bn.startswith('KEY_') or '_KEY_' in bn or 'KEY_' in bn

# ---------- Grouping & scoring ----------

def collect_pairs(records_with_src):
    buck = defaultdict(list)
    for rec, src in records_with_src:
        key = (str(rec.get('movie','')), str(rec.get('index','')))
        buck[key].append((rec, src))
    return buck

def norm(s: str) -> str:
    return ' '.join((s or '').strip().lower().split())

def score_example(rows, min_reason_len, max_reason_len, min_answer_len=0, favor_correct=2.5, center_len_bias=10.0):
    best_score, best_row = -1e9, None
    target_center = 0.5*(min_reason_len+max_reason_len)
    for rec, _ in rows:
        pr = (rec.get('pred_reasoning') or '').strip()
        pa = (rec.get('pred_answer') or '').strip()
        ga = (rec.get('gold_answer') or '').strip()
        rl = len(pr)
        if rl < min_reason_len or rl > max_reason_len: continue
        if len(pa) < min_answer_len: continue
        sc = center_len_bias - abs((rl - target_center)/20.0)
        if ga and norm(pa) == norm(ga): sc += favor_correct
        if sc > best_score: best_score, best_row = sc, rec
    return best_score, best_row

# ---------- Frame sampling ----------

def fmt_mmss(sec):
    if sec is None: return ''
    if isinstance(sec, (int, float)):
        m = int(sec//60); s = int(sec%60)
        return f"{m:02d}:{s:02d}"
    return str(sec)

def sample_frames(frames_used: List[str], frame_times: List[float], max_frames: int) -> List[Tuple[str,str,bool]]:
    """
    Return exactly `max_frames` frames whenever possible (N >= max_frames),
    prioritizing the first KEY frame + symmetric context, then filling
    uniformly from the remaining timeline. No duplicates; key need not be centered.
    """
    if not frames_used:
        return []

    # Build sortable tuples (path, time, idx, is_key)
    pts = []
    for i, p in enumerate(frames_used):
        t = frame_times[i] if i < len(frame_times) else None
        k = is_key_frame(p)
        pts.append((p, t, i, k))

    # Sort by time if available; else by path
    if any(t is not None for _, t, _, _ in pts):
        pts.sort(key=lambda x: (1e18 if x[1] is None else x[1]))
    else:
        pts.sort(key=lambda x: x[0])

    N = len(pts)
    # If fewer frames than desired, return all
    if N <= max_frames:
        return [(p, fmt_mmss(t), k) for (p, t, _, k) in pts]

    # ---- Phase 1: anchor on a KEY frame (or middle if no key), add local context
    key_idxs = [i for i, (_, _, _, k) in enumerate(pts) if k]
    anchor = key_idxs[0] if key_idxs else (N // 2)

    # Try to reserve up to ~17 around the anchor (8 before, 8 after, + anchor),
    # but clamp by boundaries.
    before = min(8, anchor)
    after  = min(8, N - 1 - anchor)

    selected = set(range(anchor - before, anchor + after + 1))  # contiguous block incl. anchor

    # If that already exceeds max_frames (very tiny max_frames), trim evenly to max_frames
    if len(selected) > max_frames:
        block = sorted(selected)
        # take evenly spaced indices from the block
        import numpy as _np
        keep_idx = set(int(round(x)) for x in _np.linspace(0, len(block) - 1, max_frames))
        selected = {block[i] for i in keep_idx}

    # ---- Phase 2: fill remaining slots uniformly from the rest (no duplicates)
    remaining = max_frames - len(selected)
    if remaining > 0:
        pool = [i for i in range(N) if i not in selected]
        # Uniform picks across pool using linspace -> unique ints
        import numpy as _np
        if len(pool) <= remaining:
            selected.update(pool)
        else:
            # evenly spread indices into pool
            idxs = [pool[int(round(x))] for x in _np.linspace(0, len(pool) - 1, remaining)]
            selected.update(idxs)

    # If collisions accidentally kept us short (paranoia), pad by scanning pool L->R
    if len(selected) < max_frames:
        for i in range(N):
            if i not in selected:
                selected.add(i)
                if len(selected) == max_frames:
                    break

    picked = sorted(selected)[:max_frames]  # hard cap
    return [(pts[i][0], fmt_mmss(pts[i][1]), pts[i][3]) for i in picked]

# ---------- HTML templates ----------

CSS = r'''
:root{ --page-w: 1024px; --gutter: 12px; --gap-after-subs: 22px; --gap-title-to-cols: 12px; --gap-header-to-box: 10px; --col-gap: 14px; --box-pad: 10px; --border: 1px solid rgba(0,0,0,0.35); --label:#0a7c32; --human:#164b9b; --model:#c46a00; --human-bg:rgba(70,130,180,.12); --model-bg:rgba(255,200,0,.16); --ts-bg:rgba(255,255,255,.85); --font: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
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
.figure:last-child{padding-bottom:8px}
'''

HTML_SHELL = r'''<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Qualitative Figure</title><style>{css}</style></head><body>{body}</body></html>'''

def fig_html(question: str, pooling: str, subtitles: str, frames: List[Tuple[str,str,bool]], human_reasoning: str, human_answer: str, model_blocks: List[Tuple[str,str,str]]) -> str:
    out = []
    out.append('<section class="figure">')
    out.append(f'<div class="h1">{escape("Question: "+question)}</div>')
    out.append('<div class="frames">')
    for src, ts, is_key in frames:
        cls = 'frame key' if is_key else 'frame'
        out.append(f'<div class="{cls}"><img src="../{escape(src)}" alt="frame"/><div class="ts">{escape(ts)}</div></div>')
    out.append('</div>')
    if pooling:
        out.append(f'<div class="meta small italic">Pooling method: {escape(pooling.replace("_"," "))}</div>')
    if subtitles:
        out.append(f'<div class="subs">{escape("Subtitles: "+subtitles)}</div>')

    # Reasoning
    out.append('<section class="section">')
    out.append('<h2 class="section-title">Reasoning Analysis:</h2>')
    out.append('<div class="section-inner cols multi">')
    out.append('<div class="human">')
    out.append('<p class="col-title human-title">Human Reference</p>')
    out.append(f'<div class="box human"><p>{escape(human_reasoning)}</p></div>')
    out.append('</div>')
    out.append('<div class="models">')
    for method, reasoning, _ in model_blocks:
        out.append('<div class="model">')
        out.append(f'<p class="col-title model-title">Model Output <span class="small italic">— {escape(method.replace("_"," "))}</span></p>')
        out.append(f'<div class="box model"><p>{escape(reasoning)}</p></div>')
        out.append('</div>')
    out.append('</div></div></section>')

    # Answers (immediately follows; no extra gap)
    out.append('<section class="section">')
    out.append('<h2 class="section-title">Answers:</h2>')
    out.append('<div class="section-inner cols multi">')
    out.append('<div class="human">')
    out.append('<p class="col-title human-title">Human Answer</p>')
    out.append(f'<div class="box human"><p>{escape(human_answer)}</p></div>')
    out.append('</div>')
    out.append('<div class="models">')
    for method, _, ans in model_blocks:
        out.append('<div class="model">')
        out.append(f'<p class="col-title model-title">Model Output <span class="small italic">— {escape(method.replace("_"," "))}</span></p>')
        out.append(f'<div class="box model"><p>{escape(ans)}</p></div>')
        out.append('</div>')
    out.append('</div></div></section>')

    out.append('</section>')
    return ''.join(out)

# ---------- Render one example ----------

def render_html_figure(example_rows, out_html_path, max_frames: int):
    base, base_score = None, -1
    for rec, _ in example_rows:
        sc = (2 if rec.get('frames_used') else 0) + (1 if rec.get('subtitle_segments') else 0)
        if sc > base_score:
            base_score, base = sc, rec
    if base is None:
        base = example_rows[0][0]

    question = base.get('question','').strip()
    pooling  = (base.get('method') or base.get('pooling') or '').strip()

    frames_used = base.get('frames_used') or []
    frame_times = base.get('frame_times') or []
    frames = sample_frames(frames_used, frame_times, max_frames=max_frames)

    subs = join_subs(base.get('subtitle_segments') or [])
    human_reasoning = base.get('gold_reasoning','').strip()
    human_answer    = base.get('gold_answer','').strip()

    methods = OrderedDict()
    for rec, _ in example_rows:
        mname = (rec.get('method') or rec.get('pooling') or '').strip()
        reasoning = (rec.get('pred_reasoning') or '').strip()
        answer = (rec.get('pred_answer') or '').strip()
        methods[mname] = (reasoning, answer)
    model_blocks = [(m, r, a) for m,(r,a) in methods.items()]

    body = fig_html(question, pooling, subs, frames, human_reasoning, human_answer, model_blocks)
    html = HTML_SHELL.format(css=CSS, body=body)
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, 'w', encoding='utf-8') as f:
        f.write(html)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', nargs='+', required=True)
    ap.add_argument('--mode', choices=['main','appendix'], required=True)
    ap.add_argument('--select_k', type=int, default=1)
    ap.add_argument('--num_main', type=int, default=1, help='How many figures to output in main mode')
    ap.add_argument('--min_reason_len', type=int, default=30)
    ap.add_argument('--max_reason_len', type=int, default=2000)
    ap.add_argument('--min_answer_len', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--pick_from_top_n', type=int, default=20)
    ap.add_argument('--prefer_diverse_movies', action='store_true')
    ap.add_argument('--max_frames', type=int, default=36)
    ap.add_argument('--out', type=str, default='figs_html/qual_main.html')
    ap.add_argument('--out_dir', type=str, default='figs_html/appendix')
    args = ap.parse_args()

    # Load
    recs_with_src = []
    for p in args.records:
        for r in load_jsonl(p):
            recs_with_src.append((r, p))
    if not recs_with_src:
        print('[ERROR] No records loaded', file=sys.stderr); sys.exit(1)

    buckets = collect_pairs(recs_with_src)

    # Score
    candidates = []
    for key, rows in buckets.items():
        sc, bestrow = score_example(rows, args.min_reason_len, args.max_reason_len, args.min_answer_len)
        if bestrow is not None:
            candidates.append((key, sc))

    import random
    random.seed(args.seed)
    candidates.sort(key=lambda x: x[1], reverse=True)

    def diversify(keys, k):
        if k >= len(keys): return [x for x,_ in keys]
        # group by movie
        by_movie = defaultdict(list)
        for (movie, idx), sc in keys:
            by_movie[movie].append(((movie, idx), sc))
        for mv in by_movie: by_movie[mv].sort(key=lambda x: x[1], reverse=True)
        picks, ptr = [], {m:0 for m in by_movie}
        order = sorted(by_movie, key=lambda m: -len(by_movie[m]))
        while len(picks) < k:
            progressed = False
            for m in order:
                i = ptr[m]
                if i < len(by_movie[m]):
                    picks.append(by_movie[m][i][0])
                    ptr[m] += 1
                    progressed = True
                    if len(picks) == k: break
            if not progressed: break
        return picks

    if args.mode == 'main':
        if not candidates:
            keys = [next(iter(buckets.keys()))]
        else:
            top_pool = candidates[:max(1, args.pick_from_top_n)]
            if args.prefer_diverse_movies:
                keys = diversify(top_pool, args.num_main)
            else:
                keys = [k for k,_ in random.sample(top_pool, k=min(args.num_main, len(top_pool)))]
        base_out = args.out
        root, ext = os.path.splitext(base_out)
        out_paths = []
        if args.num_main <= 1:
            out_paths = [base_out]
        else:
            out_paths = [f"{root}_{i+1}{ext or '.html'}" for i in range(len(keys))]
        for key, outp in zip(keys, out_paths):
            render_html_figure(buckets[key], outp, max_frames=args.max_frames)
            print(f'[OK] {outp}')
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        keys = [k for k,_ in (candidates or [(k,0) for k in buckets.keys()])]
        for key in keys:
            movie, idx = key
            out_html = os.path.join(args.out_dir, f'qual_{movie}_{idx}.html')
            render_html_figure(buckets[key], out_html, max_frames=args.max_frames)
        print(f'[OK] Wrote HTML files to {os.path.abspath(args.out_dir)}')

if __name__ == '__main__':
    main()
