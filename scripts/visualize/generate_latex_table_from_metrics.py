#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd

# ---------- Pretty helpers ----------
def prettify_method(name: str) -> str:
    if not name: return "N/A"
    name = name.strip().replace("_", " ")
    name = name.replace("blend blur with last frame", "Blend+Blur+Last")
    name = name.replace("weighted average exponential", "Weighted Avg (Exp)")
    name = name.replace("weighted average ramp", "Weighted Avg (Ramp)")
    name = name.replace("weighted average whatever5", "Weighted Avg (Var5)")
    name = name.replace("weighted average", "Weighted Avg")
    name = " ".join(name.split())
    return " ".join([w.capitalize() if w.isalpha() else w for w in name.split(" ")])

_METRIC_ALIASES = {
    "em":"EM","exact_match":"EM",
    "f1":"F1",
    "bleu":"BLEU-1","bleu1":"BLEU-1","bleu_1":"BLEU-1","bleu-1":"BLEU-1",
    "bleu4":"BLEU-4 (BP)","bleu_4":"BLEU-4 (BP)","bleu_4_bp":"BLEU-4 (BP)",
    "bleu4 bp":"BLEU-4 (BP)","bleu-4 (bp)":"BLEU-4 (BP)","bleu4bp":"BLEU-4 (BP)",
    "rouge":"ROUGE-L","rouge_l":"ROUGE-L","rouge-l":"ROUGE-L","rouge l":"ROUGE-L",
    "embedcosine":"Embed Cosine","embed cosine":"Embed Cosine",
    "embedding_cosine":"Embed Cosine","embedding cos":"Embed Cosine",
    # reasoning keys
    "rouge l reasoning":"ROUGE-L (Reasoning)","rouge-l (reasoning)":"ROUGE-L (Reasoning)",
    "rouge_l_reasoning":"ROUGE-L (Reasoning)",
    "embed cosine reasoning":"Embed Cosine (Reasoning)",
    "embedding_cosine_reasoning":"Embed Cosine (Reasoning)",
    "embedcosine reasoning":"Embed Cosine (Reasoning)",
    # your exact keys
    "embedcos":"Embed Cosine",
    "embedcos reasoning":"Embed Cosine (Reasoning)",
}
ORDER_TRAIN = ["Blend+Blur+Last","Weighted Avg","Weighted Avg (Exp)","Weighted Avg (Ramp)"]
ABBREV = {"Blend+Blur+Last":"BBLF","Weighted Avg":"WA","Weighted Avg (Exp)":"WAE","Weighted Avg (Ramp)":"WAR"}

ORDER_EVAL = ["Blend+Blur+Last","Weighted Avg","Weighted Avg (Exp)","Weighted Avg (Ramp)","Weighted Avg (Var5)"]
FRIENDLY = {
    "Blend+Blur+Last":"Blend Blur With Last Frame",
    "Weighted Avg":"Weighted Average",
    "Weighted Avg (Exp)":"Weighted Average (Exp)",
    "Weighted Avg (Ramp)":"Weighted Average (Ramp)",
    "Weighted Avg (Var5)":"Weighted Avg (Var5)",
}

ANS_METRICS = ["F1","BLEU-1","BLEU-4 (BP)","ROUGE-L","Embed Cosine"]
RSN_METRICS = ["ROUGE-L (Reasoning)","Embed Cosine (Reasoning)"]
LABEL = {
    "F1":"F1","BLEU-1":"BLEU-1","BLEU-4 (BP)":"BLEU-4 (BP)","ROUGE-L":"ROUGE-L",
    "Embed Cosine":"Embed Cosine","ROUGE-L (Reasoning)":"ROUGE-L-R","Embed Cosine (Reasoning)":"Embed Cosine-R",
}

def prettify_metric(key: str) -> str:
    if not key: return "?"
    k = key.lower().replace("_"," ").strip()
    return _METRIC_ALIASES.get(k, key)

def fmt_val(x, precision=3) -> Optional[str]:
    if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))): return None
    try: return f"{float(x):.{precision}f}"
    except Exception: return None

# ---------- Discovery ----------
def detect_roots(runs_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    sft_root = None; dpo_root = None
    for p in runs_dir.iterdir():
        if not p.is_dir(): continue
        name = p.name
        prefix = name.split("-",1)[0] if "-" in name else name
        if prefix == "sft" and sft_root is None: sft_root = p
        if prefix == "dpo" and dpo_root is None: dpo_root = p
    if sft_root is None and (runs_dir/"sft").is_dir(): sft_root = runs_dir/"sft"
    if dpo_root is None and (runs_dir/"dpo").is_dir(): dpo_root = runs_dir/"dpo"
    return sft_root, dpo_root

# ---------- IO ----------
def read_summary(fp: Path) -> Optional[Tuple[str, Dict[str,float]]]:
    try: data = json.loads(fp.read_text())
    except Exception: return None
    method = data.get("method","") or ""
    metrics = data.get("metrics",{}) or {}
    return method, metrics

def collect_matrix(root: Path, exclude_keyframe_eval: bool=True) -> pd.DataFrame:
    """rows = train_method, cols = MultiIndex (eval_method, metric)"""
    store: Dict[str, Dict[str, Dict[str,float]]] = {}
    all_metrics, all_eval_methods = set(), set()
    for train_dir in sorted([p for p in root.glob("*") if p.is_dir()]):
        train_method = prettify_method(train_dir.name)
        for eval_dir in sorted([p for p in train_dir.glob("*") if p.is_dir()]):
            if exclude_keyframe_eval and "keyframe" in eval_dir.name.lower(): continue
            eval_method = prettify_method(eval_dir.name)
            candidates = list(eval_dir.glob("*.summary.json")) + list(eval_dir.glob("*summary.json"))
            if not candidates:
                hidden = eval_dir/".summary.json"
                if hidden.exists(): candidates=[hidden]
            if not candidates: continue
            fp = max(candidates, key=lambda p: p.stat().st_mtime)
            parsed = read_summary(fp)
            if not parsed: continue
            _mname, metrics = parsed
            store.setdefault(train_method, {}).setdefault(eval_method, {})
            for k,v in metrics.items():
                nice = prettify_metric(k)
                if nice == "EM": continue  # drop EM everywhere
                store[train_method][eval_method][nice] = v
                all_metrics.add(nice)
            all_eval_methods.add(eval_method)
    if not store: return pd.DataFrame()

    evals = [e for e in ORDER_EVAL if e in all_eval_methods] + [e for e in sorted(all_eval_methods) if e not in ORDER_EVAL]
    metrics_sorted = [m for m in ANS_METRICS + RSN_METRICS if m in all_metrics]

    rows = {}
    for train_method, m_by_eval in store.items():
        row = {}
        for e in evals:
            mdict = m_by_eval.get(e,{})
            for met in metrics_sorted:
                row[(e,met)] = pd.to_numeric(mdict.get(met, float("nan")), errors="coerce")
        rows[train_method] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df = df.reindex(columns=pd.MultiIndex.from_product([evals, metrics_sorted]))
    return df

# ---------- Highlight (colors only, no bold) ----------
def highlight_row(values: List[Optional[float]], fmt_vals: List[Optional[str]]) -> List[str]:
    ser = pd.Series(values, dtype="float64")
    valid = ser.dropna()
    if valid.empty:
        return ["—" if s is None else s for s in fmt_vals]
    order = valid.sort_values(ascending=False).index.tolist()
    top1 = order[0]
    top2 = order[1] if len(order) >= 2 else None
    out = []
    for i, s in enumerate(fmt_vals):
        if s is None:
            out.append("—"); continue
        if i == top1: out.append(f"\\cellcolor{{high1}}{s}")
        elif top2 is not None and i == top2: out.append(f"\\cellcolor{{high2}}{s}")
        else: out.append(s)
    return out

# ---------- LaTeX (EXACT requested style; one header per eval block) ----------
def latex_table_exact(df: pd.DataFrame, caption: str, label: str, precision: int=3) -> str:
    if df.empty:
        return "% No cross-eval results found."

    trains = [t for t in ORDER_TRAIN if t in df.index] or list(df.index)
    abbrev = [ABBREV.get(t, t) for t in trains]
    n_cols = len(trains)

    present_evals = list(dict.fromkeys([c[0] for c in df.columns]))
    evals = [e for e in ORDER_EVAL if e in present_evals] + [e for e in present_evals if e not in ORDER_EVAL]

    def get_val(t: str, e: str, m: str):
        if (e, m) not in df.columns or t not in df.index: return float("nan")
        return pd.to_numeric(df.loc[t, (e, m)], errors="coerce")

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l" + "c"*n_cols + "}")
    lines.append("\\toprule")
    lines.append(f"& \\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{Model Trained On}}}} \\\\")
    lines.append(f"\\cmidrule(l){{2-{n_cols+1}}}")
    lines.append("Metric & " + " & ".join([f"\\makecell[ct]{{{a}}}" for a in abbrev]) + " \\\\")
    lines.append("\\midrule\n")

    for e in evals:
        friendly = FRIENDLY.get(e, e)
        lines.append(f"\\multicolumn{{{n_cols+1}}}{{l}}{{\\textit{{Evaluated on: {friendly}}}}} \\\\")
        lines.append("\\midrule")
        # Answer metrics
        for met in [m for m in ANS_METRICS if (e, m) in df.columns]:
            vals = [get_val(t, e, met) for t in trains]
            svals = [fmt_val(v, precision) for v in vals]
            cells = highlight_row(vals, svals)
            for j, t in enumerate(trains):
                if t == e and cells[j] != "—":
                    cells[j] = f"\\cellcolor{{gray!15}} {cells[j]}"
            lines.append(f"{LABEL.get(met, met)}& " + " & ".join(cells) + " \\\\")
        # Reasoning metrics under the SAME header
        for met in [m for m in RSN_METRICS if (e, m) in df.columns]:
            vals = [get_val(t, e, met) for t in trains]
            svals = [fmt_val(v, precision) for v in vals]
            cells = highlight_row(vals, svals)
            for j, t in enumerate(trains):
                if t == e and cells[j] != "—":
                    cells[j] = f"\\cellcolor{{gray!15}} {cells[j]}"
            lines.append(f"{LABEL.get(met, met)}& " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule\n")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Emit SFT and DPO LaTeX tables (exact sample format; colors only; no bold).")
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--precision", type=int, default=3)
    ap.add_argument("--include_keyframe_evals", action="store_true",
                    help="Include eval dirs containing 'keyframe' (default: excluded).")
    # Captions & labels (SFT caption matches your sample EXACTLY)
    ap.add_argument("--caption_sft", type=str, default=(
        "Cross-evaluation of fine-tuned models after SFT in \\textbf{ReasonVQA eval set}. "
        "Highlights: \\colorbox{high1}{highest}, \\colorbox{high2}{second-highest}, "
        "\\colorbox{gray!15}{same training and evaluation method}. Method abbreviations: "
        "BBLF (Blend Blur Last Frame), WA (Weighted Avg), WAE (Weighted Avg Exp), WAR (Weighted Avg Ramp), "
        "Metric-R (Metric- Reasoning)"
    ))
    ap.add_argument("--label_sft", type=str, default="tab:sft_cross_eval_fancy")
    ap.add_argument("--caption_dpo", type=str, default=(
        "Cross-evaluation of fine-tuned models after DPO in \\textbf{ReasonVQA eval set}. "
        "Highlights: \\colorbox{high1}{highest}, \\colorbox{high2}{second-highest}, "
        "\\colorbox{gray!15}{same training and evaluation method}. Method abbreviations: "
        "BBLF (Blend Blur Last Frame), WA (Weighted Avg), WAE (Weighted Avg Exp), WAR (Weighted Avg Ramp), "
        "Metric-R (Metric- Reasoning)"
    ))
    ap.add_argument("--label_dpo", type=str, default="tab:dpo_cross_eval_fancy")
    # Outputs
    ap.add_argument("--out_sft", type=str, required=True, help="Write SFT table to this .tex file.")
    ap.add_argument("--out_dpo", type=str, required=True, help="Write DPO table to this .tex file.")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    sft_root, dpo_root = detect_roots(runs_dir)
    if sft_root is None and dpo_root is None:
        raise SystemExit(f"Error: no 'sft' or 'dpo' roots under {runs_dir}")

    def build(root: Path, caption: str, label: str) -> str:
        df = collect_matrix(root, exclude_keyframe_eval=not args.include_keyframe_evals)
        return latex_table_exact(df, caption, label, precision=args.precision)

    if sft_root is not None:
        sft_tex = build(sft_root, args.caption_sft, args.label_sft)
        Path(args.out_sft).write_text(sft_tex)
    if dpo_root is not None:
        dpo_tex = build(dpo_root, args.caption_dpo, args.label_dpo)
        Path(args.out_dpo).write_text(dpo_tex)

    # Console echo
    print("="*88, "\nSFT TABLE →", args.out_sft, "\n", "="*88)
    if sft_root is not None: print(Path(args.out_sft).read_text())
    print("\n","="*88, "\nDPO TABLE →", args.out_dpo, "\n", "="*88)
    if dpo_root is not None: print(Path(args.out_dpo).read_text())
    print("\nLaTeX deps:\n\\usepackage{booktabs}\n\\usepackage{makecell}\n\\usepackage[table]{xcolor}\n")

if __name__ == "__main__":
    main()
