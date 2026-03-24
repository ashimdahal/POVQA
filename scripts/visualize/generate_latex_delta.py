#!/usr/bin/env python3
"""
Build a DPO vs. SFT Δ table (by evaluation pooler, best over training poolers) with color highlights.

Usage:
  python make_delta_table.py \
      --sft_tex sft_reasonvqa.tex \
      --dpo_tex dpo_reasonvqa.tex \
      --out_tex dpo_vs_sft_delta.tex
"""

import re
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ---- Canonical names & orders ----
ORDER_TRAIN = ["Blend+Blur+Last", "Weighted Avg", "Weighted Avg (Exp)", "Weighted Avg (Ramp)"]
ABBREV = {"Blend+Blur+Last":"BBLF","Weighted Avg":"WA","Weighted Avg (Exp)":"WAE","Weighted Avg (Ramp)":"WAR"}

EVAL_FRIENDLY = {
    "Blend+Blur+Last": "Blend Blur With Last Frame",
    "Weighted Avg": "Weighted Average",
    "Weighted Avg (Exp)": "Weighted Average (Exp)",
    "Weighted Avg (Ramp)": "Weighted Average (Ramp)",
    "Weighted Avg (Var5)": "Weighted Avg (Var5)",
}
# Only these four eval poolers in the delta table columns:
EVAL_ORDER = ["Blend+Blur+Last", "Weighted Avg", "Weighted Avg (Exp)", "Weighted Avg (Ramp)"]

# Rows to show in the delta table (answer then reasoning)
METRICS_ANS = ["F1","BLEU-1","BLEU-4 (BP)","ROUGE-L","Embed Cosine"]
METRICS_RSN_SHORT = ["ROUGE-L-R","Embed Cosine-R"]  # short labels for output

# Map row label as seen in SFT/DPO .tex -> canonical metric key
ROW_LABEL_TO_CANON = {
    "F1": "F1",
    "BLEU-1": "BLEU-1",
    "BLEU-4 (BP)": "BLEU-4 (BP)",
    "ROUGE-L": "ROUGE-L",
    "Embed Cosine": "Embed Cosine",
    "ROUGE-L-R": "ROUGE-L (Reasoning)",
    "Embed Cosine-R": "Embed Cosine (Reasoning)",
}
# Canonical -> short label for delta table output
CANON_TO_SHORT = {
    "F1": "F1",
    "BLEU-1": "BLEU-1",
    "BLEU-4 (BP)": "BLEU-4 (BP)",
    "ROUGE-L": "ROUGE-L",
    "Embed Cosine": "Embed Cos",
    "ROUGE-L (Reasoning)": "ROUGE-L-R",
    "Embed Cosine (Reasoning)": "Embed Cos-R",
}

# -------- parsing helpers --------
_float_pat = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')

def _clean_cell_to_float(cell: str) -> Optional[float]:
    """Extract the right-most number from a LaTeX cell, ignoring \\cellcolor etc."""
    matches = _float_pat.findall(cell)
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except ValueError:
        return float("nan")

def _strip_macro(s: str) -> str:
    return s.replace("\\makecell[ct]{", "").replace("}", "").strip()

def parse_cross_eval_tex(tex_path: Path) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
    """
    Parse a cross-eval .tex (your exact format) into:
      - columns: list of training column abbreviations (e.g., ['BBLF','WA','WAE','WAR'])
      - data: dict[eval_friendly][metric_canonical] -> list of floats (len=columns)
    """
    text = tex_path.read_text(encoding="utf-8")

    # Header columns: "Metric & \makecell[ct]{BBLF} & ..."
    m = re.search(r'^\s*Metric\s*&(?P<head>.+?)\\\\', text, flags=re.M)
    if not m:
        raise RuntimeError(f"Could not find header row 'Metric & ... \\\\' in {tex_path}")
    head = m.group("head")
    cols = [h.strip() for h in head.split("&")]
    columns = [_strip_macro(h) for h in cols]  # e.g. ['BBLF','WA','WAE','WAR']

    # Find each "Evaluated on: ..." block
    block_iter = list(re.finditer(
        r'\\multicolumn\{\d+\}\{l\}\{\\textit\{Evaluated on:\s*([^}]*)\}\}\s*\\\\',
        text
    ))

    data: Dict[str, Dict[str, List[float]]] = {}
    for i, mblk in enumerate(block_iter):
        friendly = mblk.group(1).strip()
        start = mblk.end()
        end = block_iter[i+1].start() if i+1 < len(block_iter) else len(text)
        chunk = text[start:end]

        # row lines inside this block
        row_iter = re.finditer(
            r'^\s*([A-Za-z0-9\- ()]+?)\s*&(?P<cells>.+?)\\\\',
            chunk,
            flags=re.M
        )
        for r in row_iter:
            raw_label = r.group(1).strip()
            # keep only known labels (answer + reasoning)
            if raw_label not in ROW_LABEL_TO_CANON and raw_label not in CANON_TO_SHORT.values():
                continue
            # normalize to canonical metric key
            canon_metric = ROW_LABEL_TO_CANON.get(raw_label, raw_label)
            cells = [c.strip() for c in r.group("cells").split("&")]
            cells = (cells + [""] * len(columns))[:len(columns)]
            vals = [_clean_cell_to_float(c) for c in cells]
            data.setdefault(friendly, {})[canon_metric] = vals

    return columns, data

# -------- computations --------
def best_over_train(vals: List[float]) -> float:
    s = pd.Series(vals, dtype="float64")
    if s.dropna().empty:
        return float("nan")
    return s.max(skipna=True)

def compute_delta_by_eval_best_over_train(
    sft: Tuple[List[str], Dict[str, Dict[str, List[float]]]],
    dpo: Tuple[List[str], Dict[str, Dict[str, List[float]]]],
    metrics_for_output: List[str],
    eval_order: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Return: delta[metric_short][eval_pretty] = best_over_train(DPO) - best_over_train(SFT)
    metrics_for_output uses SHORT labels for display; we map them to canonical for lookup.
    """
    _, sft_data = sft
    _, dpo_data = dpo
    delta: Dict[str, Dict[str, float]] = {}

    for m_short in metrics_for_output:
        m_canon = ROW_LABEL_TO_CANON.get(m_short, m_short)  # <-- FIX: map short -> canonical
        delta[m_short] = {}
        for e in eval_order:
            friendly = EVAL_FRIENDLY.get(e, e)
            sft_vals = sft_data.get(friendly, {}).get(m_canon, [])
            dpo_vals = dpo_data.get(friendly, {}).get(m_canon, [])
            b_sft = best_over_train(sft_vals)
            b_dpo = best_over_train(dpo_vals)
            delta[m_short][e] = b_dpo - b_sft
    return delta

# -------- latex helpers --------
def fmt_delta(v: float, precision: int = 3) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:+.{precision}f}"

def highlight_row(values: List[Optional[float]], cells: List[str]) -> List[str]:
    """Color top-1 (high1) and top-2 (high2) per row. No bold."""
    ser = pd.Series(values, dtype="float64")
    valid = ser.dropna()
    if valid.empty:
        return cells
    order = valid.sort_values(ascending=False).index.tolist()
    top1 = order[0]
    top2 = order[1] if len(order) >= 2 else None
    out = []
    for i, c in enumerate(cells):
        if c == "—":
            out.append(c)
        elif i == top1:
            out.append(f"\\cellcolor{{high1}}{c}")
        elif top2 is not None and i == top2:
            out.append(f"\\cellcolor{{high2}}{c}")
        else:
            out.append(c)
    return out

def latex_delta_table(
    delta: Dict[str, Dict[str, float]],
    caption: str,
    label: str,
    precision: int = 3,
    table_spec: str = "!t",
    font_size_cmd: str = "\\footnotesize",
    tabcolsep_pt: int = 3,
    arraystretch: str = "1",
    header_band_title: str = "Model Trained On",
) -> str:
    # Columns in eval order but labeled by ABBREV (BBLF/WA/WAE/WAR)
    cols_eval = EVAL_ORDER
    col_titles = [ABBREV[c] for c in cols_eval]

    lines = []
    lines.append(f"\\begin{{table}}[{table_spec}]")
    lines.append("\\centering")
    lines.append(font_size_cmd)
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}}")
    lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l" + "r"*len(cols_eval) + "}")
    lines.append("\\toprule")
    lines.append(f"& \\multicolumn{{{len(cols_eval)}}}{{c}}{{\\textbf{{{header_band_title}}}}} \\\\")
    lines.append(f"\\cmidrule(l){{2-{len(cols_eval)+1}}}")
    lines.append("Metric & " + " & ".join([f"\\makecell[ct]{{{t}}}" for t in col_titles]) + " \\\\")
    lines.append("\\midrule")

    # Row order: answer metrics then reasoning
    row_order = METRICS_ANS + METRICS_RSN_SHORT
    for m_short in row_order:
        vals = [delta.get(m_short, {}).get(ev, float("nan")) for ev in cols_eval]
        cells = [fmt_delta(v, precision) for v in vals]
        cells = highlight_row(vals, cells)
        lines.append(f"{m_short} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Build DPO vs. SFT delta table with highlights from saved .tex cross-eval tables.")
    ap.add_argument("--sft_tex", type=str, required=True, help="Path to SFT cross-eval .tex (your exact format).")
    ap.add_argument("--dpo_tex", type=str, required=True, help="Path to DPO cross-eval .tex (your exact format).")
    ap.add_argument("--out_tex", type=str, required=True, help="Output .tex file for the delta table.")
    ap.add_argument("--precision", type=int, default=3)
    ap.add_argument("--caption", type=str, default=(
        r"DPO vs.\ SFT deltas by \emph{evaluation} pooler (best over training poolers) in \textbf{ReasonVQA eval set}. "
        r"$\Delta$ = DPO $-$ SFT; positive means DPO helps."
    ))
    ap.add_argument("--label", type=str, default="tab:dpo_delta_by_eval")
    ap.add_argument("--header_band_title", type=str, default="Model Trained On",
                    help="Centered band title above columns (e.g., 'Evaluation Pooler').")
    args = ap.parse_args()

    sft_cols, sft_data = parse_cross_eval_tex(Path(args.sft_tex))
    dpo_cols, dpo_data = parse_cross_eval_tex(Path(args.dpo_tex))

    # Compute Δ by evaluation (best over training)
    delta = compute_delta_by_eval_best_over_train(
        (sft_cols, sft_data),
        (dpo_cols, dpo_data),
        metrics_for_output=METRICS_ANS + METRICS_RSN_SHORT,  # use short labels; we map to canonical inside
        eval_order=EVAL_ORDER,
    )

    tex = latex_delta_table(
        delta=delta,
        caption=args.caption,
        label=args.label,
        precision=args.precision,
        table_spec="!t",
        font_size_cmd="\\footnotesize",
        tabcolsep_pt=3,
        arraystretch="1",
        header_band_title=args.header_band_title,
    )
    Path(args.out_tex).write_text(tex, encoding="utf-8")

    # Echo
    print("="*80)
    print(Path(args.out_tex).read_text(encoding="utf-8"))
    print("="*80)
    print("LaTeX deps: \\usepackage{booktabs} \\usepackage{makecell} \\usepackage[table]{xcolor}")
    print("Define colors: \\definecolor{high1}{RGB}{255,235,190}  \\definecolor{high2}{RGB}{220,240,255}")

if __name__ == "__main__":
    main()
