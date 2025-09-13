#!/usr/bin/env python3
"""
Generate two LaTeX tables from a nested runs directory.

Key features:
- Auto-detect top-level roots by prefix before first '-' (e.g., 'base-*', 'sft-*').
- Headers are straight (no rotate) and wrapped via \makecell.
- Drops EM everywhere.
- NEW: SFT table supports a tall (transposed) layout for readability:
    rows = (Eval method × Metric), columns = SFT-on methods.
    Bold = row-wise best; diagonal cells shaded.
- Options: --sft_layout {tall,wide}, --split_reasoning, --landscape_sft, --font_size.

Preamble:
  \\usepackage{booktabs}
  \\usepackage{graphicx}
  \\usepackage[table]{xcolor}
  \\usepackage{makecell}
  \\usepackage{multirow}
  % (optional for landscape)
  % \\usepackage{rotating}   % for sidewaystable*
  % or \\usepackage{pdflscape}
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd


# ---------- Pretty helpers ----------

def prettify_method(name: str) -> str:
    if not name:
        return "N/A"
    name = name.strip().replace("_", " ")
    name = name.replace("blend blur with last frame", "Blend+Blur+Last")
    name = name.replace("weighted average exponential", "Weighted Avg (Exp)")
    name = name.replace("weighted average ramp", "Weighted Avg (Ramp)")
    name = name.replace("weighted average whatever5", "Weighted Avg (Var5)")
    name = name.replace("weighted average", "Weighted Avg")
    name = " ".join(name.split())
    return " ".join([w.capitalize() if w.isalpha() else w for w in name.split(" ")])

_METRIC_ALIASES = {
    "em": "EM",
    "exact_match": "EM",
    "f1": "F1",
    "bleu": "BLEU-1",
    "bleu1": "BLEU-1", "bleu_1": "BLEU-1", "bleu-1": "BLEU-1",
    "bleu4": "BLEU-4 (BP)", "bleu_4": "BLEU-4 (BP)", "bleu_4_bp": "BLEU-4 (BP)",
    "bleu4 bp": "BLEU-4 (BP)", "bleu-4 (bp)": "BLEU-4 (BP)", "bleu4bp": "BLEU-4 (BP)",
    "rouge": "ROUGE-L", "rouge_l": "ROUGE-L", "rouge-l": "ROUGE-L", "rouge l": "ROUGE-L",
    "embedcosine": "Embed Cosine", "embed cosine": "Embed Cosine",
    "embedding_cosine": "Embed Cosine", "embedding cos": "Embed Cosine",
    # Reasoning variants
    "rouge l reasoning": "ROUGE-L (Reasoning)",
    "rouge-l (reasoning)": "ROUGE-L (Reasoning)",
    "rouge_l_reasoning": "ROUGE-L (Reasoning)",
    "embed cosine reasoning": "Embed Cosine (Reasoning)",
    "embedding_cosine_reasoning": "Embed Cosine (Reasoning)",
    "embedcosine reasoning": "Embed Cosine (Reasoning)",
}

def prettify_metric(key: str) -> str:
    if not key:
        return "?"
    k = key.lower().replace("_", " ").strip()
    if k in _METRIC_ALIASES:
        return _METRIC_ALIASES[k]
    # Basic normalization fallback
    tokens = k.split()
    tokens = ["BLEU-1" if t=="bleu-1" else "BLEU-4 (BP)" if t in {"bleu4","bleu4bp","bleu4 bp"} else
              "ROUGE-L" if t in {"rouge","rouge-l","rouge l"} else
              "F1" if t=="f1" else t.capitalize() for t in tokens]
    return " ".join(tokens)

def latex_escape(s: str) -> str:
    return (s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_"))

def fmt_val(x, precision=3) -> Optional[str]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        return f"{float(x):.{precision}f}"
    except Exception:
        return None

def wrap_header(title: str, max_chars: int = 16) -> str:
    """
    Wrap a header string to multiple lines using \\makecell if long.
    """
    words = title.split()
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    out_lines = []
    for line in lines:
        if len(line) <= max_chars:
            out_lines.append(line)
        else:
            # hard wrap
            s = line
            while len(s) > max_chars:
                out_lines.append(s[:max_chars]); s = s[max_chars:]
            if s: out_lines.append(s)
    out_lines = [latex_escape(l) for l in out_lines]
    return "\\makecell[ct]{" + " \\\\ ".join(out_lines) + "}"


# ---------- Discovery helpers ----------

def detect_roots(runs_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find top-level dirs whose *prefix before first '-'* is 'base' or 'sft'.
    Falls back to literal 'base'/'sft'.
    """
    base_root = None
    sft_root = None
    for p in runs_dir.iterdir():
        if not p.is_dir(): continue
        name = p.name
        prefix = name.split("-", 1)[0] if "-" in name else name
        if prefix == "base" and base_root is None: base_root = p
        elif prefix == "sft" and sft_root is None: sft_root = p
    if base_root is None and (runs_dir / "base").is_dir(): base_root = runs_dir / "base"
    if sft_root is None and (runs_dir / "sft").is_dir(): sft_root = runs_dir / "sft"
    return base_root, sft_root


# ---------- IO & parsing ----------

def read_summary(fp: Path) -> Optional[Tuple[str, Dict[str, float]]]:
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return None
    method = data.get("method", "") or ""
    metrics = data.get("metrics", {}) or {}
    return method, metrics

def collect_base_results(base_root: Path) -> pd.DataFrame:
    rows = []
    for method_dir in sorted([p for p in base_root.glob("*") if p.is_dir()]):
        candidates = list(method_dir.glob("*.summary.json")) + list(method_dir.glob("*summary.json"))
        if not candidates:
            hidden = method_dir / ".summary.json"
            if hidden.exists(): candidates = [hidden]
        if not candidates: continue
        fp = max(candidates, key=lambda p: p.stat().st_mtime)
        parsed = read_summary(fp)
        if not parsed: continue
        method_raw, metrics = parsed
        method_name = prettify_method(method_raw or method_dir.name)
        row = {"Method": method_name}
        for k, v in metrics.items():
            nice = prettify_metric(k)
            if nice == "EM": continue  # drop EM
            row[nice] = v
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Method")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "EM" in df.columns:
        df = df.drop(columns=["EM"])
    return df

def collect_sft_matrix(sft_root: Path) -> pd.DataFrame:
    """
    Returns DataFrame with rows = train_method, columns = MultiIndex (eval_method, metric)
    """
    store: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_metrics = set()
    all_eval_methods = set()

    for train_dir in sorted([p for p in sft_root.glob("*") if p.is_dir()]):
        train_method = prettify_method(train_dir.name)
        for eval_dir in sorted([p for p in train_dir.glob("*") if p.is_dir()]):
            eval_method = prettify_method(eval_dir.name)
            candidates = list(eval_dir.glob("*.summary.json")) + list(eval_dir.glob("*summary.json"))
            if not candidates:
                hidden = eval_dir / ".summary.json"
                if hidden.exists(): candidates = [hidden]
            if not candidates: continue
            fp = max(candidates, key=lambda p: p.stat().st_mtime)
            parsed = read_summary(fp)
            if not parsed: continue
            _mname, metrics = parsed
            store.setdefault(train_method, {}).setdefault(eval_method, {})
            for k, v in metrics.items():
                nice = prettify_metric(k)
                if nice == "EM": continue
                store[train_method][eval_method][nice] = v
                all_metrics.add(nice)
            all_eval_methods.add(eval_method)

    if not store:
        return pd.DataFrame()

    evals = sorted(all_eval_methods)
    metrics_sorted = sorted([m for m in all_metrics if m != "EM"], key=lambda x: x.lower())

    rows = {}
    for train_method, m_by_eval in store.items():
        row = {}
        for e in evals:
            mdict = m_by_eval.get(e, {})
            for met in metrics_sorted:
                row[(e, met)] = pd.to_numeric(mdict.get(met, float("nan")), errors="coerce")
        rows[train_method] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df = df.reindex(columns=pd.MultiIndex.from_product([evals, metrics_sorted]))
    return df


# ---------- LaTeX builders ----------

def latex_table_base(df: pd.DataFrame, caption: str, label: str,
                     precision: int = 3, font_size: str = "small", resize: bool = True) -> str:
    if df.empty:
        return "% No base results found."
    col_max = df.max(skipna=True)
    cols = list(df.columns)
    colspec = "l" + "c" * len(cols)
    headers = " & ".join([wrap_header(c) for c in cols])

    lines = []
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering")
    lines.append(f"\\{font_size}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")
    if resize:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(f"\\textbf{{Method}} & {headers} \\\\")
    lines.append("\\midrule")

    for method, row in df.iterrows():
        row_out = [f"\\textbf{{{latex_escape(method)}}}"]
        for c in cols:
            val = row[c]; s = fmt_val(val, precision)
            if s is None:
                row_out.append("—")
            else:
                row_out.append(f"\\textbf{{{s}}}" if pd.notnull(val) and (val >= col_max[c]-1e-12) else s)
        lines.append(" & ".join(row_out) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if resize:
        lines.append("}%")
    lines.append("\\vspace{-0.5em}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def _filter_metrics(df: pd.DataFrame, reasoning: Optional[bool]) -> pd.DataFrame:
    """reasoning=None -> all, True -> only '(Reasoning)', False -> exclude '(Reasoning)'."""
    if df.empty: return df
    cols = []
    for (e, m) in df.columns:
        is_r = "(Reasoning)" in m
        if reasoning is None or (reasoning and is_r) or ((reasoning is False) and (not is_r)):
            cols.append((e, m))
    return df.loc[:, cols]


def latex_table_sft_wide(df: pd.DataFrame, caption: str, label: str,
                         precision: int = 3, font_size: str = "small",
                         shade_diag: bool = True, resize: bool = True,
                         landscape: bool = False) -> str:
    """Original wide layout, but with straight wrapped headers and font controls."""
    if df.empty:
        return "% No SFT cross-eval results found."
    col_max = df.max(skipna=True)
    eval_methods = list(dict.fromkeys([c[0] for c in df.columns]))
    metrics = list(dict.fromkeys([c[1] for c in df.columns]))
    colspec = "l" + "".join(["c" * len(metrics) for _ in eval_methods])

    group_headers = [f"\\multicolumn{{{len(metrics)}}}{{c}}{{\\textbf{{{wrap_header(e)}}}}}" for e in eval_methods]
    group_header_line = " & " + " & ".join(group_headers) + " \\\\"
    metric_headers = []
    for _ in eval_methods:
        metric_headers.extend([wrap_header(m) for m in metrics])
    metric_header_line = " & " + " & ".join(metric_headers) + " \\\\"

    cmidrules = []
    start = 2
    for _ in eval_methods:
        end = start + len(metrics) - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    cmidrules_line = " ".join(cmidrules)

    env = "sidewaystable*" if landscape else "table*"
    lines = []
    lines.append(f"\\begin{{{env}}}[ht]")
    lines.append("\\centering")
    lines.append(f"\\{font_size}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")
    if resize:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{\\textbf{SFT on}} " + group_header_line)
    lines.append(cmidrules_line)
    lines.append(metric_header_line)
    lines.append("\\midrule")

    for train_method, row in df.iterrows():
        row_cells = [f"\\textbf{{{latex_escape(train_method)}}}"]
        for e in eval_methods:
            for m in metrics:
                val = row[(e, m)]
                s = fmt_val(val, precision)
                if s is None:
                    cell = "—"
                else:
                    s_b = f"\\textbf{{{s}}}" if pd.notnull(val) and (val >= col_max[(e, m)]-1e-12) else s
                    cell = f"\\cellcolor{{gray!12}} {s_b}" if (shade_diag and train_method == e) else s_b
                row_cells.append(cell)
        lines.append(" & ".join(row_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if resize:
        lines.append("}%")
    lines.append("\\vspace{-0.5em}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines)


def latex_table_sft_tall(df: pd.DataFrame, caption: str, label: str,
                         precision: int = 3, font_size: str = "small",
                         shade_diag: bool = True, resize: bool = False,
                         landscape: bool = False) -> str:
    """
    Tall layout (transposed):
      Columns = SFT-on methods (train methods)
      Rows    = (Eval method × Metric)
    Bold = row-wise best across train methods.
    """
    if df.empty:
        return "% No SFT cross-eval results found."

    train_methods: List[str] = list(df.index)
    eval_methods = list(dict.fromkeys([c[0] for c in df.columns]))
    metrics = list(dict.fromkeys([c[1] for c in df.columns]))

    # Build (eval, metric) row order
    row_pairs: List[Tuple[str, str]] = []
    for e in eval_methods:
        for m in metrics:
            if (e, m) in df.columns:
                row_pairs.append((e, m))

    colspec = "l" + "c" * len(train_methods)
    env = "sidewaystable*" if landscape else "table*"

    lines = []
    lines.append(f"\\begin{{{env}}}[ht]")
    lines.append("\\centering")
    lines.append(f"\\{font_size}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")
    if resize:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Header: blank + train methods
    th = " & ".join([wrap_header(tm) for tm in train_methods])
    lines.append(f" & {th} \\\\")
    lines.append("\\midrule")

    # Rows grouped by eval method
    for e in eval_methods:
        lines.append(f"\\multicolumn{{{len(train_methods)+1}}}{{l}}{{\\textbf{{{wrap_header(e)}}}}} \\\\")
        for m in metrics:
            if (e, m) not in df.columns:
                continue
            # Compute row-wise max across train methods
            row_vals = [df.loc[tm, (e, m)] for tm in train_methods]
            row_max = pd.Series(row_vals).max(skipna=True)
            row_cells = [wrap_header(m)]
            for tm, val in zip(train_methods, row_vals):
                s = fmt_val(val, precision)
                if s is None:
                    cell = "—"
                else:
                    s_b = f"\\textbf{{{s}}}" if pd.notnull(val) and (val >= row_max-1e-12) else s
                    cell = f"\\cellcolor{{gray!12}} {s_b}" if (shade_diag and tm == e) else s_b
                row_cells.append(cell)
            lines.append(" & ".join(row_cells) + " \\\\")
        lines.append("\\addlinespace[0.25em]")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if resize:
        lines.append("}%")
    lines.append("\\vspace{-0.5em}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Generate two LaTeX tables (Base & SFT cross-eval) from nested MovieVQA runs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--runs_dir", type=str, required=True,
                    help="Path to 'runs' containing top-level dirs like 'base-*' and 'sft-*' (or plain 'base'/'sft').")
    ap.add_argument("--precision", type=int, default=3, help="Decimal places for numbers.")
    ap.add_argument("--font_size", type=str, default="small",
                    help="LaTeX font size macro for tables (e.g., small, footnotesize, scriptsize).")
    ap.add_argument("--no_resize", action="store_true", help="Do not wrap tables in \\resizebox{\\textwidth}{!}.")
    ap.add_argument("--caption_base", type=str,
                    default=("Base results across pooling methods on MovieVQA. "
                             "We report F1, BLEU, ROUGE-L, and Embed Cosine. Best per column is bold."),
                    help="Caption for the base table.")
    ap.add_argument("--label_base", type=str, default="tab:base_results",
                    help="LaTeX label for the base table.")
    ap.add_argument("--caption_sft", type=str,
                    default=("Cross-evaluation of fine-tuned models. Rows are grouped by the evaluation method; "
                             "each row is a metric, columns are SFT-on methods. Diagonals (train=evaluation) are "
                             "shaded; best in each row is bold."),
                    help="Caption for the SFT table.")
    ap.add_argument("--label_sft", type=str, default="tab:sft_cross_eval",
                    help="LaTeX label for the SFT table.")
    ap.add_argument("--out_base", type=str, default="", help="Optional path to save the base table .tex.")
    ap.add_argument("--out_sft", type=str, default="", help="Optional path to save the SFT table .tex.")
    ap.add_argument("--sft_layout", type=str, choices=["tall", "wide"], default="tall",
                    help="Layout for the SFT table.")
    ap.add_argument("--split_reasoning", action="store_true",
                    help="Emit two SFT tables: one for answer metrics, one for reasoning metrics.")
    ap.add_argument("--landscape_sft", action="store_true",
                    help="Render SFT table as sidewaystable* (requires \\usepackage{rotating}).")

    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    resize = not args.no_resize

    base_root, sft_root = detect_roots(runs_dir)
    if base_root is None:
        print(f"Error: Could not find a 'base' root in {runs_dir} (expected 'base-*' or 'base').")
    if sft_root is None:
        print(f"Error: Could not find an 'sft' root in {runs_dir} (expected 'sft-*' or 'sft').")

    # Base
    base_tex = "% No base results found."
    if base_root is not None:
        df_base = collect_base_results(base_root)
        base_tex = latex_table_base(df_base, args.caption_base, args.label_base,
                                    precision=args.precision, font_size=args.font_size, resize=resize)

    # SFT
    sft_tex = "% No SFT cross-eval results found."
    if sft_root is not None:
        df_sft_full = collect_sft_matrix(sft_root)

        def make_sft(df_part: pd.DataFrame, lbl_suffix: str = "") -> str:
            if args.sft_layout == "wide":
                return latex_table_sft_wide(df_part, args.caption_sft, args.label_sft + lbl_suffix,
                                            precision=args.precision, font_size=args.font_size,
                                            shade_diag=True, resize=resize, landscape=args.landscape_sft)
            else:
                return latex_table_sft_tall(df_part, args.caption_sft, args.label_sft + lbl_suffix,
                                            precision=args.precision, font_size=args.font_size,
                                            shade_diag=True, resize=False, landscape=args.landscape_sft)

        if args.split_reasoning:
            df_ans = _filter_metrics(df_sft_full, reasoning=False)
            df_rsn = _filter_metrics(df_sft_full, reasoning=True)
            sft_tex = make_sft(df_ans, lbl_suffix="_ans") + "\n\n" + make_sft(df_rsn, lbl_suffix="_rsn")
        else:
            sft_tex = make_sft(df_sft_full)

    print("\n" + "=" * 90)
    print("TABLE 1: BASE RESULTS (paste into LaTeX)")
    print("=" * 90 + "\n")
    print(base_tex)

    print("\n" + "=" * 90)
    print("TABLE 2: SFT CROSS-EVALUATION RESULTS (paste into LaTeX)")
    print("=" * 90 + "\n")
    print(sft_tex)

    if args.out_base:
        Path(args.out_base).write_text(base_tex)
    if args.out_sft:
        Path(args.out_sft).write_text(sft_tex)

    print("\n" + "=" * 90)
    print("LaTeX packages to include:")
    print("\\usepackage{booktabs}")
    print("\\usepackage{graphicx}")
    print("\\usepackage[table]{xcolor}")
    print("\\usepackage{makecell}")
    print("\\usepackage{multirow}")
    if args.landscape_sft:
        print("\\usepackage{rotating}  % for sidewaystable*  (or pdflscape)")
    print("=" * 90)


if __name__ == "__main__":
    main()
