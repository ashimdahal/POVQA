#!/usr/bin/env python3
import os, json, glob, argparse, re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ----------------------------- config / constants -----------------------------

KEYFRAME_TOKEN = "keyframe"  # matches "KeyFrame only", etc. (case-insensitive)

ORDER_METHODS = [
    "Blend Blur With Last Frame",
    "Weighted Average",
    "Weighted Average (Exp)",
    "Weighted Average (Ramp)",
]
ABBR = {
    "Blend Blur With Last Frame": "BBLF",
    "Weighted Average": "WA",
    "Weighted Average (Exp)": "WAE",
    "Weighted Average (Ramp)": "WAR",
}

# Map file metric keys -> display labels (drop EM)
METRIC_KEY_MAP = {
    "EM": None,
    "F1": "F1",
    "BLEU1": "BLEU-1", "BLEU_1": "BLEU-1", "BLEU-1": "BLEU-1",
    "BLEU4_BP": "BLEU-4 (BP)", "BLEU_4_BP": "BLEU-4 (BP)", "BLEU-4_BP": "BLEU-4 (BP)",
    "ROUGE_L": "ROUGE-L", "ROUGE-L": "ROUGE-L",
    "EmbedCos": "Embed Cosine", "EMBEDCOS": "Embed Cosine",
    "ROUGE_L_Reasoning": "ROUGE-L-R",
    "EmbedCos_Reasoning": "Embed Cos-R",
}
METRIC_ROW_ORDER = ["F1", "BLEU-1", "BLEU-4 (BP)", "ROUGE-L", "Embed Cosine", "ROUGE-L-R", "Embed Cos-R"]

# ----------------------------- helpers ---------------------------------------

def titleize(s: str) -> str:
    return s.replace("_", " ").title()

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_files(base_dir: str) -> List[str]:
    """Only *.summary.json under runs/*(sft|dpo)*/<method>/<eval>/."""
    pats = []
    for m in ("sft", "dpo"):
        pats.append(os.path.join(base_dir, "runs", f"*{m}*", "*", "*", "*.summary.json"))
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    # keep only roots that indeed contain sft/dpo
    keep = []
    for fp in sorted(set(files)):
        parts = fp.split(os.sep)
        if len(parts) < 4: 
            continue
        root = parts[-4].lower()
        if "sft" in root or "dpo" in root:
            keep.append(fp)
    return keep

def parse_path(fp: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = fp.split(os.sep)
    if len(parts) < 4: 
        return None, None, None
    root = parts[-4].lower()
    model = "dpo" if "dpo" in root else ("sft" if "sft" in root else None)
    method = titleize(parts[-3])
    evaldir = titleize(parts[-2])
    return model, method, evaldir

def load_summary(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def is_keyframe(eval_dir: str, json_method: str, match_mode: str = "either") -> bool:
    """match_mode: 'either' (dir OR json) or 'both'."""
    dir_hit = KEYFRAME_TOKEN in slug(eval_dir)
    json_hit = KEYFRAME_TOKEN in slug(json_method)
    return (dir_hit or json_hit) if match_mode == "either" else (dir_hit and json_hit)

# ----------------------------- aggregation w/ debug ---------------------------

def collect_keyframe_max_by_model(files: List[str], match_mode: str, debug: bool):
    """
    Returns:
      store_vals[method][model][metric] = best_val
      store_srcs[method][model][metric] = filepath of best_val
    KeyFrame detection via is_keyframe(eval_dir, json['method'], match_mode).
    """
    store_vals: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    store_srcs: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))

    for fp in files:
        model, method, evaldir = parse_path(fp)
        if not model or not method or not evaldir:
            continue
        data = load_summary(fp)
        if not isinstance(data, dict):
            continue
        json_method = str(data.get("method", ""))
        if not is_keyframe(evaldir, json_method, match_mode):
            continue

        metrics = (data.get("metrics") or {})
        if not isinstance(metrics, dict):
            continue
        for raw_key, val in metrics.items():
            disp = METRIC_KEY_MAP.get(raw_key)
            if disp is None:
                continue # drop EM/unknown
            try:
                v = float(val)
            except Exception:
                continue
            prev = store_vals[method][model].get(disp)
            if (prev is None) or (v > prev):
                store_vals[method][model][disp] = v
                store_srcs[method][model][disp] = fp

    if debug:
        print("\n=== DEBUG: Best per (method, model) for KeyFrame-only ===")
        for method in sorted(store_vals.keys()):
            for model in ("sft", "dpo"):
                if model not in store_vals[method]:
                    continue
                print(f"\n[{method}] [{model.upper()}]")
                for m in METRIC_ROW_ORDER:
                    if m in store_vals[method][model]:
                        v = store_vals[method][model][m]
                        src = store_srcs[method][model][m]
                        print(f"  {m:16s} = {v:.6f}  <- {src}")
    return store_vals, store_srcs

def collapse_max_over_models(store_vals, store_srcs, debug: bool):
    """
    best_vals[method][metric] = max(SFT,DPO), best_srcs[method][metric] = file that won
    """
    best_vals: Dict[str, Dict[str, float]] = defaultdict(dict)
    best_srcs: Dict[str, Dict[str, str]] = defaultdict(dict)

    for method, by_model in store_vals.items():
        for metric_label in METRIC_ROW_ORDER:
            sft_val = by_model.get("sft", {}).get(metric_label)
            dpo_val = by_model.get("dpo", {}).get(metric_label)
            sft_src = store_srcs.get(method, {}).get("sft", {}).get(metric_label)
            dpo_src = store_srcs.get(method, {}).get("dpo", {}).get(metric_label)

            if sft_val is None and dpo_val is None:
                continue
            if dpo_val is None or (sft_val is not None and sft_val > dpo_val):
                best_vals[method][metric_label] = sft_val
                best_srcs[method][metric_label] = sft_src
                winner = "SFT"
            else:
                best_vals[method][metric_label] = dpo_val
                best_srcs[method][metric_label] = dpo_src
                winner = "DPO"

            if debug:
                print(f"[choose] {method:30s} {metric_label:16s} -> {winner} "
                      f"{best_vals[method][metric_label]:.6f}  {best_srcs[method][metric_label]}")
    return best_vals, best_srcs

# ----------------------------- delta ------------------------------------------

def compute_delta_per_metric(store_vals, best_vals, methods_used: List[str]):
    """
    Δ(metric) = max_over_methods( DPO-only ) − max_over_methods( ablation best )
    """
    delta = {}
    for met in METRIC_ROW_ORDER:
        dpo_best = None
        abl_best = None
        # max over methods
        for m in methods_used:
            dv = store_vals.get(m, {}).get("dpo", {}).get(met)
            if isinstance(dv, (int, float)):
                dpo_best = dv if dpo_best is None else max(dpo_best, dv)
            av = best_vals.get(m, {}).get(met)
            if isinstance(av, (int, float)):
                abl_best = av if abl_best is None else max(abl_best, av)
        delta[met] = None if (dpo_best is None or abl_best is None) else (dpo_best - abl_best)
    return delta

# ----------------------------- highlight --------------------------------------

def top2_indices(values: List[Optional[float]]):
    present = {i: v for i, v in enumerate(values) if isinstance(v, (int, float))}
    if not present:
        return None, None
    order = sorted(present.keys(), key=lambda i: present[i], reverse=True)
    t1 = order[0]
    t2 = order[1] if len(order) >= 2 else None
    return t1, t2

# ----------------------------- LaTeX -----------------------------------------

def make_transposed_table_with_delta(
    best_vals, store_vals, methods_present,
    caption, label="tab:keyframe_ablation_transposed_delta",
    precision=3, table_spec="H", tabcolsep_pt=3, arraystretch="1"
):
    headers = [ABBR.get(m, m) for m in methods_present]
    metrics_present = [met for met in METRIC_ROW_ORDER if any(met in best_vals.get(method, {}) for method in methods_present)]
    delta_map = compute_delta_per_metric(store_vals, best_vals, methods_present)

    lines = []
    lines.append(f"\\begin{{table}}[{table_spec}]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}}")
    lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l" + "c"*len(headers) + "r}")
    lines.append("\\toprule")
    lines.append("Metric & " + " & ".join([f"\\makecell[ct]{{{h}}}" for h in headers]) + " & $\\Delta$ \\\\")
    lines.append("\\midrule")

    for met in metrics_present:
        vals = [best_vals.get(m, {}).get(met) for m in methods_present]
        t1, t2 = top2_indices(vals)
        method_cells = []
        for j, v in enumerate(vals):
            if isinstance(v, (int, float)):
                s = f"{v:.{precision}f}"
                if j == t1:
                    s = f"\\cellcolor{{high1}} {s}"
                elif j == t2:
                    s = f"\\cellcolor{{high2}} {s}"
            else:
                s = "—"
            method_cells.append(s)
        d = delta_map.get(met, None)
        delta_str = "—" if d is None else f"{d:+.{precision}f}"
        lines.append(f"{met} & " + " & ".join(method_cells) + f" & {delta_str} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="KeyFrame-only ablation (transposed) with high1/high2 and Δ; robust KeyFrame matching.")
    ap.add_argument("--base_dir", type=str, default=".", help="Root containing runs/")
    ap.add_argument("--out_tex", type=str, required=True, help="Write .tex snippet here")
    ap.add_argument("--precision", type=int, default=3)
    ap.add_argument("--table_spec", type=str, default="H")
    ap.add_argument("--match_mode", type=str, choices=["either","both"], default="either",
                    help="Accept KeyFrame-only if dir OR json say keyframe (either), or require both.")
    ap.add_argument("--debug", action="store_true", help="Print audit with filepaths of chosen maxima.")
    ap.add_argument("--caption", type=str, default=(
        "KeyFrame-only ablation (max over SFT/DPO per method). "
        "\\colorbox{high1}{highest}, \\colorbox{high2}{second-highest} per row. "
        "$\\Delta$ = best DPO (over methods) $-$ best KeyFrame ablation (over methods)."
    ))
    args = ap.parse_args()

    files = find_files(args.base_dir)
    if not files:
        raise SystemExit("No *.summary.json under runs/*(sft|dpo)*/…")

    if args.debug:
        print(f"Found {len(files)} candidate summaries.")

    store_vals, store_srcs = collect_keyframe_max_by_model(files, match_mode=args.match_mode, debug=args.debug)

    # Order methods actually present
    methods_present = [m for m in ORDER_METHODS if m in store_vals] + [m for m in sorted(store_vals.keys()) if m not in ORDER_METHODS]
    if not methods_present:
        raise SystemExit("No KeyFrame-only summaries matched (try --match_mode either and/or --debug).")

    if args.debug:
        print("\n=== METHODS PRESENT ===")
        for m in methods_present:
            print(" -", m)

    best_vals, best_srcs = collapse_max_over_models(store_vals, store_srcs, debug=args.debug)

    tex = make_transposed_table_with_delta(
        best_vals=best_vals,
        store_vals=store_vals,
        methods_present=methods_present,
        caption=args.caption,
        label="tab:keyframe_ablation_transposed_delta",
        precision=args.precision,
        table_spec=args.table_spec,
        tabcolsep_pt=3,
        arraystretch="1",
    )

    os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print("="*80)
    print(f"Wrote: {args.out_tex}")
    print("="*80)
    print(tex)
    print("="*80)
    print("Requires: \\usepackage{booktabs} \\usepackage[table]{xcolor}")
    print("Define colors: \\definecolor{high1}{RGB}{255,235,190}  \\definecolor{high2}{RGB}{220,240,255}")

if __name__ == "__main__":
    main()
