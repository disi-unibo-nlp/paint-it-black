#!/usr/bin/env python3
"""
Generate three appendix LaTeX tables (one file, all models).

  Table 1 – Spatial F1 @ tau     E2E macro F1 at tau in {0.25, 0.5, 0.75}
  Table 2 – Text quality         Char-F1 / EMR / Approximate Span F1
  Table 3 – Micro Text F1        span-exact micro F1/P/R

Usage
-----
python scripts/latex_appendix_tables.py \\
    --model "Gemma-4-31B-it" base_dir medium_dir hard_dir \\
    --model "Qwen3.5-9B"     base_dir medium_dir hard_dir \\
    --output appendix_tables.tex

Models are emitted in the order given; sort them before calling.
"""

import argparse
import json
import sys
from pathlib import Path


# ── IoU / pass-rate helpers ────────────────────────────────────────────────────

def _enclosing_box(bboxes):
    bboxes = [[float(c) for c in b] for b in bboxes]
    return [
        min(b[0] for b in bboxes), min(b[1] for b in bboxes),
        max(b[2] for b in bboxes), max(b[3] for b in bboxes),
    ]


def _iou(pred_bboxes, gt_bboxes):
    if not pred_bboxes and not gt_bboxes:
        return 1.0
    if not pred_bboxes or not gt_bboxes:
        return 0.0
    pb = _enclosing_box(pred_bboxes)
    gb = _enclosing_box(gt_bboxes)
    iy1 = max(pb[0], gb[0]); ix1 = max(pb[1], gb[1])
    iy2 = min(pb[2], gb[2]); ix2 = min(pb[3], gb[3])
    inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
    union = (pb[2]-pb[0])*(pb[3]-pb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter
    return inter / union if union > 0 else 0.0


def _compute_pass_rate(per_sample, gt_by_idx, iou_thr=0.5):
    if not gt_by_idx:
        return None
    total_gt = joint_tp = 0
    for s in per_sample:
        if not s.get("parse_success", False):
            continue
        gts   = gt_by_idx.get(s["idx"], [])
        preds = s.get("predictions", [])
        total_gt += len(gts)
        candidates = []
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                if p.get("label") != g.get("label"):
                    continue
                exact = p.get("text", "").strip().lower() == g.get("text", "").strip().lower()
                iou   = _iou(p.get("bbox_2d", []), list(g.get("bboxes", [])))
                if exact and iou > iou_thr:
                    candidates.append((iou, pi, gi))
        candidates.sort(reverse=True)
        used_p, used_g = set(), set()
        for _, pi, gi in candidates:
            if pi not in used_p and gi not in used_g:
                used_p.add(pi); used_g.add(gi); joint_tp += 1
    return joint_tp / total_gt if total_gt > 0 else None


_gt_cache: dict = {}


def _load_gt(args: dict) -> dict:
    key = (args.get("input_dataset"), args.get("input_split"), args.get("from_hub", False))
    if key in _gt_cache:
        return _gt_cache[key]
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        print("[warn] 'datasets' not installed — PR will be '--'", file=sys.stderr)
        return {}
    try:
        if args.get("from_hub"):
            ds = load_dataset(args["input_dataset"], split=args["input_split"])
        else:
            ds = load_from_disk(str(Path(args["input_dataset"]) / args["input_split"]))
        rows = list(ds)
        result = {i: rows[i].get("annotations", []) for i in range(len(rows))}
        _gt_cache[key] = result
        return result
    except Exception as exc:
        print(f"[warn] Could not load dataset — PR will be '--': {exc}", file=sys.stderr)
        return {}


# ── Metric loading ─────────────────────────────────────────────────────────────

def load_metrics(results_dir: str) -> dict:
    data  = json.loads((Path(results_dir) / "results.json").read_text())
    m     = data["metrics"]
    te    = m["text_extraction"]
    bl    = m["bbox_localization"]
    e2e   = bl["end_to_end"]
    summ  = m["summary"]

    if "pass_rate" in summ:
        pr = summ["pass_rate"]
    else:
        pr = _compute_pass_rate(data["per_sample"], _load_gt(data["args"]))

    return {
        # Table 1 – Span Partial F1 (detection, partial text, macro)
        "det_f1": te["detection"]["macro_f1"],
        "det_p":  te["detection"]["macro_precision"],
        "det_r":  te["detection"]["macro_recall"],

        # Table 2 – Spatial F1 @ tau (E2E macro F1 per threshold)
        "e2e_025": e2e["@0.25"]["macro_f1"],
        "e2e_050": e2e["@0.5"]["macro_f1"],
        "e2e_075": e2e["@0.75"]["macro_f1"],

        # Table 3 – Exact match metrics
        "char_f1":     summ["char_f1"],
        "emr":         summ["exact_match_rate"],
        "spatial_emr": summ["spatial_exact_match_rate"],

        # Table 4 – Span Exact micro F1/P/R
        "ex_micro_f1": te["span_exact"]["micro"]["f1"],
        "ex_micro_p":  te["span_exact"]["micro"]["precision"],
        "ex_micro_r":  te["span_exact"]["micro"]["recall"],
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────

def pct(v) -> str:
    return "--" if v is None else f"{v * 100:.1f}"


def mean(*vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt_name(name: str) -> str:
    return name.replace("_", r"\_")


# ── Table builders ─────────────────────────────────────────────────────────────

def _fpr_row(name, mb, mm, mh, key_f, key_p, key_r):
    avg_f = mean(mb[key_f], mm[key_f], mh[key_f])
    cells = []
    for m in (mb, mm, mh):
        cells += [pct(m[key_f]), pct(m[key_p]), pct(m[key_r])]
    cells.append(pct(avg_f))
    return f"{fmt_name(name)} & {' & '.join(cells)} \\\\"


# ──────────────────────────────────────────────────────────────────────────────

TABLE2_HEADER = r"""\begin{table*}[t]
\centering
\caption{%
  End-to-end spatial F1 at IoU thresholds $\tau \in \{0.25, 0.50, 0.75\}$
  (macro-averaged over entity types) across Base / Medium / Hard splits.
  A prediction is a true positive when its label matches the ground truth
  \emph{and} the bounding-box IoU exceeds $\tau$.
  Models are sorted by decreasing average Pass Rate as in the main table.
}
\label{tab:spatial_f1_tau}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccc|ccc|ccc|c}
\toprule
 & \multicolumn{3}{c|}{\textbf{Base}}
 & \multicolumn{3}{c|}{\textbf{Medium}}
 & \multicolumn{3}{c|}{\textbf{Hard}}
 & \textbf{AVG} \\
\textbf{Model}
 & $\tau{=}.25$ & $\tau{=}.50$ & $\tau{=}.75$
 & $\tau{=}.25$ & $\tau{=}.50$ & $\tau{=}.75$
 & $\tau{=}.25$ & $\tau{=}.50$ & $\tau{=}.75$
 & F1 \\
\midrule"""

TABLE2_FOOTER = r"""\bottomrule
\end{tabular}%
}
\end{table*}"""


def build_table2(models):
    lines = [TABLE2_HEADER]
    for name, mb, mm, mh in models:
        all_vals = [mb["e2e_025"], mb["e2e_050"], mb["e2e_075"],
                    mm["e2e_025"], mm["e2e_050"], mm["e2e_075"],
                    mh["e2e_025"], mh["e2e_050"], mh["e2e_075"]]
        avg_f = mean(*all_vals)
        cells = [pct(v) for v in all_vals] + [pct(avg_f)]
        lines.append(f"{fmt_name(name)} & {' & '.join(cells)} \\\\")
    lines.append(TABLE2_FOOTER)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────

TABLE3_HEADER = r"""\begin{table*}[t]
\centering
\caption{%
  Text quality metrics across Base / Medium / Hard splits.
  \textbf{CharF1}: SQuAD-style character-level F1 averaged over text-matched entity pairs.
  \textbf{EMR}: fraction of text-matched pairs with exact case-insensitive string match.
  \textbf{ApxF1}: approximate span F1 (detection-level, char-F1 $>0.5$ threshold,
  macro-averaged over entity types).
  Models are sorted by decreasing average Pass Rate as in the main table.
}
\label{tab:text_quality}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccc|ccc|ccc|c}
\toprule
 & \multicolumn{3}{c|}{\textbf{Base}}
 & \multicolumn{3}{c|}{\textbf{Medium}}
 & \multicolumn{3}{c|}{\textbf{Hard}}
 & \textbf{AVG} \\
\textbf{Model}
 & CharF1 & EMR & ApxF1
 & CharF1 & EMR & ApxF1
 & CharF1 & EMR & ApxF1
 & EMR \\
\midrule"""

TABLE3_FOOTER = r"""\bottomrule
\end{tabular}%
}
\end{table*}"""


def build_table3(models):
    lines = [TABLE3_HEADER]
    for name, mb, mm, mh in models:
        avg_emr = mean(mb["emr"], mm["emr"], mh["emr"])
        cells = []
        for m in (mb, mm, mh):
            cells += [pct(m["char_f1"]), pct(m["emr"]), pct(m["det_f1"])]
        cells.append(pct(avg_emr))
        lines.append(f"{fmt_name(name)} & {' & '.join(cells)} \\\\")
    lines.append(TABLE3_FOOTER)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────

TABLE4_HEADER = r"""\begin{table*}[t]
\centering
\caption{%
  Span-exact text F1 with \emph{micro} averaging across Base / Medium / Hard splits.
  Unlike the main table (which reports macro F1), micro averaging weights each entity
  occurrence equally regardless of label frequency, and therefore reflects performance
  on the most common entity types.
  Models are sorted by decreasing average Pass Rate as in the main table.
}
\label{tab:micro_text_f1}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccc|ccc|ccc|c}
\toprule
 & \multicolumn{3}{c|}{\textbf{Base}}
 & \multicolumn{3}{c|}{\textbf{Medium}}
 & \multicolumn{3}{c|}{\textbf{Hard}}
 & \textbf{AVG} \\
\textbf{Model} & F1 & P & R & F1 & P & R & F1 & P & R & F1 \\
\midrule"""

TABLE4_FOOTER = r"""\bottomrule
\end{tabular}%
}
\end{table*}"""


def build_table4(models):
    lines = [TABLE4_HEADER]
    for name, mb, mm, mh in models:
        lines.append(_fpr_row(name, mb, mm, mh, "ex_micro_f1", "ex_micro_p", "ex_micro_r"))
    lines.append(TABLE4_FOOTER)
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

class _ModelAction(argparse.Action):
    """Collect --model NAME BASE MEDIUM HARD into a list of 4-tuples."""
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None) or []
        items.append(tuple(values))
        setattr(namespace, self.dest, items)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", dest="models", metavar=("NAME", "BASE", "MEDIUM", "HARD"),
        nargs=4, action=_ModelAction, required=True,
        help="Model name followed by base / medium / hard result directories. "
             "Repeat for each model.",
    )
    parser.add_argument("--output", default="-",
                        help="Output .tex file (default: stdout).")
    args = parser.parse_args()

    # Load all metrics
    rows = []
    for name, base, medium, hard in args.models:
        print(f"Loading {name} …", file=sys.stderr)
        mb = load_metrics(base)
        mm = load_metrics(medium)
        mh = load_metrics(hard)
        rows.append((name, mb, mm, mh))

    tables = "\n\n".join([
        build_table2(rows),
        build_table3(rows),
        build_table4(rows),
    ])

    if args.output == "-":
        print(tables)
    else:
        Path(args.output).write_text(tables + "\n")
        print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
