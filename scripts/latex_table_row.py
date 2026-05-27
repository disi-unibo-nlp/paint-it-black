#!/usr/bin/env python3
"""
Generate a LaTeX table row for one model across base / medium / hard splits.

Usage
-----
# Print the table header once:
python scripts/latex_table_row.py --header

# Append a data row for each model:
python scripts/latex_table_row.py \
    --base   output/inference/MyModel_base_all_... \
    --medium output/inference/MyModel_medium_all_... \
    --hard   output/inference/MyModel_hard_all_...

# Override the model name shown in the table:
    ... --name "Qwen3.5-27B-FP8"

# Print the closing lines:
python scripts/latex_table_row.py --footer

Pass Rate (PR)
--------------
PR is the fraction of GT entities where the model simultaneously hit correct
label + span (char-F1 > 0.5) + bbox (IoU > 0.5).  It is computed by
reloading the original HF dataset (or local split) referenced in results.json
args, since GT annotations are not stored in per_sample.  Falls back to '--'
if the dataset is unavailable.  Future inference runs that use the updated
metrics.py will store pass_rate directly in results.json and avoid the reload.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# ── Inline geometry / text helpers (mirrors src/inference/metrics.py) ─────────

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
    iy1 = max(pb[0], gb[0]);  ix1 = max(pb[1], gb[1])
    iy2 = min(pb[2], gb[2]);  ix2 = min(pb[3], gb[3])
    inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
    union = (pb[2]-pb[0])*(pb[3]-pb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter
    return inter / union if union > 0 else 0.0


def _char_f1(pred, gt):
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    pc, gc = Counter(pred), Counter(gt)
    common = sum((pc & gc).values())
    if not common:
        return 0.0
    p = common / len(pred);  r = common / len(gt)
    return 2 * p * r / (p + r)


# ── Pass-rate computation ──────────────────────────────────────────────────────

def _compute_pass_rate(per_sample, gt_by_idx, iou_thr=0.5):
    """
    For each GT entity across all successfully parsed samples, check whether
    the model produced a prediction with the same label AND exact text match
    (case-insensitive) AND bbox IoU > iou_thr simultaneously.  Greedy
    bipartite matching avoids double-counting predictions.

    Returns fraction in [0, 1], or None if gt_by_idx is empty.
    """
    if not gt_by_idx:
        return None

    total_gt = joint_tp = 0
    for s in per_sample:
        if not s.get("parse_success", False):
            continue
        gts  = gt_by_idx.get(s["idx"], [])
        preds = s.get("predictions", [])
        total_gt += len(gts)

        # Build (score, pred_idx, gt_idx) candidates where both conditions hold:
        # exact text match (case-insensitive) AND bbox IoU > iou_thr
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
                used_p.add(pi);  used_g.add(gi);  joint_tp += 1

    return joint_tp / total_gt if total_gt > 0 else None


def _load_gt(args):
    """
    Load GT annotations from the dataset referenced in results.json args.
    Returns {idx: annotation_list} or {} if unavailable.
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        print("[warn] 'datasets' not installed — PR column will be '--'", file=sys.stderr)
        return {}
    try:
        if args.get("from_hub"):
            ds = load_dataset(args["input_dataset"], split=args["input_split"])
        else:
            split_path = Path(args["input_dataset"]) / args["input_split"]
            ds = load_from_disk(str(split_path))
        rows = list(ds)
        return {i: rows[i].get("annotations", []) for i in range(len(rows))}
    except Exception as exc:
        print(f"[warn] Could not load dataset — PR column will be '--': {exc}", file=sys.stderr)
        return {}


# ── Metric loading ─────────────────────────────────────────────────────────────

def load_metrics(results_dir: str) -> tuple:
    path = Path(results_dir) / "results.json"
    data = json.loads(path.read_text())
    model = data["args"].get("model", Path(results_dir).name)
    m = data["metrics"]

    # Fast path: future runs store pass_rate directly in results.json
    if "pass_rate" in m.get("summary", {}):
        pr = m["summary"]["pass_rate"]
    else:
        gt_by_idx = _load_gt(data["args"])
        pr = _compute_pass_rate(data["per_sample"], gt_by_idx)

    return model, {
        "f1":  m["text_extraction"]["span_exact"]["macro_f1"],
        "p":   m["text_extraction"]["span_exact"]["macro_precision"],
        "r":   m["text_extraction"]["span_exact"]["macro_recall"],
        "e2e": m["bbox_localization"]["avg_e2e_f1"],
        "iou": m["bbox_localization"]["unconditional_mean_iou"],
        "pr":  pr,
    }


def pct(v) -> str:
    return "--" if v is None else f"{v * 100:.1f}"


HEADER = r"""\begin{tabular}{l|cccccc|cccccc|cccccc|ccc}
\toprule
 & \multicolumn{6}{c|}{\textbf{Base}}
 & \multicolumn{6}{c|}{\textbf{Medium}}
 & \multicolumn{6}{c|}{\textbf{Hard}}
 & \multicolumn{3}{c}{\textbf{AVG}} \\
 & \multicolumn{3}{c}{\textit{Text}} & \multicolumn{2}{c}{\textit{Spatial}} & \textit{Joint}
 & \multicolumn{3}{c}{\textit{Text}} & \multicolumn{2}{c}{\textit{Spatial}} & \textit{Joint}
 & \multicolumn{3}{c}{\textit{Text}} & \multicolumn{2}{c}{\textit{Spatial}} & \textit{Joint}
 & \textit{Text} & \textit{Spatial} & \textit{Joint} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}
\cmidrule(lr){8-10}\cmidrule(lr){11-12}\cmidrule(lr){13-13}
\cmidrule(lr){14-16}\cmidrule(lr){17-18}\cmidrule(lr){19-19}
\cmidrule(lr){20-20}\cmidrule(lr){21-21}\cmidrule(lr){22-22}
\textbf{Model}
 & F1 & P & R & E2E & IoU & PR
 & F1 & P & R & E2E & IoU & PR
 & F1 & P & R & E2E & IoU & PR
 & F1 & E2E & PR \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}"""


def main():
    parser = argparse.ArgumentParser(
        description="Emit one LaTeX table row for a model's base/medium/hard results."
    )
    parser.add_argument("--base",   help="Path to base results directory")
    parser.add_argument("--medium", help="Path to medium results directory")
    parser.add_argument("--hard",   help="Path to hard results directory")
    parser.add_argument("--name",   default=None,
                        help="Override model name shown in the table")
    parser.add_argument("--header", action="store_true",
                        help="Print the table header and exit")
    parser.add_argument("--footer", action="store_true",
                        help="Print the table footer and exit")
    args = parser.parse_args()

    if args.header:
        print(HEADER)
        return

    if args.footer:
        print(FOOTER)
        return

    if not (args.base and args.medium and args.hard):
        parser.error("--base, --medium, and --hard are all required")

    model_name, mb = load_metrics(args.base)
    _,          mm = load_metrics(args.medium)
    _,          mh = load_metrics(args.hard)

    if args.name:
        model_name = args.name
    elif "/" in model_name:
        model_name = model_name.split("/")[-1]
    model_name = model_name.replace("_", r"\_")

    avg_f1  = (mb["f1"]  + mm["f1"]  + mh["f1"])  / 3
    avg_e2e = (mb["e2e"] + mm["e2e"] + mh["e2e"]) / 3
    pr_vals = [v["pr"] for v in (mb, mm, mh) if v["pr"] is not None]
    avg_pr  = sum(pr_vals) / len(pr_vals) if pr_vals else None

    cells = []
    for m in (mb, mm, mh):
        cells += [pct(m["f1"]), pct(m["p"]), pct(m["r"]), pct(m["e2e"]), pct(m["iou"]), pct(m["pr"])]
    cells += [pct(avg_f1), pct(avg_e2e), pct(avg_pr)]

    print(f"{model_name} & {' & '.join(cells)} \\\\")


if __name__ == "__main__":
    main()
