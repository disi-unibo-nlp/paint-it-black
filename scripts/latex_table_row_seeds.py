#!/usr/bin/env python3
"""
Generate a LaTeX table row for one model broken down by document seed
(MRI / CT / Gyn. / Lab) aggregated across all difficulty splits.

Usage
-----
# Print the table header once:
python scripts/latex_table_row_seeds.py --header

# Append a data row for each model:
python scripts/latex_table_row_seeds.py \
    --base   output/inference/MyModel_base_all_... \
    --medium output/inference/MyModel_medium_all_... \
    --hard   output/inference/MyModel_hard_all_...

# Override the model name shown in the table:
    ... --name "Qwen3.5-27B-FP8"

# Print the closing lines:
python scripts/latex_table_row_seeds.py --footer

Metrics per seed are computed by combining all samples that share the same
doc_type across base / medium / hard splits, then rerunning compute_metrics
on the combined subset (parsed samples only).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.metrics import compute_metrics  # noqa: E402


SEEDS = ["mri_reports", "ct_reports", "gyn_reports", "lab_reports"]
SEED_LABELS = {
    "mri_reports": "MRI",
    "ct_reports":  "CT",
    "gyn_reports": "Gyn.",
    "lab_reports": "Lab",
}


# ── GT loading (mirrors latex_table_row.py) ────────────────────────────────────

def _load_gt(args: dict) -> dict:
    """Load GT annotations from the dataset referenced in results.json args.
    Returns {idx: annotation_list} or {} if unavailable."""
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        print("[warn] 'datasets' not installed — GT unavailable", file=sys.stderr)
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
        print(f"[warn] Could not load dataset — GT unavailable: {exc}", file=sys.stderr)
        return {}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_per_sample(base_dir, medium_dir, hard_dir):
    """
    Load per_sample from all three result dirs and group by doc_type.

    Returns:
        model_name (str)
        by_seed (dict): {seed: {"preds": [...], "gts": [...], "parse_ok": [...]}}
    """
    by_seed = defaultdict(lambda: {"preds": [], "gts": [], "parse_ok": []})
    model_name = None

    for results_dir in (base_dir, medium_dir, hard_dir):
        if not results_dir:
            continue
        data = json.loads((Path(results_dir) / "results.json").read_text())
        if model_name is None:
            model_name = data["args"].get("model", Path(results_dir).name)
        gt_by_idx = _load_gt(data["args"])

        for s in data["per_sample"]:
            seed = s.get("doc_type", "unknown")
            idx  = s["idx"]
            by_seed[seed]["preds"].append(s.get("predictions", []))
            by_seed[seed]["gts"].append(gt_by_idx.get(idx, []))
            by_seed[seed]["parse_ok"].append(s.get("parse_success", False))

    return model_name or "unknown", by_seed


# ── Metric computation ─────────────────────────────────────────────────────────

def metrics_for_seed(seed_data: dict):
    """
    Recompute benchmark metrics for one seed's accumulated samples.
    Only parsed samples are included (matching the full-pipeline behaviour).
    Returns a dict with f1/p/r/e2e/iou/pr keys, or None on failure.
    """
    pairs = [
        (p, g)
        for p, g, ok in zip(seed_data["preds"], seed_data["gts"], seed_data["parse_ok"])
        if ok
    ]
    if not pairs:
        return None
    preds_ok, gts_ok = map(list, zip(*pairs))

    try:
        m = compute_metrics(preds_ok, gts_ok)
    except Exception as exc:
        print(f"[warn] compute_metrics failed: {exc}", file=sys.stderr)
        return None

    return {
        "f1":  m["text_extraction"]["span_exact"]["macro_f1"],
        "p":   m["text_extraction"]["span_exact"]["macro_precision"],
        "r":   m["text_extraction"]["span_exact"]["macro_recall"],
        "e2e": m["bbox_localization"]["avg_e2e_f1"],
        "iou": m["bbox_localization"]["unconditional_mean_iou"],
        "pr":  m["summary"]["pass_rate"],
    }


def pct(v) -> str:
    return "--" if v is None else f"{v * 100:.1f}"


# ── LaTeX boilerplate ──────────────────────────────────────────────────────────

# Column layout: 1(model) + 6*4(seeds) = 25 columns
# MRI:  2-4 text | 5-6 spatial | 7 joint
# CT:   8-10     | 11-12       | 13
# Gyn.: 14-16    | 17-18       | 19
# Lab:  20-22    | 23-24       | 25

HEADER = r"""\begin{tabular}{l|cccccc|cccccc|cccccc|cccccc}
\toprule
 & \multicolumn{6}{c|}{\textbf{MRI}}
 & \multicolumn{6}{c|}{\textbf{CT}}
 & \multicolumn{6}{c|}{\textbf{Gyn.}}
 & \multicolumn{6}{c}{\textbf{Lab}} \\
 & \multicolumn{3}{c}{\textit{Text}} & \multicolumn{2}{c}{\textit{Spatial}} & \textit{Joint}
 & \multicolumn{3}{c}{\textit{Text}} & \multicolumn{2}{c}{\textit{Spatial}} & \textit{Joint}
 & \multicolumn{3}{c}{\textit{Text}} & \multicolumn{2}{c}{\textit{Spatial}} & \textit{Joint}
 & \multicolumn{3}{c}{\textit{Text}} & \multicolumn{2}{c}{\textit{Spatial}} & \textit{Joint} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}
\cmidrule(lr){8-10}\cmidrule(lr){11-12}\cmidrule(lr){13-13}
\cmidrule(lr){14-16}\cmidrule(lr){17-18}\cmidrule(lr){19-19}
\cmidrule(lr){20-22}\cmidrule(lr){23-24}\cmidrule(lr){25-25}
\textbf{Model}
 & F1 & P & R & E2E & IoU & PR
 & F1 & P & R & E2E & IoU & PR
 & F1 & P & R & E2E & IoU & PR
 & F1 & P & R & E2E & IoU & PR \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Emit one LaTeX table row broken down by document seed (mri/ct/gyn/lab)."
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

    model_name, by_seed = load_all_per_sample(args.base, args.medium, args.hard)

    if args.name:
        model_name = args.name
    elif "/" in model_name:
        model_name = model_name.split("/")[-1]
    model_name = model_name.replace("_", r"\_")

    seed_metrics = {seed: metrics_for_seed(by_seed[seed]) for seed in SEEDS}

    cells = []
    for seed in SEEDS:
        m = seed_metrics[seed]
        if m:
            cells += [pct(m["f1"]), pct(m["p"]), pct(m["r"]),
                      pct(m["e2e"]), pct(m["iou"]), pct(m["pr"])]
        else:
            cells += ["--"] * 6

    print(f"{model_name} & {' & '.join(cells)} \\\\")


if __name__ == "__main__":
    main()
