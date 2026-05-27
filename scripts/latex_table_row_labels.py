#!/usr/bin/env python3
"""
Generate a LaTeX table row showing span-exact F1 per label, grouped by
macro-category (Name / DoB / Datetime / Age / ID / Contact / Address),
aggregated across all difficulty splits.

Each macro-category group shows one column per sub-label plus an Avg column
(omitted for single-label groups).  Metrics are computed from parsed samples
combined across base / medium / hard.

Usage
-----
python scripts/latex_table_row_labels.py --header
python scripts/latex_table_row_labels.py \
    --base   output/inference/MyModel_base_all_... \
    --medium output/inference/MyModel_medium_all_... \
    --hard   output/inference/MyModel_hard_all_...
python scripts/latex_table_row_labels.py --footer
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.metrics import compute_metrics  # noqa: E402


# ── Taxonomy ───────────────────────────────────────────────────────────────────

TAXONOMY = [
    ("Datetime", [
        "DATETIME",
    ]),
    ("Name", [
        "NAME:PATIENT", "NAME:STAFF", "NAME:ASSOCIATE",
        "NAME:FACILITY", "NAME:DEPARTMENT",
    ]),
    ("DoB", [
        "DATE_OF_BIRTH:UNDER_89", "DATE_OF_BIRTH:OVER_89",
    ]),
    ("Age", [
        "AGE:UNDER_89", "AGE:OVER_89",
    ]),
    ("ID", [
        "ID:PATIENT_ID", "ID:DOCUMENT_ID", "ID:SPECIMEN_ID", "ID:STAFF_ID",
        "ID:DEVICE_ID", "ID:EXAM_ID", "ID:ADMISSION_ID",
    ]),
    ("Contact", [
        "CONTACT:PATIENT", "CONTACT:STAFF", "CONTACT:ASSOCIATE", "CONTACT:FACILITY",
    ]),
    ("Address", [
        "ADDRESS:PATIENT", "ADDRESS:FACILITY",
    ]),
]

SHORT = {
    "NAME:PATIENT":           "Pat.",
    "NAME:STAFF":             "Staff",
    "NAME:ASSOCIATE":         "Assoc.",
    "NAME:FACILITY":          "Fac.",
    "NAME:DEPARTMENT":        "Dept.",
    "DATE_OF_BIRTH:UNDER_89": r"$<$89",
    "DATE_OF_BIRTH:OVER_89":  r"$>$89",
    "DATETIME":               "DT",
    "AGE:UNDER_89":           r"$<$89",
    "AGE:OVER_89":            r"$>$89",
    "ID:PATIENT_ID":          "Pat.",
    "ID:DOCUMENT_ID":         "Doc.",
    "ID:SPECIMEN_ID":         "Spec.",
    "ID:STAFF_ID":            "Staff",
    "ID:DEVICE_ID":           "Dev.",
    "ID:EXAM_ID":             "Exam",
    "ID:ADMISSION_ID":        "Adm.",
    "CONTACT:PATIENT":        "Pat.",
    "CONTACT:STAFF":          "Staff",
    "CONTACT:ASSOCIATE":      "Assoc.",
    "CONTACT:FACILITY":       "Fac.",
    "ADDRESS:PATIENT":        "Pat.",
    "ADDRESS:FACILITY":       "Fac.",
}


# ── GT loading ─────────────────────────────────────────────────────────────────

def _load_gt(args: dict) -> dict:
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
        print(f"[warn] Could not load dataset: {exc}", file=sys.stderr)
        return {}


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_per_label_f1(base_dir, medium_dir, hard_dir):
    """
    Combine parsed samples from all three splits and compute span-exact F1
    per label.  Returns (model_name, {label: f1_float_or_None}).
    """
    all_preds, all_gts = [], []
    model_name = None

    for results_dir in (base_dir, medium_dir, hard_dir):
        if not results_dir:
            continue
        data = json.loads((Path(results_dir) / "results.json").read_text())
        if model_name is None:
            model_name = data["args"].get("model", Path(results_dir).name)
        gt_by_idx = _load_gt(data["args"])

        for s in data["per_sample"]:
            if not s.get("parse_success", False):
                continue
            all_preds.append(s.get("predictions", []))
            all_gts.append(gt_by_idx.get(s["idx"], []))

    if not all_preds:
        return model_name or "unknown", {}

    try:
        m = compute_metrics(all_preds, all_gts)
    except Exception as exc:
        print(f"[warn] compute_metrics failed: {exc}", file=sys.stderr)
        return model_name or "unknown", {}

    per_label = m["text_extraction"]["span_exact"]["per_label"]
    return model_name or "unknown", {
        label: per_label.get(label, {}).get("f1")
        for _, labels in TAXONOMY
        for label in labels
    }


def pct(v) -> str:
    return "--" if v is None else f"{v * 100:.1f}"


# ── LaTeX boilerplate ──────────────────────────────────────────────────────────

# Column layout (30 total):
#  1      : Model
#  2      : Datetime — DT                                          (1)
#  3-8    : Name     — Pat. Staff Assoc. Fac. Dept. | Avg          (6)
#  9-11   : DoB      — <89  >89               | Avg               (3)
#  12-14  : Age      — <89  >89               | Avg               (3)
#  15-22  : ID       — Pat. Doc. Spec. Staff Dev. Exam Adm. | Avg  (8)
#  23-27  : Contact  — Pat. Staff Assoc. Fac.  | Avg              (5)
#  28-30  : Address  — Pat. Fac.               | Avg              (3)

HEADER = r"""\begin{tabular}{l|c|cccccc|ccc|ccc|cccccccc|ccccc|ccc}
\toprule
 & \multicolumn{1}{c|}{\textbf{Datetime}}
 & \multicolumn{6}{c|}{\textbf{Name}}
 & \multicolumn{3}{c|}{\textbf{DoB}}
 & \multicolumn{3}{c|}{\textbf{Age}}
 & \multicolumn{8}{c|}{\textbf{ID}}
 & \multicolumn{5}{c|}{\textbf{Contact}}
 & \multicolumn{3}{c}{\textbf{Address}} \\
\cmidrule(lr){2-2}\cmidrule(lr){3-8}\cmidrule(lr){9-11}
\cmidrule(lr){12-14}\cmidrule(lr){15-22}\cmidrule(lr){23-27}\cmidrule(lr){28-30}
\textbf{Model}
 & DT
 & Pat. & Staff & Assoc. & Fac. & Dept. & \textit{Avg}
 & $<$89 & $>$89 & \textit{Avg}
 & $<$89 & $>$89 & \textit{Avg}
 & Pat. & Doc. & Spec. & Staff & Dev. & Exam & Adm. & \textit{Avg}
 & Pat. & Staff & Assoc. & Fac. & \textit{Avg}
 & Pat. & Fac. & \textit{Avg} \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Emit one LaTeX table row with span-exact F1 per label."
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

    model_name, label_f1 = compute_per_label_f1(args.base, args.medium, args.hard)

    if args.name:
        model_name = args.name
    elif "/" in model_name:
        model_name = model_name.split("/")[-1]
    model_name = model_name.replace("_", r"\_")

    cells = []
    for _, labels in TAXONOMY:
        for label in labels:
            cells.append(pct(label_f1.get(label)))
        if len(labels) > 1:
            vals = [label_f1[l] for l in labels if label_f1.get(l) is not None]
            cells.append(pct(sum(vals) / len(vals) if vals else None))

    print(f"{model_name} & {' & '.join(cells)} \\\\")


if __name__ == "__main__":
    main()
