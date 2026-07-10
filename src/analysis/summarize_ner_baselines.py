"""
Aggregate NER baseline results for a given split into a Markdown report.

Scans output/ner_baseline/ for run directories whose name ends with _{split},
loads each results.json, and writes a single .md summary to the same directory.

Usage:
    python src/analysis/summarize_ner_baselines.py
    python src/analysis/summarize_ner_baselines.py --split base
    python src/analysis/summarize_ner_baselines.py --output_dir output/ner_baseline
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.labels import load_labels


def _load_runs(output_dir: Path, split: str) -> list[dict]:
    runs = []
    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        data = json.loads(results_path.read_text())
        if data["args"].get("input_split") != split:
            continue
        if "span_exact" not in data["metrics"]:
            print(f"  Skipping {run_dir.name} — old results.json format, re-run the baseline to update.",
                  file=sys.stderr)
            continue
        runs.append({
            "run_name": run_dir.name,
            "args":     data["args"],
            "metrics":  data["metrics"],
        })
    return runs


def _fmt(v: float, pct: bool = False) -> str:
    if pct:
        return f"{v * 100:.1f}"
    return f"{v:.4f}"


def _build_report(runs: list[dict], split: str, labels: list[str]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# NER Baseline Results — split: `{split}`",
        f"",
        f"Generated: {ts}  ",
        f"Runs: {len(runs)}",
        f"",
    ]

    # ── Model inventory ───────────────────────────────────────────────────────
    lines += ["## Models\n"]
    lines += ["| # | Config | Model | Backend | Threshold |"]
    lines += ["|---|--------|-------|---------|-----------|"]
    for i, r in enumerate(runs, 1):
        a = r["args"]
        thr = a.get("threshold", "—")
        lines.append(
            f"| {i} | `{a.get('config', r['run_name'])}` "
            f"| `{a['model']}` "
            f"| {a['backend']} "
            f"| {thr} |"
        )
    lines.append("")

    # ── Summary table ─────────────────────────────────────────────────────────
    lines += ["## Summary\n"]
    lines += [
        "All values in %. "
        "Halluc = hallucination rate (FP / predicted); Miss = miss rate (FN / GT).\n"
        "Both Halluc and Miss are computed from span-exact matching.\n"
    ]
    lines += ["| Config | Span P | Span R | Span F1 | Macro F1 | Halluc↓ | Miss↓ |"]
    lines += ["|--------|-------:|-------:|--------:|---------:|--------:|------:|"]
    for r in runs:
        a   = r["args"]
        ex  = r["metrics"]["span_exact"]
        s   = r["metrics"]["summary"]
        cfg = a.get("config", r["run_name"])
        lines.append(
            f"| `{cfg}` "
            f"| {_fmt(ex['micro']['precision'], pct=True)} "
            f"| {_fmt(ex['micro']['recall'], pct=True)} "
            f"| **{_fmt(ex['micro']['f1'], pct=True)}** "
            f"| **{_fmt(ex['macro_f1'], pct=True)}** "
            f"| {_fmt(s['hallucination_rate'], pct=True)} "
            f"| {_fmt(s['miss_rate'], pct=True)} |"
        )
    lines.append("")

    # ── Per-label span exact F1 ───────────────────────────────────────────────
    lines += ["## Per-label span exact F1 (%)\n"]
    cfgs   = [r["args"].get("config", r["run_name"]) for r in runs]
    header = "| Label | " + " | ".join(f"`{c}`" for c in cfgs) + " |"
    sep    = "|-------|" + "|".join(["------:"] * len(runs)) + "|"
    lines += [header, sep]

    present_labels = set()
    for r in runs:
        present_labels |= set(r["metrics"]["span_exact"]["per_label"].keys())
    ordered = [l for l in labels if l in present_labels]
    ordered += sorted(present_labels - set(labels))

    for label in ordered:
        cells = []
        for r in runs:
            pl  = r["metrics"]["span_exact"]["per_label"]
            f1  = pl.get(label, {}).get("f1", None)
            cells.append(_fmt(f1, pct=True) if f1 is not None else "—")
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    lines.append("")

    # ── Per-label hallucination rate ──────────────────────────────────────────
    lines += ["## Per-label hallucination rate (%)\n"]
    lines += [header, sep]

    for label in ordered:
        cells = []
        for r in runs:
            hr  = r["metrics"]["hallucination_rate"]["per_label"]
            v   = hr.get(label, None)
            cells.append(_fmt(v, pct=True) if v is not None else "—")
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    lines.append("")

    # ── Run configs ───────────────────────────────────────────────────────────
    lines += ["## Run configs\n"]
    for r in runs:
        a   = r["args"]
        cfg = a.get("config", r["run_name"])
        lines += [
            f"### `{cfg}`",
            f"",
            f"- **Model**: `{a['model']}`",
            f"- **Backend**: {a['backend']}",
            f"- **Split**: {a['input_split']}",
            f"- **Samples**: {a.get('max_samples') or 'all'}",
            f"- **Threshold**: {a.get('threshold', '—')}",
            f"- **Chunking**: {'grouped (' + str(a.get('max_chunk_words')) + ' words)' if a.get('group_sentences') else 'sentence-level'}",
            f"",
        ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize NER baseline results into a Markdown report."
    )
    parser.add_argument("--split", type=str, default="base",
                        help="Dataset split to aggregate (default: base)")
    parser.add_argument("--output_dir", type=str, default="output/ner_baseline",
                        help="Directory containing run subdirectories")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    labels = load_labels()
    runs = _load_runs(output_dir, args.split)

    if not runs:
        print(f"No results found for split '{args.split}' in {output_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} run(s) for split '{args.split}':")
    for r in runs:
        print(f"  {r['run_name']}")

    report = _build_report(runs, args.split, labels)
    out_path = output_dir / f"baseline_results_{args.split}.md"
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
