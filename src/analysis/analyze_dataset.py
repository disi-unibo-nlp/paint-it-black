"""
Analyze statistics for a pre-built HuggingFace dataset and write a markdown report.

Loads a dataset from HF Hub or local disk, collects statistics, and writes a
timestamped report to output/analysis_<split>_<timestamp>.md.

Usage:
    python src/analysis/analyze_dataset.py --dataset disi-unibo-nlp/paint-it-black
    python src/analysis/analyze_dataset.py --dataset data/test_dataset --local --split base
"""

import os
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets as _ds
_ds.logging.set_verbosity_error()
_ds.disable_progress_bar()
from datasets import Dataset, load_dataset, load_from_disk
import huggingface_hub as _hfhub
_hfhub.utils.disable_progress_bars()
from huggingface_hub import login

from core.utils import ConfigArgumentParser, init_logger
from core.labels import load_labels

_RARE_THRESHOLD = 10
_DENSITY_BUCKETS = [("0", 0, 0), ("1–5", 1, 5), ("6–15", 6, 15), ("16+", 16, 9999)]


# ── Stats collection ──────────────────────────────────────────────────────────

def collect_stats(dataset: Dataset, canonical: list[str]) -> dict:
    s = {
        "total_pages":        len(dataset),
        "pages_with_annos":   0,
        "total_annos":        0,
        "label_counts":       Counter({l: 0 for l in canonical}),
        "label_page_presence": Counter(),
        "label_to_pdfs":      defaultdict(set),
        "label_text_lengths": defaultdict(list),
        "label_bbox_counts":  defaultdict(list),
        "pdf_names":          set(),
        "pages_per_pdf":      Counter(),
        "doc_type_to_pdfs":   defaultdict(set),
        "by_type":            defaultdict(lambda: Counter({l: 0 for l in canonical})),
        "density":            Counter(),
        "cooccurrence":       Counter(),
        "country_counts":     Counter(),
        "sex_counts":         Counter(),
    }

    for row in dataset:
        pdf   = row["source_pdf"]
        dtype = row["doc_type"]
        annos = row["annotations"]

        s["pdf_names"].add(pdf)
        s["pages_per_pdf"][pdf] += 1
        s["doc_type_to_pdfs"][dtype].add(pdf)

        n = len(annos)
        s["total_annos"] += n
        if n:
            s["pages_with_annos"] += 1

        bucket = next(k for k, lo, hi in _DENSITY_BUCKETS if lo <= n <= hi)  # noqa: F841 (hi used in condition)
        s["density"][bucket] += 1

        m_country = re.search(r'_(uk|us|canada|australia)_', pdf, re.I)
        if m_country:
            s["country_counts"][m_country.group(1).lower()] += 1
        m_sex = re.search(r'_(male|female)(?:_|\.)', pdf, re.I)
        if m_sex:
            s["sex_counts"][m_sex.group(1).lower()] += 1

        labels_on_page = set()
        for ann in annos:
            label = ann["label"]
            s["label_counts"][label] += 1
            s["label_page_presence"][label] += 1
            s["label_to_pdfs"][label].add(pdf)
            s["label_text_lengths"][label].append(len(ann.get("text", "")))
            s["label_bbox_counts"][label].append(len(ann.get("bboxes", [])))
            s["by_type"][dtype][label] += 1
            labels_on_page.add(label)

        labels_sorted = sorted(labels_on_page)
        for i in range(len(labels_sorted)):
            for j in range(i + 1, len(labels_sorted)):
                s["cooccurrence"][frozenset([labels_sorted[i], labels_sorted[j]])] += 1

    return s


# ── Report rendering ──────────────────────────────────────────────────────────

def _pct(n, total):
    return f"{100*n/total:.1f}%" if total else "—"

def _avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def build_report(s: dict, split: str, dataset_name: str) -> str:
    lines = []
    total      = s["total_pages"]
    annotated  = s["pages_with_annos"]
    n_annos    = s["total_annos"]
    n_pdfs     = len(s["pdf_names"])
    multi_page = sum(1 for c in s["pages_per_pdf"].values() if c > 1)

    def h(text): lines.extend([f"## {text}", ""])
    def h3(text): lines.extend([f"### {text}", ""])
    def row(*cols): lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    def sep(*widths): lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    def blank(): lines.append("")

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("# Dataset Analysis Report")
    blank()
    lines.append(f"**Dataset:** `{dataset_name}` &nbsp; **Split:** `{split}` &nbsp; **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    blank()

    # ── Overview ──────────────────────────────────────────────────────────────
    h("Overview")
    row("Metric", "Value")
    sep(32, 16)
    row("Total pages", total)
    row("Annotated pages", f"{annotated} ({_pct(annotated, total)})")
    row("Empty pages", f"{total - annotated} ({_pct(total - annotated, total)})")
    row("Total annotations", n_annos)
    row("Avg annotations / page", f"{n_annos/total:.2f}" if total else "—")
    row("Avg annotations / annotated page", f"{n_annos/annotated:.2f}" if annotated else "—")
    row("Unique PDFs", n_pdfs)
    row("Multi-page PDFs", multi_page)
    row("Doc types", len(s["doc_type_to_pdfs"]))
    blank()

    # ── Annotation density ────────────────────────────────────────────────────
    h("Annotation Density")
    row("Annotations / page", "Pages", "%")
    sep(20, 8, 8)
    for key, lo, hi in _DENSITY_BUCKETS:
        c = s["density"].get(key, 0)
        row(key, c, _pct(c, total))
    blank()

    # ── Label distribution ────────────────────────────────────────────────────
    h("Label Distribution")
    row("Label", "Count", "% annos", "Pages", "Avg/page", "Avg chars", "Multi-bbox %")
    sep(28, 7, 8, 7, 9, 10, 13)
    for label, count in sorted(s["label_counts"].items(), key=lambda x: -x[1]):
        pages   = s["label_page_presence"].get(label, 0)
        bboxes  = s["label_bbox_counts"].get(label, [])
        texts   = s["label_text_lengths"].get(label, [])
        multi   = _pct(sum(1 for b in bboxes if b > 1), len(bboxes)) if bboxes else "—"
        avg_ch  = f"{_avg(texts):.0f}" if texts else "—"
        avg_pp  = f"{count/pages:.1f}" if pages else "—"
        row(f"`{label}`", count, _pct(count, n_annos), pages, avg_pp, avg_ch, multi)
    blank()

    counts_nz = [c for c in s["label_counts"].values() if c > 0]
    if counts_nz:
        most, least = max(counts_nz), min(counts_nz)
        zero = [l for l, c in s["label_counts"].items() if c == 0]
        lines.append(f"**Imbalance:** most common = {most} · rarest = {least} · ratio = {most/least:.1f}×")
        if zero:
            lines.append(f"  \n**Zero-count labels:** {', '.join(f'`{l}`' for l in sorted(zero))}")
        blank()

    # ── Rare labels ───────────────────────────────────────────────────────────
    rare = {l: (c, sorted(s["label_to_pdfs"].get(l, set())))
            for l, c in s["label_counts"].items() if 0 < c <= _RARE_THRESHOLD}
    if rare:
        h(f"Rare Labels (≤ {_RARE_THRESHOLD} occurrences)")
        row("Label", "Count", "Source PDFs")
        sep(28, 7, 50)
        for label, (count, pdfs) in sorted(rare.items(), key=lambda x: x[1][0]):
            row(f"`{label}`", count, ", ".join(pdfs))
        blank()

    # ── Co-occurrence ─────────────────────────────────────────────────────────
    if s["cooccurrence"]:
        h("Top Label Co-occurrences (same page)")
        row("Label A", "Label B", "Pages")
        sep(28, 28, 7)
        for pair, count in s["cooccurrence"].most_common(15):
            a, b = sorted(pair)
            row(f"`{a}`", f"`{b}`", count)
        blank()

    # ── Source breakdown ──────────────────────────────────────────────────────
    if s["country_counts"] or s["sex_counts"]:
        h("Source Breakdown (from filenames)")
        if s["country_counts"]:
            lines.append("**Country**")
            blank()
            row("Country", "Pages")
            sep(12, 8)
            for country, count in s["country_counts"].most_common():
                row(country, count)
            blank()
        if s["sex_counts"]:
            lines.append("**Sex**")
            blank()
            row("Sex", "Pages")
            sep(10, 8)
            for sex, count in s["sex_counts"].most_common():
                row(sex, count)
            blank()

    # ── Per doc-type ──────────────────────────────────────────────────────────
    h("Per Doc-Type Breakdown")
    for dtype, type_counts in sorted(s["by_type"].items()):
        n_type_pdfs = len(s["doc_type_to_pdfs"][dtype])
        total_type  = sum(type_counts.values())
        h3(f"{dtype}  ({n_type_pdfs} PDFs · {total_type} annotations)")
        row("Label", "Count", "%")
        sep(28, 7, 8)
        for label, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            if count:
                row(f"`{label}`", count, _pct(count, total_type))
        blank()

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = ConfigArgumentParser(description="Analyze HF dataset statistics")
    parser.add_argument("--dataset",    default="disi-unibo-nlp/paint-it-black")
    parser.add_argument("--split",      default="base")
    parser.add_argument("--local",      action="store_true")
    parser.add_argument("--log_level",  default="INFO")
    parser.add_argument("--output_dir", default="output")

    args   = parser.parse_args()
    logger = init_logger(__name__, level=args.log_level)

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    logger.info("Loading '%s' (split: %s)...", args.dataset, args.split)
    if args.local:
        dataset = load_from_disk(str(Path(args.dataset) / args.split))
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    canonical = load_labels()
    stats     = collect_stats(dataset, canonical)
    report    = build_report(stats, args.split, args.dataset)

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"analysis_{args.split}_{ts}.md"
    out_path.write_text(report)

    logger.info("Report → %s", out_path)
    logger.info("%d pages · %d PDFs · %d annotations · %d active label types",
                stats["total_pages"], len(stats["pdf_names"]), stats["total_annos"],
                sum(1 for c in stats["label_counts"].values() if c > 0))


if __name__ == "__main__":
    main()
