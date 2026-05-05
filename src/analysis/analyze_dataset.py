"""
Analyze statistics for a pre-built HuggingFace dataset.

Loads a dataset from HF Hub or local disk and prints comprehensive statistics:
- Page counts (total, with annotations, empty)
- Annotation counts and averages
- Label distribution with visual bar charts
- Rare labels showing source PDFs
- Balance metrics (imbalance ratio)
- PDFs per doc_type
- Per-doc-type label breakdown

Usage:
    python src/analysis/analyze_dataset.py --dataset dfreddi/multimodal-deid
    python src/analysis/analyze_dataset.py --dataset data/test_dataset --local --split base
"""

import os
from pathlib import Path
from collections import Counter
from logging import Logger
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import login
from core.utils import ConfigArgumentParser, init_logger
from core.labels import load_labels

_RARE_THRESHOLD = 10  # show source PDFs for labels with at most this many occurrences

def log_dataset_stats(dataset: Dataset, logger: Logger) -> None:
    """
    Log comprehensive statistics for a dataset loaded from HF Hub or disk.

    Reconstructs per-PDF and per-doc-type information from dataset rows,
    then prints the same statistics as the original _log_annotation_stats().
    """
    # Initialize label tracking
    canonical = load_labels()
    label_counts: Counter = Counter({label: 0 for label in canonical})
    label_to_pdfs: dict = {}
    pages_with_annos = 0
    total_annos = 0

    # Track unique PDFs and doc_types
    pdf_names = set()
    doc_type_to_pdfs = {}
    by_type: dict = {}

    # Process all dataset rows
    for row in dataset:
        pdf_name = row["source_pdf"]
        doc_type = row["doc_type"]

        # Track unique PDFs
        pdf_names.add(pdf_name)

        # Track PDFs per doc_type
        doc_type_to_pdfs.setdefault(doc_type, set()).add(pdf_name)

        # Count pages with annotations
        if row["annotations"]:
            pages_with_annos += 1
        total_annos += len(row["annotations"])

        # Initialize per-doc-type counter if needed
        if doc_type not in by_type:
            by_type[doc_type] = Counter({l: 0 for l in canonical})

        # Count labels
        for anno in row["annotations"]:
            label = anno["label"]
            label_counts[label] += 1
            label_to_pdfs.setdefault(label, set()).add(pdf_name)
            by_type[doc_type][label] += 1

    # ── Overall statistics ────────────────────────────────────────────────────
    total_pages = len(dataset)
    logger.info("─── Annotation Statistics ───────────────────────────────")
    logger.info("Pages:        %d total  |  %d with annotations  |  %d empty",
                total_pages, pages_with_annos, total_pages - pages_with_annos)
    logger.info("Annotations:  %d total  |  %.2f avg per page  |  %.2f avg per annotated page",
                total_annos,
                total_annos / total_pages if total_pages else 0,
                total_annos / pages_with_annos if pages_with_annos else 0)

    # ── Label distribution ────────────────────────────────────────────────────
    if label_counts:
        logger.info("Label distribution (%d unique labels):", len(label_counts))
        max_count = max(label_counts.values()) or 1
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(20 * count / max_count)
            pdfs = label_to_pdfs.get(label, set())
            if count == 0:
                logger.info("  %-32s %4d  (none)", label, count)
            elif count <= _RARE_THRESHOLD:
                logger.info("  %-32s %4d  %s  ← %s", label, count, bar, ", ".join(sorted(pdfs)))
            else:
                logger.info("  %-32s %4d  %s", label, count, bar)

        counts = list(label_counts.values())
        rarest, most_common = min(counts), max(counts)
        logger.info("Balance:  most common=%d  rarest=%d  imbalance ratio=%.1fx",
                    most_common, rarest, most_common / rarest if rarest else float("inf"))

    # ── PDFs per doc_type ─────────────────────────────────────────────────────
    logger.info("PDFs included per doc_type (%d types):", len(doc_type_to_pdfs))
    for doc_type, pdfs in sorted(doc_type_to_pdfs.items()):
        logger.info("  %-48s %d PDF(s)", doc_type, len(pdfs))

    # ── Per-doc-type label breakdown ──────────────────────────────────────────
    for doc_type, counts in sorted(by_type.items()):
        total = sum(counts.values())
        logger.info("── %s  (%d annotations) ──", doc_type, total)
        max_count = max(counts.values()) or 1
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(20 * count / max_count)
            if count == 0:
                logger.info("  %-32s %4d  (none)", label, count)
            else:
                logger.info("  %-32s %4d  %s", label, count, bar)

    logger.info("─────────────────────────────────────────────────────────")

def main():
    parser = ConfigArgumentParser(description="Analyze HF dataset statistics")
    parser.add_argument("--dataset", required=True,
                        help="HF repo ID or local dataset root path")
    parser.add_argument("--split", default="base",
                        help="Dataset split to load")
    parser.add_argument("--local", action="store_true",
                        help="Load from local disk instead of HF Hub")
    parser.add_argument("--log_level", default="INFO",
                        help="Logging level")

    args = parser.parse_args()
    logger = init_logger(__name__, level=args.log_level)

    # Optional HF authentication
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    # Load dataset
    logger.info("Loading dataset...")
    if args.local:
        dataset = load_from_disk(str(Path(args.dataset) / args.split))
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    logger.info(f"Loaded {len(dataset)} rows from split '{args.split}'")

    # Analyze and log statistics
    log_dataset_stats(dataset, logger)

if __name__ == "__main__":
    main()
