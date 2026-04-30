import os
import json
from collections import Counter
from pathlib import Path
from typing import List
from logging import Logger
from core.utils import ConfigArgumentParser, init_logger
from core.labels import load_labels
from datasets import Dataset, DatasetDict
from datasets import Image as HFImage
from huggingface_hub import login
from pdf2image import convert_from_path

"""
ANNOTATION JSON STRUCTURE

The root object has three keys: files, created, and updated (ISO 8601 timestamps for the annotation
session).

files is a dict keyed by PDF filename. Each file entry has:
- status: string ("completed" or "in_progress")
- pages: dict keyed by page number as a string (e.g. "1"). Each page is a list of annotation objects.

Each annotation object has:
- id: unique string identifier
- label: string in CATEGORY:SUBCATEGORY format (e.g. "NAME:PATIENT", "DATE", "ID:DOCUMENT_ID")
- text: the annotated text content as it appears in the document
- page_width, page_height: page dimensions in points (PDF units)
- created: ISO 8601 timestamp
- bboxes: list (of atleast 1 element) of bounding boxes — each bbox is a list of 4 floats [x_min, y_min, x_max, y_max]
normalized to [0, 1] relative to page dimensions. A single annotation can have multiple bounding boxes
(e.g. for multi-line spans).
"""

def process_leaf_directory(leaf_dir: str, root_dir: str, logger: Logger) -> List[dict]:
    """
    Gets the instance list for a single leaf directory. Performs the necessary checks and logs errors/warnings as needed.

    Here are the main checks:
    - ERRORS (skip entire leaf)
        - check if the leaf directory contains atleast a PDF file and a JSON file
        - check that the json file is unique
    - ERRORS (skip specific pdf file)
        - check that there are no missing annotations in the json file (e.g. missing page numbers, pid with an empty bounding box list, etc.)
        - check that there are no missing pdfs (e.g. pdfs that are in the annotations but not in the directory, or pdfs that are in the directory but not in the annotations)
    - WARNINGS
        - if there are instances still "in_progress", log them and skip (but not the entire leaf directory)
        - if there are "completed" instances that are correctly formatted but have NO annotations, include them but with a warning (no skipping here)
    """
    instances = []
    leaf = Path(leaf_dir)
    rel_leaf = leaf.relative_to(root_dir)

    pdfs_in_dir = {p.name for p in leaf.glob("*.pdf")}
    jsons_in_dir = list(leaf.glob("*.json"))

    # ERRORS: leaf-level checks
    if not pdfs_in_dir or not jsons_in_dir:
        logger.error(f"{rel_leaf}/ — Missing PDF or JSON files ({len(pdfs_in_dir)} PDFs skipped)")
        return []

    if len(jsons_in_dir) > 1:
        logger.error(
            f"{rel_leaf}/ — Multiple JSON files found ({[j.name for j in jsons_in_dir]}) ({len(pdfs_in_dir)} PDFs skipped)"
        )
        return []

    with open(jsons_in_dir[0]) as f:
        data = json.load(f)

    pdfs_in_json = set(data["files"].keys())
    skipped = 0

    # ERROR (per-PDF): PDF in directory but not in annotations
    for pdf_name in pdfs_in_dir - pdfs_in_json:
        logger.error(f"{rel_leaf / pdf_name} — in directory but not in annotations")
        skipped += 1

    for pdf_name, entry in data["files"].items():
        status = entry["status"]
        pages = entry["pages"]
        rel_pdf = rel_leaf / pdf_name

        # ERROR: PDF in annotations but not in directory
        if pdf_name not in pdfs_in_dir:
            logger.error(f"{rel_pdf} — in annotations but not in directory")
            skipped += 1
            continue

        # WARNING: not completed
        if status in ("pending", "in_progress"):
            logger.warning(f"{rel_pdf} — {status}, skipping")
            skipped += 1
            continue

        # ERROR: annotation with empty bboxes list
        bad_annotation = False
        for page_num, annotations in pages.items():
            for ann in annotations:
                if not ann.get("bboxes"):
                    logger.error(f"{rel_pdf} — page {page_num} annotation '{ann['id']}' has empty bboxes")
                    bad_annotation = True
                    break
            if bad_annotation:
                break
        if bad_annotation:
            skipped += 1
            continue

        # WARNING: completed but zero annotations across all pages
        total_annotations = sum(len(anns) for anns in pages.values())
        if total_annotations == 0:
            logger.warning(f"{rel_pdf} — completed with no annotations across all pages")

        instances.append({
            "pdf_path": str(leaf / pdf_name),
            "annotations": pages,
            "rel_dir": str(rel_leaf),
        })

    if skipped:
        logger.info(f"{rel_leaf}/: {len(instances)} included, {skipped} skipped")

    return instances

def build_instance_list(input_dir: str, logger: Logger) -> List[dict]:
    """
    Build a list of instances from the input directory.

    Args:
        input_dir (str): Root directory containing annotation cases.
        logger (Logger): Logger for logging information.

    Returns:
        List[dict]: a list of instances as dicts, each dict contains:
            - pdf_path: str, path to the PDF file
            - annotations: the content of the "pages" key in the annotation JSON file for the corresponding PDF
    """
    # get all leaf directories in the input directory
    leaf_dirs = [
        dirpath
        for dirpath, dirs, files in os.walk(input_dir)
        if not dirs  # no subdirectories → leaf
    ]

    instances = []
    for leaf in leaf_dirs:
        leaf_instances = process_leaf_directory(leaf, input_dir, logger)
        instances.extend(leaf_instances)

    return instances

def build_dataset(instances: List[dict], dpi: int, logger: Logger) -> Dataset:
    "For each instance, convert the pdf to image(s) and return an annotated Hugging Face Dataset"
    rows = []
    for inst in instances:
        pages_images = convert_from_path(inst["pdf_path"], dpi=dpi)
        doc_type    = Path(inst["rel_dir"]).parts[0]
        total_pages = len(pages_images)
        source_pdf  = Path(inst["pdf_path"]).name
        for i, pil_img in enumerate(pages_images):
            page_num = i + 1
            page_key = str(page_num)
            rows.append({
                "image":       pil_img,
                "page":        page_num,
                "total_pages": total_pages,
                "doc_type":    doc_type,
                "source_pdf":  source_pdf,
                "annotations": inst["annotations"].get(page_key, []),
            })
    logger.info(f"Built dataset with {len(rows)} rows from {len(instances)} PDFs")
    dataset = Dataset.from_list(rows).cast_column("image", HFImage())
    return dataset

_RARE_THRESHOLD = 10  # show source PDFs for labels with at most this many occurrences

def _log_annotation_stats(dataset: Dataset, instances: List[dict], logger: Logger) -> None:
    label_counts: Counter = Counter({label: 0 for label in load_labels()})
    label_to_pdfs: dict = {}
    pages_with_annos = 0
    total_annos = 0

    for inst in instances:
        pdf_name = Path(inst["pdf_path"]).name
        for annos in inst["annotations"].values():
            for a in annos:
                label = a["label"]
                label_counts[label] += 1
                label_to_pdfs.setdefault(label, set()).add(pdf_name)

    for row in dataset:
        if row["annotations"]:
            pages_with_annos += 1
        total_annos += len(row["annotations"])

    total_pages = len(dataset)
    logger.info("─── Annotation Statistics ───────────────────────────────")
    logger.info("Pages:        %d total  |  %d with annotations  |  %d empty",
                total_pages, pages_with_annos, total_pages - pages_with_annos)
    logger.info("Annotations:  %d total  |  %.2f avg per page  |  %.2f avg per annotated page",
                total_annos,
                total_annos / total_pages if total_pages else 0,
                total_annos / pages_with_annos if pages_with_annos else 0)

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

    dir_counts: Counter = Counter(inst["rel_dir"] for inst in instances)
    logger.info("PDFs included per subdirectory (%d dirs):", len(dir_counts))
    for rel_dir, count in sorted(dir_counts.items()):
        logger.info("  %-48s %d PDF(s)", rel_dir + "/", count)
    logger.info("─────────────────────────────────────────────────────────")


def main():
    parser = ConfigArgumentParser(description="Build dataset from annotated PDFs")

    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--input_dir", type=str, default="./data/dev_annotations_cases", help="Root directory of annotations. each leaf directory should contain a set of PDF files and a corresponding .json file for annotations")
    parser.add_argument("--output", type=str, default="dfreddi/multimodal-deid", help="Destination for the dataset: a local directory path (default) or a Hugging Face repo ID (when --push_to_hub is set)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the dataset to the Hugging Face Hub instead of saving locally")
    parser.add_argument("--split_name", type=str, default="base", help="Name of the split to build")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for image resolution in pdf to image conversion")

    args = parser.parse_args()
    logger = init_logger(__name__, level=args.log_level)

    instances = build_instance_list(args.input_dir, logger)
    if not instances:
        logger.error("No valid instances found. Exiting.")
        return

    logger.info(f"Total valid instances: {len(instances)}")
    dataset = build_dataset(instances, args.dpi, logger)
    _log_annotation_stats(dataset, instances, logger)

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            logger.error("HF_TOKEN environment variable is not set — cannot push to Hub.")
            return
        login(token=token)
        DatasetDict({args.split_name: dataset}).push_to_hub(args.output, private=True)
        logger.info(f"Pushed to HF Hub: {args.output} (split: {args.split_name})")
    else:
        out_path = Path(args.output) / args.split_name
        dataset.save_to_disk(str(out_path))
        logger.info(f"Saved locally to: {out_path}")


if __name__ == "__main__":
    main()
