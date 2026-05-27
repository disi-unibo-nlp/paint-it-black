"""
Render inference predictions alongside ground-truth annotations.

Reads a results.json produced by run_inference.py and re-loads the original
dataset to obtain input images. For each sample it draws:
  - Ground-truth entities  (green border + label above the box)
  - Predicted entities     (red   border + label below the box)

Output images are written to <results_dir>/renders/NNNN.png.
Existing renders are cleared before writing.

Usage:
    python3 src/inference/render_predictions.py --results output/inference/my_run/results.json
    python3 src/inference/render_predictions.py --results output/inference/my_run/results.json --limit 20
    python3 src/inference/render_predictions.py --results output/inference/my_run/results.json \\
        --dataset data/test_ds --split base --local
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Colour constants ───────────────────────────────────────────────────────────

_GT_COLOR   = (34,  197,  94)   # green-500
_PRED_COLOR = (239,  68,  68)   # red-500
_MISS_COLOR = (251, 191,  37)   # amber-400  (GT not matched by any pred)


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ── Core rendering ─────────────────────────────────────────────────────────────

def _draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: list,
    w: int,
    h: int,
    rgb: tuple,
    label: str,
    label_above: bool,
    scale: float = 1.0,
    font=None,
) -> None:
    """Draw a single bbox with a label chip. bbox = [y_min, x_min, y_max, x_max] in [0,1]."""
    y_min, x_min, y_max, x_max = bbox
    x0 = int(x_min * w)
    y0 = int(y_min * h)
    x1 = int(x_max * w)
    y1 = int(y_max * h)
    x0, x1 = sorted([max(0, x0), min(w - 1, x1)])
    y0, y1 = sorted([max(0, y0), min(h - 1, y1)])

    fill     = (*rgb, 35)
    stroke   = (*rgb, 200)
    stroke_w = max(2, int(2 * scale))
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=stroke, width=stroke_w)

    chip_h = max(14, int(14 * scale))
    tb     = draw.textbbox((0, 0), label, font=font)
    chip_w = tb[2] - tb[0] + int(8 * scale)
    if label_above:
        cy = max(0, y0 - chip_h)
    else:
        cy = min(h - chip_h, y1)

    draw.rectangle([x0, cy, x0 + chip_w, cy + chip_h], fill=(*rgb, 200))
    draw.text((x0 + int(3 * scale), cy + int(2 * scale)), label, fill="white", font=font)


def render_sample(
    image: Image.Image,
    gt_entities: list,
    pred_entities: list,
    sample_meta: dict,
) -> Image.Image:
    """
    Render a single sample image with GT (green) and predicted (red) bboxes.

    Args:
        image:        PIL Image of the document page.
        gt_entities:  Ground-truth annotation list (from dataset["annotations"]).
        pred_entities: Predicted entity list (from results["per_sample"][idx]["predictions"]).
        sample_meta:  Dict with keys idx, page, total_pages, doc_type, source_pdf, parse_success.

    Returns:
        Annotated PIL Image.
    """
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    scale = max(1.0, w / 1000)
    font  = _get_font(max(10, int(11 * scale)))

    for ann in gt_entities:
        label = ann.get("label", "?")
        for bbox in ann.get("bboxes", []):
            _draw_box(draw, bbox, w, h, _GT_COLOR, label, label_above=True, scale=scale, font=font)

    for ent in pred_entities:
        label = ent.get("label", "?")
        for bbox in ent.get("bbox_2d", []):
            _draw_box(draw, bbox, w, h, _PRED_COLOR, label, label_above=False, scale=scale, font=font)

    return img


# ── CLI ────────────────────────────────────────────────────────────────────────

def _load_dataset(args_from_results: dict, override_dataset=None, override_split=None, override_local=None):
    from datasets import load_dataset, Dataset

    dataset_path = override_dataset or args_from_results.get("input_dataset")
    split        = override_split   or args_from_results.get("input_split", "base")
    from_hub     = args_from_results.get("from_hub", False) if override_local is None else (not override_local)

    if not dataset_path:
        raise ValueError("Cannot determine dataset path — pass --dataset explicitly.")

    if from_hub:
        token = os.environ.get("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token=token)
        return load_dataset(dataset_path, split=split)
    else:
        from datasets import load_from_disk
        split_path = Path(dataset_path) / split
        return load_from_disk(str(split_path))


def main():
    parser = argparse.ArgumentParser(
        description="Render inference predictions vs. ground truth for a results.json."
    )
    parser.add_argument("--results",   required=True,
                        help="Path to results.json from run_inference.py")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Render only the first N samples (default: all)")
    parser.add_argument("--dataset",   default=None,
                        help="Override dataset path/repo (inferred from results.json by default)")
    parser.add_argument("--split",     default=None,
                        help="Override dataset split (inferred from results.json by default)")
    parser.add_argument("--local",     action="store_true", default=False,
                        help="Force local disk loading (override from_hub in results.json)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for renders (default: <results_dir>/renders)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    results = json.loads(results_path.read_text())
    per_sample  = results["per_sample"]
    saved_args  = results.get("args", {})

    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()

    print(f"Loading dataset...")
    ds = _load_dataset(
        saved_args,
        override_dataset=args.dataset,
        override_split=args.split,
        override_local=args.local if args.local else None,
    )

    limit = min(args.limit, len(per_sample)) if args.limit else len(per_sample)
    print(f"Rendering {limit}/{len(per_sample)} samples to '{out_dir}' ...")

    raw_lines = []
    for i in range(limit):
        sample  = per_sample[i]
        idx     = sample["idx"]
        row     = ds[idx]
        image   = row["image"] if isinstance(row["image"], Image.Image) else Image.fromarray(row["image"])
        gt      = row.get("annotations", [])
        preds   = sample.get("predictions", [])

        dpi      = image.info.get("dpi")
        rendered = render_sample(image, gt, preds, sample)
        fname    = out_dir / f"{idx:04d}.png"
        rendered.save(fname, dpi=dpi) if dpi else rendered.save(fname)

        # Accumulate raw output entry
        source   = sample.get("source_pdf", "?")
        page     = sample.get("page", "?")
        total_p  = sample.get("total_pages", "?")
        dtype    = sample.get("doc_type", "?")
        status   = "OK" if sample.get("parse_success") else "PARSE FAIL"
        raw_lines.append(
            f"=== [{idx:04d}] {source}  p.{page}/{total_p}  ({dtype})  [{status}] ===\n"
            f"{sample.get('raw_output', '').strip()}\n"
        )

        if (i + 1) % 10 == 0 or i + 1 == limit:
            print(f"  {i + 1}/{limit}")

    raw_path = out_dir / "raw_outputs.txt"
    raw_path.write_text("\n".join(raw_lines))
    print(f"Done. Renders saved to '{out_dir}'.")
    print(f"Raw outputs:  {raw_path}")


if __name__ == "__main__":
    main()
