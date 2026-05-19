"""
Render all pages of a dataset split to images with bounding boxes drawn.

Loads from the HF Hub (repo ID) or from a local HF dataset directory.
Each output image is named by its row index (e.g. 0042.png).

Usage:
    python3 src/analysis/render_dataset.py --dataset dfreddi/multimodal-deid
    python3 src/analysis/render_dataset.py --dataset data/test_ds --local --split base
    python3 src/analysis/render_dataset.py --dataset dfreddi/multimodal-deid --limit 50
"""

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _get_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

# ── Label colours (mirrors annotation_app.html LABELS array) ──────────────────
_PARENT_COLORS = {
    "NAME":          "#e74c3c",
    "DATE_OF_BIRTH": "#e67e22",
    "DATETIME":      "#ff9800",
    "AGE":           "#f1c40f",
    "ID":            "#9b59b6",
    "CONTACT":       "#e91e63",
    "ADDRESS":       "#3498db",
}
_FALLBACK_COLOR = "#888888"

def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _color_for(label: str) -> str:
    parent = label.split(":")[0] if ":" in label else label
    return _PARENT_COLORS.get(parent, _FALLBACK_COLOR)


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_row(row: dict, idx: int = 0, total: int = 0) -> Image.Image:
    image = row["image"].copy().convert("RGB")
    draw  = ImageDraw.Draw(image, "RGBA")
    w, h  = image.size

    scale  = max(1.0, w / 1000)
    chip_h = max(14, int(14 * scale))
    font   = _get_font(max(10, int(11 * scale)))
    stroke_w = max(2, int(2 * scale))

    for ann in row["annotations"]:
        label  = ann["label"]
        color  = _color_for(label)
        rgb    = _hex_to_rgb(color)
        fill   = (*rgb, 40)
        stroke = (*rgb, 220)

        for bbox in ann["bboxes"]:
            y_min, x_min, y_max, x_max = bbox
            x0, y0 = int(x_min * w), int(y_min * h)
            x1, y1 = int(x_max * w), int(y_max * h)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=stroke, width=stroke_w)
            tb     = draw.textbbox((0, 0), label, font=font)
            chip_w = tb[2] - tb[0] + int(8 * scale)
            cy     = max(0, y0 - chip_h)
            draw.rectangle([x0, cy, x0 + chip_w, cy + chip_h], fill=(*rgb, 200))
            draw.text((x0 + int(3 * scale), cy + int(2 * scale)), label, fill="white", font=font)

    return image


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from datasets import load_dataset, Dataset

    parser = argparse.ArgumentParser(description="Render dataset pages with bounding boxes.")
    parser.add_argument("--dataset",    required=True,
                        help="HF repo ID (e.g. dfreddi/multimodal-deid) or local dataset root path")
    parser.add_argument("--split",      default="base",
                        help="Dataset split to load (default: base)")
    parser.add_argument("--local",      action="store_true",
                        help="Load from local disk instead of HF Hub")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: data/renders/<split>)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Render only the first N rows")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path("data/renders") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()

    if args.local:
        ds = Dataset.load_from_disk(str(Path(args.dataset) / args.split))
    else:
        token = os.environ.get("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token=token)
        ds = load_dataset(args.dataset, split=args.split)

    total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Rendering {total} images to '{out_dir}' ...")

    for idx in range(total):
        row   = ds[idx]
        src_img = row["image"] if isinstance(row["image"], Image.Image) else Image.fromarray(row["image"])
        dpi     = src_img.info.get("dpi")
        img     = render_row(row, idx, total)
        fname   = out_dir / f"{idx:04d}.png"
        img.save(fname, dpi=dpi) if dpi else img.save(fname)
        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"  {idx + 1}/{total}")

    print(f"Done. Images saved to '{out_dir}'.")


if __name__ == "__main__":
    main()
