"""
push_to_hub.py — Resize all local dataset splits and push to HuggingFace Hub.

For each split, images wider than target_dpi × page_width_in pixels are downscaled
proportionally using LANCZOS. Images already at or below the target width are unchanged.
This means:
  - base (300 DPI, ~2480 px wide) → downscaled to ~1191 px
  - medium / hard (already ~144 DPI, ~1191 px wide) → untouched

Usage:
    python3 src/dataprep/push_to_hub.py \
        --input_dataset ./data/paint_it_black \
        --hub_dataset   disi-unibo-nlp/paint-it-black \
        --splits base medium hard \
        --target_dpi 144 \
        --private
"""

import os
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from datasets import Image as HFImage
from huggingface_hub import login
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.utils import ConfigArgumentParser, init_logger

logger = None  # initialised in main()


def _resize_to_width(img: Image.Image, target_width: int) -> Image.Image:
    """Resize img (up or down) so its width matches target_width exactly. No-op if already equal."""
    if img.width == target_width:
        return img
    scale = target_width / img.width
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)


def _make_resize_fn(target_width: int):
    def _fn(row):
        row["image"] = _resize_to_width(row["image"], target_width)
        return row
    return _fn


def _load_and_resize(split_path: Path, split: str, target_width: int):
    if not split_path.is_dir():
        logger.error("Split path not found: %s", split_path)
        return None

    logger.info("Loading split '%s' from %s", split, split_path)
    ds = load_from_disk(str(split_path))
    logger.info("  %d rows loaded", len(ds))

    sample_before = ds[0]["image"]
    logger.info("  sample image size before resize: %dx%d", sample_before.width, sample_before.height)

    ds = ds.map(_make_resize_fn(target_width))

    sample_after = ds[0]["image"]
    logger.info("  sample image size after resize:  %dx%d", sample_after.width, sample_after.height)

    return ds


def main():
    parser = ConfigArgumentParser(
        description="Resize local dataset splits and push to HuggingFace Hub.",
        config_dir=Path("./config"),
    )
    parser.add_argument("--input_dataset", type=str, default=None,
                        help="Local dataset root directory (required)")
    parser.add_argument("--hub_dataset", type=str, default=None,
                        help="HuggingFace Hub repo ID (required)")
    parser.add_argument("--splits", type=str, nargs="+", default=["base", "medium", "hard"],
                        help="Splits to upload (default: base medium hard)")
    parser.add_argument("--target_dpi", type=int, default=144,
                        help="Target DPI; images wider than target_dpi × page_width_in are downscaled (default: 144)")
    parser.add_argument("--page_width_in", type=float, default=8.27,
                        help="Page width in inches used to derive the target pixel width (default: 8.27 for A4)")
    parser.add_argument("--private", action="store_true", default=False,
                        help="Push as a private repository")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()

    global logger
    logger = init_logger("push_to_hub", level=args.log_level)

    if args.input_dataset is None:
        logger.error("--input_dataset is required.")
        sys.exit(1)
    if args.hub_dataset is None:
        logger.error("--hub_dataset is required.")
        sys.exit(1)

    target_width = int(args.target_dpi * args.page_width_in)
    logger.info("Target pixel width: %d px  (%.0f DPI × %.2f\")",
                target_width, args.target_dpi, args.page_width_in)

    # Load and resize each split
    splits_dict = {}
    for split in args.splits:
        split_path = Path(args.input_dataset) / split
        ds = _load_and_resize(split_path, split, target_width)
        if ds is None:
            logger.error("Aborting — failed to load split '%s'.", split)
            sys.exit(1)
        splits_dict[split] = ds

    # Login
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN environment variable is not set — cannot push to Hub.")
        sys.exit(1)
    login(token=token)

    # Load existing splits from Hub and merge
    try:
        current = dict(load_dataset(args.hub_dataset))
        logger.info("Loaded existing splits from Hub: %s", list(current.keys()))
    except Exception:
        current = {}

    for split, ds in splits_dict.items():
        current[split] = ds

    # Drop any splits whose schema no longer matches
    ref_features = next(iter(splits_dict.values())).features
    compatible = {k: v for k, v in current.items() if v.features == ref_features}
    stale = set(current) - set(compatible)
    if stale:
        logger.warning("Dropping schema-incompatible splits: %s", sorted(stale))

    logger.info("Pushing splits %s to %s (private=%s) ...",
                sorted(compatible.keys()), args.hub_dataset, args.private)
    DatasetDict(compatible).push_to_hub(args.hub_dataset, private=args.private)
    logger.info("Done. Pushed %d split(s) to %s", len(compatible), args.hub_dataset)


if __name__ == "__main__":
    main()
