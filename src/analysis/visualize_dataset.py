"""
FiftyOne dataset visualizer for multimodal-deid.

Loads a HuggingFace dataset from disk, converts annotations to FiftyOne
Detections, and launches an interactive browser-based UI.

Run from the repo root:
    python3 src/analysis/visualize_dataset.py
    python3 src/analysis/visualize_dataset.py --dataset_path data/test_ds/base --port 5151

VS Code Remote SSH will automatically detect the open port and offer to forward
it — click "Open in Browser" in the popup, or use the Ports tab in the bottom panel.
"""

import sys
from pathlib import Path

import fiftyone as fo
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.utils import ConfigArgumentParser, init_logger


def build_fo_dataset(hf_dataset: Dataset, name: str, img_dir: Path, logger) -> fo.Dataset:
    """
    Convert a HuggingFace Dataset to a persistent FiftyOne Dataset.

    Images are extracted to img_dir on the first run. On subsequent runs,
    the cached FiftyOne dataset is loaded directly (no re-extraction).
    Use --reload to force a fresh build.
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    fo_dataset = fo.Dataset(name, persistent=True)
    samples = []

    for row in hf_dataset:
        pdf_stem = Path(row["pdf_path"]).stem
        page = row["page"]
        img_path = img_dir / f"{pdf_stem}_page{page:03d}.png"

        if not img_path.exists():
            row["image"].save(str(img_path))

        # Convert annotations: each annotation may have multiple bboxes
        # HF format: [x_min, y_min, x_max, y_max] normalized
        # FiftyOne format: [x, y, width, height] normalized (top-left origin)
        detections = []
        for ann in row["annotations"]:
            for bbox in ann["bboxes"]:
                y_min, x_min, y_max, x_max = bbox
                detections.append(fo.Detection(
                    label=ann["label"],
                    bounding_box=[x_min, y_min, x_max - x_min, y_max - y_min],
                    text=ann["text"],
                    annotation_id=ann["id"],
                ))

        samples.append(fo.Sample(
            filepath=str(img_path),
            page=row["page"],
            pdf_path=row["pdf_path"],
            ground_truth=fo.Detections(detections=detections),
        ))

    fo_dataset.add_samples(samples)
    logger.info(f"Created FiftyOne dataset '{name}' with {len(fo_dataset)} samples")
    return fo_dataset


def main():
    parser = ConfigArgumentParser(description="Visualize a multimodal-deid dataset with FiftyOne")

    parser.add_argument("--dataset_path", type=str, default="data/test_ds/base",
                        help="Path to a dataset saved with Dataset.save_to_disk()")
    parser.add_argument("--name", type=str, default=None,
                        help="FiftyOne dataset name. Defaults to '<parent>_<split>' (e.g. 'test_ds_base')")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Directory to write extracted page images. Defaults to '<dataset_path>/viz_images/'")
    parser.add_argument("--port", type=int, default=5151,
                        help="Port for the FiftyOne web app")
    parser.add_argument("--reload", action="store_true",
                        help="Delete any existing FiftyOne dataset with this name and rebuild from scratch")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    logger = init_logger(__name__, level=args.log_level)

    dataset_path = Path(args.dataset_path)
    name = args.name or f"{dataset_path.parent.name}_{dataset_path.name}"
    img_dir = Path(args.img_dir) if args.img_dir else dataset_path / "viz_images"

    if args.reload and fo.dataset_exists(name):
        fo.delete_dataset(name)
        logger.info(f"Deleted existing FiftyOne dataset '{name}'")

    if fo.dataset_exists(name) and not args.reload:
        logger.info(f"Loading cached FiftyOne dataset '{name}' (use --reload to rebuild)")
        fo_dataset = fo.load_dataset(name)
    else:
        logger.info(f"Loading HuggingFace dataset from '{dataset_path}'")
        hf_dataset = Dataset.load_from_disk(str(dataset_path))
        logger.info(f"Loaded {len(hf_dataset)} rows")
        fo_dataset = build_fo_dataset(hf_dataset, name, img_dir, logger)

    logger.info(f"Launching FiftyOne app on port {args.port}")
    logger.info(f"VS Code: click 'Open in Browser' in the Ports tab, or open http://localhost:{args.port}")

    session = fo.launch_app(fo_dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
