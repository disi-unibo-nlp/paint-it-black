import json
import logging
import os
import re
import sys
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import ConfigArgumentParser, init_logger, set_seed, make_output_dir
from core.template_handler import TemplateHandler
from core.llm_client import LLMClient
from core.labels import load_labels
from inference.metrics import compute_metrics

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_output(text: str) -> tuple[list, bool]:
    """Strip markdown fences and parse JSON. Returns (entities, success)."""
    m = _FENCE_RE.search(text)
    cleaned = m.group(1) if m else text.strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result, True
        logger.warning("Model output parsed as JSON but is not a list: %s", type(result))
        return [], False
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse model output as JSON: %s\nOutput: %.200s", exc, text)
        return [], False


def _build_parser() -> ConfigArgumentParser:
    parser = ConfigArgumentParser(
        description="Run multimodal de-identification inference and evaluation.",
        config_dir=Path("./config/inference"),
    )

    # General
    parser.add_argument("--run_name", type=str, default="inference_run")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)

    # Dataset
    parser.add_argument("--input_dataset", type=str, default=None,
                        help="Path to HF dataset root (e.g. data/test_ds)")
    parser.add_argument("--input_split", type=str, default="base")
    parser.add_argument("--output_dir", type=str, default="output/inference")

    # Template
    parser.add_argument("--template", type=str, default="templates/deid_template.yaml")

    # Backend / API
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["openai", "vllm", "anthropic"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)

    # Evaluation
    parser.add_argument("--iou_thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])

    # Tracking
    parser.add_argument("--wandb", action="store_true", default=False)

    return parser


def _run(args) -> None:
    init_logger(args.log_level)
    set_seed(args.seed)
    _init_wandb(args)

    if args.input_dataset is None:
        logger.error("--input_dataset is required.")
        sys.exit(1)
    if args.model is None:
        logger.error("--model is required.")
        sys.exit(1)

    out_dir = make_output_dir(args.run_name, base=args.output_dir)
    logger.info("Output directory: %s", out_dir)

    labels = load_labels()
    handler = TemplateHandler.from_yaml(args.template, labels=labels)
    logger.info("Loaded template from %s", args.template)

    split_path = Path(args.input_dataset) / args.input_split
    ds = load_from_disk(str(split_path))
    logger.info("Loaded dataset split '%s': %d samples", args.input_split, len(ds))

    client = LLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    per_sample_results = []
    all_predictions = []
    all_gt = []
    parse_successes = 0

    for idx, row in enumerate(ds):
        logger.info("Processing sample %d/%d (page %s)", idx + 1, len(ds), row.get("page", "?"))
        try:
            messages = handler.format(page_image=row["image"])
            raw = client.complete(messages)
            entities, success = _parse_json_output(raw)
        except Exception as exc:
            logger.error("Error on sample %d: %r", idx, exc)
            raw, entities, success = "", [], False

        if success:
            parse_successes += 1

        all_predictions.append(entities)
        all_gt.append(row.get("annotations", []))
        per_sample_results.append({
            "idx":          idx,
            "page":         row.get("page"),
            "total_pages":  row.get("total_pages"),
            "doc_type":     row.get("doc_type"),
            "source_pdf":   row.get("source_pdf"),
            "predictions":  entities,
            "raw_output":   raw,
            "parse_success": success,
        })

    format_compliance = parse_successes / len(ds) if len(ds) > 0 else 0.0
    logger.info("Format compliance: %.1f%%", format_compliance * 100)

    metrics = compute_metrics(
        all_predictions=all_predictions,
        all_gt=all_gt,
        iou_thresholds=args.iou_thresholds,
        label_set=labels,
        format_compliance=format_compliance,
    )

    _log_metrics_summary(metrics)
    _log_wandb_metrics(metrics)

    output = {
        "args": vars(args),
        "metrics": metrics,
        "per_sample": per_sample_results,
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", results_path)


def _log_metrics_summary(metrics: dict) -> None:
    det = metrics["entity_detection_f1"]["micro"]
    e2e_05 = metrics["end_to_end_f1"].get(0.5, {}).get("micro", {})
    logger.info("─── Benchmark Results ───────────────────────────────────")
    logger.info("Entity Detection  P=%.3f  R=%.3f  F1=%.3f",
                det["precision"], det["recall"], det["f1"])
    if e2e_05:
        logger.info("End-to-End @0.5   P=%.3f  R=%.3f  F1=%.3f",
                    e2e_05["precision"], e2e_05["recall"], e2e_05["f1"])
    logger.info("Char F1: %.3f  |  Exact Match: %.3f  |  Mean IoU: %.3f",
                metrics["char_f1"], metrics["exact_match_rate"], metrics["mean_iou"])
    logger.info("Hallucination rate: %.3f  |  Miss rate: %.3f",
                metrics["hallucination_rate"]["overall"],
                metrics["miss_rate"]["overall"])
    logger.info("Format compliance: %.3f", metrics["format_compliance"] or 0.0)
    logger.info("─────────────────────────────────────────────────────────")


def _init_wandb(args) -> None:
    if not getattr(args, "wandb", False):
        return
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("wandb enabled but WANDB_API_KEY is not set — skipping wandb init.")
        return
    try:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "multimodal-deid"),
            name=args.run_name,
            config=vars(args),
        )
        logger.info("wandb run initialized: %s", wandb.run.url)
    except Exception as exc:
        logger.warning("wandb init failed: %r — continuing without tracking.", exc)


def _log_wandb_metrics(metrics: dict) -> None:
    try:
        import wandb
        if wandb.run is None:
            return
    except ImportError:
        return

    det = metrics["entity_detection_f1"]
    e2e = metrics["end_to_end_f1"]

    flat = {
        "detection/micro/f1": det["micro"]["f1"],
        "detection/micro/precision": det["micro"]["precision"],
        "detection/micro/recall": det["micro"]["recall"],
        "detection/macro_f1": det["macro_f1"],
        "e2e/f1_at_0.25": e2e.get(0.25, {}).get("micro", {}).get("f1", 0.0),
        "e2e/f1_at_0.5": e2e.get(0.5, {}).get("micro", {}).get("f1", 0.0),
        "e2e/f1_at_0.75": e2e.get(0.75, {}).get("micro", {}).get("f1", 0.0),
        "char_f1": metrics["char_f1"],
        "exact_match_rate": metrics["exact_match_rate"],
        "mean_iou": metrics["mean_iou"],
        "hallucination_rate": metrics["hallucination_rate"]["overall"],
        "miss_rate": metrics["miss_rate"]["overall"],
        "format_compliance": metrics["format_compliance"] or 0.0,
        **{f"per_label/{label}/f1": prf["f1"]
           for label, prf in det["per_label"].items()},
    }
    wandb.log(flat)
    wandb.summary.update(flat)


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    _run(args)
