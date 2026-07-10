import json
import logging
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

import datasets as _ds
_ds.logging.set_verbosity_error()
_ds.disable_progress_bar()
import huggingface_hub as _hfhub
_hfhub.utils.disable_progress_bars()
import logging as _logging
_logging.getLogger("docling").setLevel(_logging.WARNING)
_logging.getLogger("docling_core").setLevel(_logging.WARNING)

from datasets import load_from_disk, load_dataset
from huggingface_hub import login

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

try:
    from gliner2 import GLiNER2
except ImportError:
    GLiNER2 = None

from collections import defaultdict

from core.utils import ConfigArgumentParser, init_logger, set_seed, make_output_dir
from core.labels import load_labels, load_label_guidelines
from inference.metrics import compute_metrics, _prf
from inference.ocr import run_ocr

logger = logging.getLogger(__name__)


def _rate(num: int, den: int) -> float:
    return num / den if den > 0 else 0.0


def _patch_tokenizer_compat() -> None:
    """Patch transformers to accept extra_special_tokens as a list (old tokenizer config format).

    transformers >= 4.46 changed _set_model_specific_special_tokens to expect a dict,
    but some older model configs still store it as a list. This makes loading idempotent
    and safe to call multiple times.
    """
    import transformers
    orig = transformers.PreTrainedTokenizerBase._set_model_specific_special_tokens
    if getattr(orig, "_compat_patched", False):
        return
    def _patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        return orig(self, special_tokens)
    _patched._compat_patched = True
    transformers.PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched


def _load_label_map(path: str) -> dict[str, str]:
    with open(path) as f:
        return yaml.safe_load(f)


def _ocr_cache_path(args) -> Path:
    dataset_slug = args.input_dataset.replace("/", "__")
    samples_slug = str(args.max_samples) if args.max_samples is not None else "all"
    return Path(args.ocr_cache_dir) / dataset_slug / args.input_split / samples_slug / "ocr_outputs.jsonl"


def _load_ocr_cache(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    return [json.loads(line)["markdown"] for line in path.read_text().splitlines() if line.strip()]


def _save_ocr_cache(path: Path, rows: list, markdowns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, (row, md) in enumerate(zip(rows, markdowns)):
            f.write(json.dumps({
                "idx": i,
                "source_pdf": row.get("source_pdf"),
                "page": row.get("page"),
                "markdown": md,
            }) + "\n")


def _split_sentences(text: str, nlp) -> list[str]:
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


def _group_sentences(sentences: list[str], max_words: int) -> list[str]:
    groups, current, current_words = [], [], 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and current_words + n > max_words:
            groups.append(" ".join(current))
            current, current_words = [sentence], n
        else:
            current.append(sentence)
            current_words += n
    if current:
        groups.append(" ".join(current))
    return groups


def _predict_gliner(model, chunks, gliner_labels, threshold, reverse_map):
    preds = []
    if not chunks:
        return preds
    raw_batch = model.batch_predict_entities(chunks, gliner_labels, threshold=threshold)
    for raw in raw_batch:
        for ent in raw:
            gt_label = reverse_map.get(ent["label"])
            if gt_label:
                preds.append({"label": gt_label, "text": ent["text"], "score": ent["score"]})
    return preds


def _predict_gliner2(model, chunks, gliner_labels, threshold, reverse_map):
    preds = []
    for chunk in chunks:
        result = model.extract_entities(chunk, entity_types=gliner_labels,
                                        threshold=threshold, include_confidence=True)
        for label_str, entities in result.get("entities", {}).items():
            gt_label = reverse_map.get(label_str)
            if gt_label is None:
                continue
            for ent in entities:
                preds.append({"label": gt_label, "text": ent["text"],
                              "score": ent.get("confidence", 1.0)})
    return preds


def _predict_openbioner(nlp_zshot, chunks, reverse_map):
    preds = []
    for chunk in chunks:
        for ent in nlp_zshot(chunk).ents:
            gt_label = reverse_map.get(ent.label_)
            if gt_label:
                preds.append({"label": gt_label, "text": ent.text, "score": 1.0})
    return preds


def _build_parser() -> ConfigArgumentParser:
    parser = ConfigArgumentParser(
        description="Run OCR + NER baseline (gliner / gliner2 / openbioner) for de-identification evaluation.",
        config_dir=Path("./config/ner_baseline"),
    )

    # General
    parser.add_argument("--run_name", type=str, default="ner_baseline_run")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)

    # Dataset
    parser.add_argument("--input_dataset", type=str, default=None,
                        help="HF repo ID (with --from_hub) or local path to HF dataset root")
    parser.add_argument("--input_split", type=str, default="base")
    parser.add_argument("--from_hub", action="store_true", default=False,
                        help="Load dataset from HuggingFace Hub instead of local disk")
    parser.add_argument("--output_dir", type=str, default="output/ner_baseline")

    # Backend
    parser.add_argument("--backend", type=str, choices=["gliner", "gliner2", "openbioner"],
                        help="NER backend to use")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (HuggingFace repo or local path)")
    parser.add_argument("--label_map", type=str, default="config/ner_baseline/label_map.yaml",
                        help="Path to label_map YAML (GT label → NER label string)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for entity detection (gliner / gliner2 only)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model: 'cuda' or 'cpu'")

    # Chunking
    parser.add_argument("--group_sentences", action="store_true", default=False,
                        help="Group sentences into chunks before inference (default: one sentence per call)")
    parser.add_argument("--max_chunk_words", type=int, default=100,
                        help="Max words per chunk when --group_sentences is set")

    # Dataset cap
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap the number of samples processed (useful for quick checks)")

    # OCR cache
    parser.add_argument("--ocr_cache_dir", type=str, default="data/ocr_cache",
                        help="Shared OCR cache directory (keyed by dataset+split+max_samples)")

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
    if args.backend is None:
        logger.error("--backend is required (gliner / gliner2 / openbioner).")
        sys.exit(1)

    out_dir = make_output_dir(args.run_name, base=args.output_dir)
    logger.info("Output directory: %s", out_dir)

    # Label map: GT label → NER label string (forward), NER label string → GT label (reverse)
    forward_map = _load_label_map(args.label_map)
    reverse_map = {v: k for k, v in forward_map.items()}
    if len(reverse_map) != len(forward_map):
        logger.error("label_map is not injective — duplicate NER label strings detected.")
        sys.exit(1)
    gliner_labels = list(forward_map.values())
    logger.info("Loaded label map from %s (%d labels)", args.label_map, len(forward_map))

    labels = load_labels()

    # Optional HF authentication
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    # Load dataset
    if args.from_hub:
        logger.info("Loading dataset '%s' (split: %s) from HuggingFace Hub...",
                    args.input_dataset, args.input_split)
        ds = load_dataset(args.input_dataset, split=args.input_split)
    else:
        split_path = Path(args.input_dataset) / args.input_split
        logger.info("Loading dataset from local path: %s", split_path)
        ds = load_from_disk(str(split_path))

    logger.info("Loaded dataset split '%s': %d samples", args.input_split, len(ds))

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        logger.info("Capped to %d samples (--max_samples)", len(ds))

    rows = list(ds)
    n = len(rows)

    # ── OCR phase ─────────────────────────────────────────────────────────────
    cache_path = _ocr_cache_path(args)
    markdowns = _load_ocr_cache(cache_path)
    if markdowns is not None:
        logger.info("OCR cache hit (%d pages): %s", len(markdowns), cache_path)
    else:
        logger.info("OCR cache miss — running Docling OCR...")
        markdowns = run_ocr(ds, logger=logger)
        _save_ocr_cache(cache_path, rows, markdowns)
        logger.info("OCR results cached to %s", cache_path)

    # ── Sentence splitter (shared across all backends) ────────────────────────
    import spacy
    nlp_sent = spacy.blank("en")
    nlp_sent.add_pipe("sentencizer")

    # ── Model init (backend-specific) ─────────────────────────────────────────
    if args.backend == "gliner":
        if GLiNER is None:
            logger.error("gliner package not installed. Run: pip install gliner")
            sys.exit(1)
        logger.info("Loading GLiNER model: %s", args.model)
        model = GLiNER.from_pretrained(args.model).to(args.device)

    elif args.backend == "gliner2":
        if GLiNER2 is None:
            logger.error("gliner2 package not installed. Run: pip install gliner2")
            sys.exit(1)
        _patch_tokenizer_compat()
        logger.info("Loading GLiNER2 model: %s", args.model)
        model = GLiNER2.from_pretrained(args.model).to(args.device)

    elif args.backend == "openbioner":
        import spacy as _spacy_zshot
        import zshot
        from zshot.utils.data_models import Entity
        from zshot.linker import LinkerSMXM
        guidelines = dict(load_label_guidelines())
        entities = [
            Entity(name=forward_map[gt], description=guidelines[gt])
            for gt in forward_map
        ]
        model = _spacy_zshot.blank("en")
        model.add_pipe("zshot", config=zshot.PipelineConfig(
            linker=LinkerSMXM(model_name=args.model),
            entities=entities,
            device=args.device,
        ))
        logger.info("OpenBioNER pipeline ready: %s (%d entities)", args.model, len(entities))

    chunk_mode = "grouped (max %d words)" % args.max_chunk_words if args.group_sentences else "sentence-level"
    logger.info("Backend '%s' ready — chunking: %s", args.backend, chunk_mode)

    # ── NER inference ─────────────────────────────────────────────────────────
    all_predictions = []
    all_gt = []

    for markdown, row in tqdm(zip(markdowns, rows), desc="NER inference", total=n, unit="page"):
        sentences = _split_sentences(markdown, nlp_sent)
        chunks = _group_sentences(sentences, args.max_chunk_words) if args.group_sentences else sentences

        if args.backend == "gliner":
            preds = _predict_gliner(model, chunks, gliner_labels, args.threshold, reverse_map)
        elif args.backend == "gliner2":
            preds = _predict_gliner2(model, chunks, gliner_labels, args.threshold, reverse_map)
        else:  # openbioner
            preds = _predict_openbioner(model, chunks, reverse_map)

        all_predictions.append(preds)
        all_gt.append(row.get("annotations", []))

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(
        all_predictions=all_predictions,
        all_gt=all_gt,
        label_set=labels,
        format_compliance=1.0,
    )

    ex       = metrics["text_extraction"]["span_exact"]
    ex_micro = ex["micro"]

    hall_overall = _rate(ex_micro["fp"], ex_micro["tp"] + ex_micro["fp"])
    miss_overall = _rate(ex_micro["fn"], ex_micro["tp"] + ex_micro["fn"])

    hall_per_label, miss_per_label = {}, {}
    for label, pl in ex["per_label"].items():
        hall_per_label[label] = _rate(pl["fp"], pl["tp"] + pl["fp"])
        miss_per_label[label] = _rate(pl["fn"], pl["tp"] + pl["fn"])

    coarse: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for label, pl in ex["per_label"].items():
        parent = label.split(":")[0] if ":" in label else label
        for k in ("tp", "fp", "fn"):
            coarse[parent][k] += pl[k]
    span_coarse_prf = {c: _prf(v["tp"], v["fp"], v["fn"]) for c, v in coarse.items()}

    metrics_out = {
        "summary": {
            "span_exact_micro_f1": metrics["summary"]["span_exact_micro_f1"],
            "span_exact_macro_f1": metrics["summary"]["span_exact_macro_f1"],
            "hallucination_rate":  hall_overall,
            "miss_rate":           miss_overall,
        },
        "span_exact":      ex,
        "hallucination_rate": {"overall": hall_overall, "per_label": hall_per_label},
        "miss_rate":          {"overall": miss_overall, "per_label": miss_per_label},
        "coarse_f1":          span_coarse_prf,
    }

    _log_metrics_summary(metrics_out)
    _log_wandb_metrics(metrics_out)

    per_sample = []
    for i, (row, preds, md) in enumerate(zip(rows, all_predictions, markdowns)):
        per_sample.append({
            "idx": i,
            "page": row.get("page"),
            "total_pages": row.get("total_pages"),
            "doc_type": row.get("doc_type"),
            "source_pdf": row.get("source_pdf"),
            "annotations": row.get("annotations", []),
            "predictions": preds,
            "markdown": md,
        })

    output = {
        "args": vars(args),
        "metrics": metrics_out,
        "per_sample": per_sample,
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", results_path)


def _log_metrics_summary(metrics: dict) -> None:
    ex = metrics["span_exact"]
    s  = metrics["summary"]
    logger.info("─── NER Baseline Results ─────────────────────────────────")
    logger.info("Span exact  micro  P=%.3f  R=%.3f  F1=%.3f",
                ex["micro"]["precision"], ex["micro"]["recall"], ex["micro"]["f1"])
    logger.info("Span exact  macro  P=%.3f  R=%.3f  F1=%.3f",
                ex["macro_precision"], ex["macro_recall"], ex["macro_f1"])
    logger.info("Hallucination rate: %.3f  |  Miss rate: %.3f",
                s["hallucination_rate"], s["miss_rate"])
    logger.info("──────────────────────────────────────────────────────────")


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

    ex = metrics["span_exact"]
    s  = metrics["summary"]
    flat = {
        "span_exact/micro/f1":        ex["micro"]["f1"],
        "span_exact/micro/precision":  ex["micro"]["precision"],
        "span_exact/micro/recall":     ex["micro"]["recall"],
        "span_exact/macro_f1":        ex["macro_f1"],
        "hallucination_rate":         s["hallucination_rate"],
        "miss_rate":                  s["miss_rate"],
        **{f"per_label/{label}/f1": prf["f1"]
           for label, prf in ex["per_label"].items()},
    }
    wandb.log(flat)
    wandb.summary.update(flat)


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    _run(args)
