"""
Tests for run_ner_baseline.py.

Tests cover:
- Dataset loading from local disk
- Dataset loading from HuggingFace Hub with --from_hub flag
- HF token authentication for Hub loading
- Public dataset loading without token
- CLI argument parsing
- Sentence grouping logic
- Multi-backend dispatch
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import argparse
from unittest.mock import patch, MagicMock
from datasets import Dataset
from datasets import Image as HFImage
import numpy as np
from PIL import Image as PILImage

from inference.run_ner_baseline import _group_sentences, _split_sentences


# ── Helpers ───────────────────────────────────────────────────────────────────

def _white_pil(h=50, w=40):
    return PILImage.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))


def _make_test_dataset():
    rows = [
        {
            "image": _white_pil(),
            "page": 1,
            "total_pages": 1,
            "doc_type": "medical",
            "source_pdf": "doc1.pdf",
            "annotations": [
                {"id": "a1", "label": "NAME:PATIENT", "text": "John Doe",
                 "bboxes": [[0.1, 0.1, 0.2, 0.3]]},
            ],
        },
        {
            "image": _white_pil(),
            "page": 1,
            "total_pages": 1,
            "doc_type": "legal",
            "source_pdf": "doc2.pdf",
            "annotations": [
                {"id": "a2", "label": "DATETIME", "text": "2024-01-01",
                 "bboxes": [[0.3, 0.3, 0.4, 0.5]]},
            ],
        },
    ]
    return Dataset.from_list(rows).cast_column("image", HFImage())


def _mock_metrics():
    return {
        "summary": {
            "span_exact_micro_f1": 0.9,
            "span_exact_macro_f1": 0.85,
            "char_f1": 0.88,
            "exact_match_rate": 0.7,
            "hallucination_rate": 0.1,
            "miss_rate": 0.15,
            "pass_rate": 0.8,
            "avg_e2e_f1": 0.0,
            "unconditional_mean_iou": 0.0,
            "mean_iou": 0.0,
            "spatial_char_f1": 0.0,
            "spatial_exact_match_rate": 0.0,
            "format_compliance": 1.0,
        },
        "text_extraction": {
            "detection": {
                "micro": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
                "per_label": {},
                "macro_f1": 0.85, "macro_precision": 0.85, "macro_recall": 0.85,
                "macro_f1_all": 0.85,
            },
            "span_exact": {
                "micro": {"precision": 0.88, "recall": 0.88, "f1": 0.88,
                           "tp": 8, "fp": 1, "fn": 1},
                "per_label": {},
                "macro_f1": 0.83, "macro_precision": 0.83, "macro_recall": 0.83,
            },
            "char_f1": 0.88,
            "edit_distance": 0.05,
            "exact_match_rate": 0.7,
        },
        "bbox_localization": {},
        "hallucination_rate": {"overall": 0.1, "per_label": {}},
        "miss_rate": {"overall": 0.15, "per_label": {}},
        "coarse_f1": {},
        "label_confusion": {},
        "format_compliance": 1.0,
    }


def _make_args(tmp_path, **overrides):
    defaults = dict(
        input_dataset="data/test_dataset",
        input_split="base",
        from_hub=False,
        backend="gliner",
        model="nvidia/gliner-PII",
        label_map="config/ner_baseline/label_map.yaml",
        threshold=0.5,
        device="cpu",
        group_sentences=False,
        max_chunk_words=100,
        max_samples=None,
        ocr_cache_dir=str(tmp_path / "ocr_cache"),
        output_dir=str(tmp_path),
        run_name="test_run",
        log_level="INFO",
        seed=42,
        wandb=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _gliner_mock_returning(per_page_results):
    """per_page_results: list of sentence-batch results, one entry per page call."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.batch_predict_entities.side_effect = per_page_results
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value = mock_model
    return mock_cls


def _gliner2_mock_returning(per_sentence_results):
    """per_sentence_results: list of entity lists, one per extract_entities call."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.extract_entities.side_effect = per_sentence_results
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value = mock_model
    return mock_cls


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGroupSentences:

    def test_single_sentence_within_limit(self):
        assert _group_sentences(["hello world"], 10) == ["hello world"]

    def test_splits_when_over_limit(self):
        result = _group_sentences(["one two three", "four five six"], 3)
        assert result == ["one two three", "four five six"]

    def test_groups_when_under_limit(self):
        result = _group_sentences(["one two", "three four"], 10)
        assert result == ["one two three four"]

    def test_empty_input(self):
        assert _group_sentences([], 100) == []

    def test_single_sentence_exactly_at_limit(self):
        result = _group_sentences(["one two three"], 3)
        assert result == ["one two three"]

    def test_greedy_fills_first_then_overflows(self):
        # "a b"(2)+"c d"(2)=4>3 → split; "c d"(2)+"e"(1)=3, not >3 → join
        result = _group_sentences(["a b", "c d", "e"], 3)
        assert result == ["a b", "c d e"]

    def test_sentence_exceeding_limit_still_added(self):
        # A single sentence larger than max_words must still be emitted
        result = _group_sentences(["one two three four five"], 3)
        assert result == ["one two three four five"]


class TestOCRCache:

    def test_cache_path_slug(self, tmp_path):
        from inference.run_ner_baseline import _ocr_cache_path
        args = _make_args(tmp_path,
                          input_dataset="disi-unibo-nlp/paint-it-black",
                          input_split="base",
                          max_samples=None)
        p = _ocr_cache_path(args)
        assert "disi-unibo-nlp__paint-it-black" in str(p)
        assert "/base/" in str(p)
        assert "/all/" in str(p)

    def test_cache_path_with_max_samples(self, tmp_path):
        from inference.run_ner_baseline import _ocr_cache_path
        args = _make_args(tmp_path, max_samples=10)
        p = _ocr_cache_path(args)
        assert "/10/" in str(p)

    def test_cache_miss_runs_ocr(self, tmp_path):
        from inference.run_ner_baseline import _run

        with patch("inference.run_ner_baseline.load_from_disk") as mock_disk, \
             patch("inference.run_ner_baseline.load_dataset"), \
             patch("inference.run_ner_baseline.login"), \
             patch("inference.run_ner_baseline.init_logger"), \
             patch("inference.run_ner_baseline.set_seed"), \
             patch("inference.run_ner_baseline._init_wandb"), \
             patch("inference.run_ner_baseline.load_labels") as mock_labels, \
             patch("inference.run_ner_baseline._load_label_map") as mock_lm, \
             patch("inference.run_ner_baseline.run_ocr") as mock_ocr, \
             patch("inference.run_ner_baseline.GLiNER",
                   _gliner_mock_returning([[[]],[[]]])), \
             patch("inference.run_ner_baseline.compute_metrics") as mock_metrics:

            mock_disk.return_value = _make_test_dataset()
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_lm.return_value = {"NAME:PATIENT": "patient name", "DATETIME": "date or time value"}
            mock_ocr.return_value = ["markdown page 1", "markdown page 2"]
            mock_metrics.return_value = _mock_metrics()

            _run(_make_args(tmp_path))

            mock_ocr.assert_called_once()

    def test_cache_hit_skips_ocr(self, tmp_path):
        from inference.run_ner_baseline import _run, _ocr_cache_path, _save_ocr_cache

        args = _make_args(tmp_path)
        cache_path = _ocr_cache_path(args)
        _save_ocr_cache(cache_path,
                        [{"source_pdf": "a.pdf", "page": 1},
                         {"source_pdf": "b.pdf", "page": 1}],
                        ["cached md1", "cached md2"])

        with patch("inference.run_ner_baseline.load_from_disk") as mock_disk, \
             patch("inference.run_ner_baseline.load_dataset"), \
             patch("inference.run_ner_baseline.login"), \
             patch("inference.run_ner_baseline.init_logger"), \
             patch("inference.run_ner_baseline.set_seed"), \
             patch("inference.run_ner_baseline._init_wandb"), \
             patch("inference.run_ner_baseline.load_labels") as mock_labels, \
             patch("inference.run_ner_baseline._load_label_map") as mock_lm, \
             patch("inference.run_ner_baseline.run_ocr") as mock_ocr, \
             patch("inference.run_ner_baseline.GLiNER",
                   _gliner_mock_returning([[[]],[[]]])), \
             patch("inference.run_ner_baseline.compute_metrics") as mock_metrics:

            mock_disk.return_value = _make_test_dataset()
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_lm.return_value = {"NAME:PATIENT": "patient name", "DATETIME": "date or time value"}
            mock_metrics.return_value = _mock_metrics()

            _run(args)

            mock_ocr.assert_not_called()

    def test_cache_roundtrip(self, tmp_path):
        from inference.run_ner_baseline import _save_ocr_cache, _load_ocr_cache
        path = tmp_path / "cache" / "ocr_outputs.jsonl"
        rows = [{"source_pdf": "doc.pdf", "page": 1}, {"source_pdf": "doc.pdf", "page": 2}]
        markdowns = ["# Page 1 content", "# Page 2 content"]
        _save_ocr_cache(path, rows, markdowns)
        loaded = _load_ocr_cache(path)
        assert loaded == markdowns


class TestDatasetLoading:

    def test_local_dataset_loading(self, tmp_path):
        from inference.run_ner_baseline import _run

        with patch("inference.run_ner_baseline.load_from_disk") as mock_disk, \
             patch("inference.run_ner_baseline.load_dataset"), \
             patch("inference.run_ner_baseline.login"), \
             patch("inference.run_ner_baseline.init_logger"), \
             patch("inference.run_ner_baseline.set_seed"), \
             patch("inference.run_ner_baseline._init_wandb"), \
             patch("inference.run_ner_baseline.load_labels") as mock_labels, \
             patch("inference.run_ner_baseline._load_label_map") as mock_lm, \
             patch("inference.run_ner_baseline.run_ocr") as mock_ocr, \
             patch("inference.run_ner_baseline.GLiNER",
                   _gliner_mock_returning([
                       [[{"text": "John Doe", "label": "patient name",
                          "start": 0, "end": 8, "score": 0.9}]],
                       [[]],
                   ])), \
             patch("inference.run_ner_baseline.compute_metrics") as mock_metrics:

            mock_disk.return_value = _make_test_dataset()
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_lm.return_value = {"NAME:PATIENT": "patient name", "DATETIME": "date or time value"}
            mock_ocr.return_value = ["markdown page 1", "markdown page 2"]
            mock_metrics.return_value = _mock_metrics()

            _run(_make_args(tmp_path))

            mock_disk.assert_called_once()
            assert "data/test_dataset" in str(mock_disk.call_args)
            assert "base" in str(mock_disk.call_args)

    def test_hub_dataset_loading(self, tmp_path):
        from inference.run_ner_baseline import _run

        with patch("inference.run_ner_baseline.load_from_disk"), \
             patch("inference.run_ner_baseline.load_dataset") as mock_hub, \
             patch("inference.run_ner_baseline.login") as mock_login, \
             patch("inference.run_ner_baseline.init_logger"), \
             patch("inference.run_ner_baseline.set_seed"), \
             patch("inference.run_ner_baseline._init_wandb"), \
             patch("inference.run_ner_baseline.load_labels") as mock_labels, \
             patch("inference.run_ner_baseline._load_label_map") as mock_lm, \
             patch("inference.run_ner_baseline.run_ocr") as mock_ocr, \
             patch("inference.run_ner_baseline.GLiNER", _gliner_mock_returning([[[]],[[]]])), \
             patch("inference.run_ner_baseline.compute_metrics") as mock_metrics, \
             patch.dict("os.environ", {"HF_TOKEN": "test_token"}):

            mock_hub.return_value = _make_test_dataset()
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_lm.return_value = {"NAME:PATIENT": "patient name", "DATETIME": "date or time value"}
            mock_ocr.return_value = ["markdown page 1", "markdown page 2"]
            mock_metrics.return_value = _mock_metrics()

            _run(_make_args(tmp_path,
                            input_dataset="disi-unibo-nlp/paint-it-black",
                            from_hub=True))

            mock_hub.assert_called_once_with("disi-unibo-nlp/paint-it-black", split="base")
            mock_login.assert_called_once_with(token="test_token")

    def test_hf_token_authentication(self, tmp_path):
        from inference.run_ner_baseline import _run

        with patch("inference.run_ner_baseline.load_from_disk"), \
             patch("inference.run_ner_baseline.load_dataset") as mock_hub, \
             patch("inference.run_ner_baseline.login") as mock_login, \
             patch("inference.run_ner_baseline.init_logger"), \
             patch("inference.run_ner_baseline.set_seed"), \
             patch("inference.run_ner_baseline._init_wandb"), \
             patch("inference.run_ner_baseline.load_labels") as mock_labels, \
             patch("inference.run_ner_baseline._load_label_map") as mock_lm, \
             patch("inference.run_ner_baseline.run_ocr") as mock_ocr, \
             patch("inference.run_ner_baseline.GLiNER", _gliner_mock_returning([[[]],[[]]])), \
             patch("inference.run_ner_baseline.compute_metrics") as mock_metrics, \
             patch.dict("os.environ", {"HF_TOKEN": "my_secret_token"}):

            mock_hub.return_value = _make_test_dataset()
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_lm.return_value = {"NAME:PATIENT": "patient name", "DATETIME": "date or time value"}
            mock_ocr.return_value = ["markdown page 1", "markdown page 2"]
            mock_metrics.return_value = _mock_metrics()

            _run(_make_args(tmp_path,
                            input_dataset="disi-unibo-nlp/paint-it-black",
                            from_hub=True))

            mock_login.assert_called_once_with(token="my_secret_token")

    def test_no_hf_token_still_works(self, tmp_path):
        from inference.run_ner_baseline import _run

        with patch("inference.run_ner_baseline.load_from_disk"), \
             patch("inference.run_ner_baseline.load_dataset") as mock_hub, \
             patch("inference.run_ner_baseline.login") as mock_login, \
             patch("inference.run_ner_baseline.init_logger"), \
             patch("inference.run_ner_baseline.set_seed"), \
             patch("inference.run_ner_baseline._init_wandb"), \
             patch("inference.run_ner_baseline.load_labels") as mock_labels, \
             patch("inference.run_ner_baseline._load_label_map") as mock_lm, \
             patch("inference.run_ner_baseline.run_ocr") as mock_ocr, \
             patch("inference.run_ner_baseline.GLiNER", _gliner_mock_returning([[[]],[[]]])), \
             patch("inference.run_ner_baseline.compute_metrics") as mock_metrics, \
             patch.dict("os.environ", {}, clear=True):

            mock_hub.return_value = _make_test_dataset()
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_lm.return_value = {"NAME:PATIENT": "patient name", "DATETIME": "date or time value"}
            mock_ocr.return_value = ["markdown page 1", "markdown page 2"]
            mock_metrics.return_value = _mock_metrics()

            _run(_make_args(tmp_path,
                            input_dataset="disi-unibo-nlp/paint-it-black",
                            from_hub=True))

            mock_hub.assert_called_once_with("disi-unibo-nlp/paint-it-black", split="base")
            mock_login.assert_not_called()


class TestCLIParser:

    def test_threshold_default(self):
        from inference.run_ner_baseline import _build_parser
        args = _build_parser().parse_args(["--model", "nvidia/gliner-PII",
                                            "--input_dataset", "data/ds",
                                            "--backend", "gliner"])
        assert args.threshold == 0.5

    def test_threshold_override(self):
        from inference.run_ner_baseline import _build_parser
        args = _build_parser().parse_args(["--model", "nvidia/gliner-PII",
                                            "--input_dataset", "data/ds",
                                            "--backend", "gliner",
                                            "--threshold", "0.3"])
        assert args.threshold == 0.3

    def test_from_hub_flag(self):
        from inference.run_ner_baseline import _build_parser
        args = _build_parser().parse_args(["--model", "nvidia/gliner-PII",
                                            "--input_dataset", "disi-unibo-nlp/paint-it-black",
                                            "--backend", "gliner",
                                            "--from_hub"])
        assert args.from_hub is True
        assert args.input_dataset == "disi-unibo-nlp/paint-it-black"

    def test_from_hub_default_false(self):
        from inference.run_ner_baseline import _build_parser
        args = _build_parser().parse_args(["--model", "nvidia/gliner-PII",
                                            "--input_dataset", "data/ds",
                                            "--backend", "gliner"])
        assert args.from_hub is False

    def test_model_and_device_parsed(self):
        from inference.run_ner_baseline import _build_parser
        args = _build_parser().parse_args(["--model", "knowledgator/gliner-pii-large-v1.0",
                                            "--input_dataset", "data/ds",
                                            "--backend", "gliner",
                                            "--device", "cpu"])
        assert args.model == "knowledgator/gliner-pii-large-v1.0"
        assert args.device == "cpu"

    def test_backend_parsed(self):
        from inference.run_ner_baseline import _build_parser
        args = _build_parser().parse_args(["--model", "nvidia/gliner-PII",
                                            "--input_dataset", "data/ds",
                                            "--backend", "gliner"])
        assert args.backend == "gliner"

    def test_group_sentences_default_false(self):
        from inference.run_ner_baseline import _build_parser
        args = _build_parser().parse_args(["--model", "nvidia/gliner-PII",
                                            "--input_dataset", "data/ds",
                                            "--backend", "gliner"])
        assert args.group_sentences is False
        assert args.max_chunk_words == 100

    def test_group_sentences_flag(self):
        from inference.run_ner_baseline import _build_parser
        args = _build_parser().parse_args(["--model", "nvidia/gliner-PII",
                                            "--input_dataset", "data/ds",
                                            "--backend", "gliner",
                                            "--group_sentences",
                                            "--max_chunk_words", "50"])
        assert args.group_sentences is True
        assert args.max_chunk_words == 50


class TestBackends:

    def test_gliner2_backend_dispatched(self, tmp_path):
        from inference.run_ner_baseline import _run

        # GLiNER2 returns {"entities": {"label_str": [{"text": ..., "confidence": ...}]}}
        gliner2_cls = _gliner2_mock_returning([
            {"entities": {"patient name": [{"text": "John Doe", "confidence": 0.95}]}},
            {"entities": {}},
        ])

        with patch("inference.run_ner_baseline.load_from_disk") as mock_disk, \
             patch("inference.run_ner_baseline.load_dataset"), \
             patch("inference.run_ner_baseline.login"), \
             patch("inference.run_ner_baseline.init_logger"), \
             patch("inference.run_ner_baseline.set_seed"), \
             patch("inference.run_ner_baseline._init_wandb"), \
             patch("inference.run_ner_baseline.load_labels") as mock_labels, \
             patch("inference.run_ner_baseline._load_label_map") as mock_lm, \
             patch("inference.run_ner_baseline.run_ocr") as mock_ocr, \
             patch("inference.run_ner_baseline.GLiNER2", gliner2_cls), \
             patch("inference.run_ner_baseline.compute_metrics") as mock_metrics:

            mock_disk.return_value = _make_test_dataset()
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_lm.return_value = {"NAME:PATIENT": "patient name", "DATETIME": "date or time value"}
            mock_ocr.return_value = ["markdown page 1", "markdown page 2"]
            mock_metrics.return_value = _mock_metrics()

            _run(_make_args(tmp_path, backend="gliner2",
                            model="fastino/gliner2-privacy-filter-PII-multi"))

            model_instance = gliner2_cls.from_pretrained.return_value.to.return_value
            assert model_instance.extract_entities.call_count >= 1

            results_path = next(tmp_path.rglob("results.json"))
            import json
            result = json.loads(results_path.read_text())
            first_preds = result["per_sample"][0]["predictions"]
            assert any(p["label"] == "NAME:PATIENT" for p in first_preds)

    def test_openbioner_backend_dispatched(self, tmp_path):
        from inference.run_ner_baseline import _run

        # Create dataset BEFORE patching sys.modules to avoid datasets' dill
        # fingerprinting calling issubclass(obj, spacy.Language) on a MagicMock.
        dataset = _make_test_dataset()

        # Patch _predict_openbioner directly to avoid spaCy/zshot mock complexity.
        # The key things to verify: backend dispatches to _predict_openbioner and
        # load_label_guidelines is called to build entity descriptions.
        with patch("inference.run_ner_baseline.load_from_disk") as mock_disk, \
             patch("inference.run_ner_baseline.load_dataset"), \
             patch("inference.run_ner_baseline.login"), \
             patch("inference.run_ner_baseline.init_logger"), \
             patch("inference.run_ner_baseline.set_seed"), \
             patch("inference.run_ner_baseline._init_wandb"), \
             patch("inference.run_ner_baseline.load_labels") as mock_labels, \
             patch("inference.run_ner_baseline.load_label_guidelines") as mock_guidelines, \
             patch("inference.run_ner_baseline._load_label_map") as mock_lm, \
             patch("inference.run_ner_baseline.run_ocr") as mock_ocr, \
             patch("inference.run_ner_baseline._predict_openbioner", return_value=[]) as mock_predict, \
             patch("inference.run_ner_baseline.compute_metrics") as mock_metrics, \
             patch.dict("sys.modules", {
                 "spacy": MagicMock(),
                 "zshot": MagicMock(),
                 "zshot.utils": MagicMock(),
                 "zshot.utils.data_models": MagicMock(),
                 "zshot.linker": MagicMock(),
             }):

            mock_disk.return_value = dataset
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_lm.return_value = {"NAME:PATIENT": "patient name", "DATETIME": "date or time value"}
            mock_guidelines.return_value = [
                ("NAME:PATIENT", "The name of the patient"),
                ("DATETIME", "A date or time value"),
            ]
            mock_ocr.return_value = ["markdown page 1", "markdown page 2"]
            mock_metrics.return_value = _mock_metrics()

            _run(_make_args(tmp_path, backend="openbioner",
                            model="disi-unibo-nlp/openbioner-base-v2-deid"))

            mock_guidelines.assert_called_once()
            assert mock_predict.call_count == 2  # once per page


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
