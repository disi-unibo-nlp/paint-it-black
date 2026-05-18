"""
Tests for run_inference.py.

Tests cover:
- Dataset loading from local disk (backward compatibility)
- Dataset loading from HuggingFace Hub with --from_hub flag
- HF token authentication for Hub loading
- Public dataset loading without token
- CLI argument parsing
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _white_pil(h=50, w=40):
    """Create a white PIL image."""
    return PILImage.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))


def _make_test_dataset():
    """
    Build a minimal in-memory HF Dataset for testing.

    Returns a dataset with:
    - 2 pages with annotations
    """
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


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_local_dataset_loading(self):
        """Verify local dataset loading works (backward compatibility)."""
        from inference.run_inference import _run

        with patch("inference.run_inference.load_from_disk") as mock_load_disk, \
             patch("inference.run_inference.load_dataset"), \
             patch("inference.run_inference.login"), \
             patch("inference.run_inference.init_logger"), \
             patch("inference.run_inference.set_seed"), \
             patch("inference.run_inference._init_wandb"), \
             patch("inference.run_inference.make_output_dir") as mock_output_dir, \
             patch("inference.run_inference.load_labels") as mock_labels, \
             patch("inference.run_inference.TemplateHandler"), \
             patch("inference.run_inference.LLMClient"), \
             patch("inference.run_inference.compute_metrics") as mock_metrics, \
             patch("inference.run_inference.Path.write_text"):

            # Mock return values
            mock_load_disk.return_value = _make_test_dataset()
            mock_output_dir.return_value = Path("/tmp/test_output")
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_metrics.return_value = {
                "summary": {
                    "detection_micro_f1": 0.9, "detection_macro_f1": 0.9,
                    "avg_e2e_f1": 0.8, "unconditional_mean_iou": 0.6, "mean_iou": 0.75,
                    "char_f1": 0.85, "exact_match_rate": 0.7,
                    "spatial_char_f1": 0.8, "spatial_exact_match_rate": 0.7,
                    "hallucination_rate": 0.1, "miss_rate": 0.15, "format_compliance": 0.95,
                },
                "text_extraction": {
                    "detection": {
                        "micro": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
                        "per_label": {}, "macro_f1": 0.9, "macro_precision": 0.9,
                        "macro_recall": 0.9, "macro_f1_all": 0.9,
                    },
                    "span_exact": {
                        "micro": {"precision": 0.85, "recall": 0.85, "f1": 0.85},
                        "per_label": {}, "macro_f1": 0.85, "macro_precision": 0.85, "macro_recall": 0.85,
                    },
                    "char_f1": 0.85, "edit_distance": 0.1, "exact_match_rate": 0.7,
                },
                "bbox_localization": {
                    "avg_e2e_f1": 0.8, "mean_iou": 0.75, "unconditional_mean_iou": 0.6,
                    "spatial_char_f1": 0.8, "spatial_exact_match_rate": 0.7,
                    "end_to_end": {
                        "@0.5": {"micro": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
                                 "per_label": {}, "macro_f1": 0.8, "macro_precision": 0.8, "macro_recall": 0.8},
                    },
                },
                "hallucination_rate": {"overall": 0.1, "per_label": {}},
                "miss_rate": {"overall": 0.15, "per_label": {}},
                "coarse_f1": {}, "label_confusion": {}, "format_compliance": 0.95,
            }

            # Mock arguments - local loading (from_hub=False)
            args = argparse.Namespace(
                input_dataset="data/test_dataset",
                input_split="base",
                from_hub=False,
                model="test-model",
                template="templates/deid_template.yaml",
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                max_new_tokens=2048,
                timeout=120,
                max_retries=3,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=None,
                enable_thinking=True,
                max_samples=None,
                iou_thresholds=[0.5],
                output_dir="output/inference",
                run_name="test_run",
                log_level="INFO",
                seed=42,
                wandb=False,
                batch_size=8,
                guided_json=False,
            )

            _run(args)

            # Verify load_from_disk was called with correct path
            mock_load_disk.assert_called_once()
            call_args = str(mock_load_disk.call_args)
            assert "data/test_dataset" in call_args
            assert "base" in call_args

    def test_hub_dataset_loading(self):
        """Verify HF Hub dataset loading works with --from_hub flag."""
        from inference.run_inference import _run

        with patch("inference.run_inference.load_from_disk"), \
             patch("inference.run_inference.load_dataset") as mock_load_hub, \
             patch("inference.run_inference.login") as mock_login, \
             patch("inference.run_inference.init_logger"), \
             patch("inference.run_inference.set_seed"), \
             patch("inference.run_inference._init_wandb"), \
             patch("inference.run_inference.make_output_dir") as mock_output_dir, \
             patch("inference.run_inference.load_labels") as mock_labels, \
             patch("inference.run_inference.TemplateHandler"), \
             patch("inference.run_inference.LLMClient"), \
             patch("inference.run_inference.compute_metrics") as mock_metrics, \
             patch("inference.run_inference.Path.write_text"), \
             patch.dict("os.environ", {"HF_TOKEN": "test_token"}):

            # Mock return values
            mock_load_hub.return_value = _make_test_dataset()
            mock_output_dir.return_value = Path("/tmp/test_output")
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_metrics.return_value = {
                "summary": {
                    "detection_micro_f1": 0.9, "detection_macro_f1": 0.9,
                    "avg_e2e_f1": 0.8, "unconditional_mean_iou": 0.6, "mean_iou": 0.75,
                    "char_f1": 0.85, "exact_match_rate": 0.7,
                    "spatial_char_f1": 0.8, "spatial_exact_match_rate": 0.7,
                    "hallucination_rate": 0.1, "miss_rate": 0.15, "format_compliance": 0.95,
                },
                "text_extraction": {
                    "detection": {
                        "micro": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
                        "per_label": {}, "macro_f1": 0.9, "macro_precision": 0.9,
                        "macro_recall": 0.9, "macro_f1_all": 0.9,
                    },
                    "span_exact": {
                        "micro": {"precision": 0.85, "recall": 0.85, "f1": 0.85},
                        "per_label": {}, "macro_f1": 0.85, "macro_precision": 0.85, "macro_recall": 0.85,
                    },
                    "char_f1": 0.85, "edit_distance": 0.1, "exact_match_rate": 0.7,
                },
                "bbox_localization": {
                    "avg_e2e_f1": 0.8, "mean_iou": 0.75, "unconditional_mean_iou": 0.6,
                    "spatial_char_f1": 0.8, "spatial_exact_match_rate": 0.7,
                    "end_to_end": {
                        "@0.5": {"micro": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
                                 "per_label": {}, "macro_f1": 0.8, "macro_precision": 0.8, "macro_recall": 0.8},
                    },
                },
                "hallucination_rate": {"overall": 0.1, "per_label": {}},
                "miss_rate": {"overall": 0.15, "per_label": {}},
                "coarse_f1": {}, "label_confusion": {}, "format_compliance": 0.95,
            }

            # Mock arguments - Hub loading (from_hub=True)
            args = argparse.Namespace(
                input_dataset="dfreddi/multimodal-deid",
                input_split="base",
                from_hub=True,
                model="test-model",
                template="templates/deid_template.yaml",
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                max_new_tokens=2048,
                timeout=120,
                max_retries=3,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=None,
                enable_thinking=True,
                max_samples=None,
                iou_thresholds=[0.5],
                output_dir="output/inference",
                run_name="test_run",
                log_level="INFO",
                seed=42,
                wandb=False,
                batch_size=8,
                guided_json=False,
            )

            _run(args)

            # Verify load_dataset was called with correct arguments
            mock_load_hub.assert_called_once_with("dfreddi/multimodal-deid", split="base")

            # Verify login was called with token
            mock_login.assert_called_once_with(token="test_token")

    def test_hf_token_authentication(self):
        """Verify HF token is used for authentication when provided."""
        from inference.run_inference import _run

        with patch("inference.run_inference.load_from_disk"), \
             patch("inference.run_inference.load_dataset") as mock_load_hub, \
             patch("inference.run_inference.login") as mock_login, \
             patch("inference.run_inference.init_logger"), \
             patch("inference.run_inference.set_seed"), \
             patch("inference.run_inference._init_wandb"), \
             patch("inference.run_inference.make_output_dir") as mock_output_dir, \
             patch("inference.run_inference.load_labels") as mock_labels, \
             patch("inference.run_inference.TemplateHandler"), \
             patch("inference.run_inference.LLMClient"), \
             patch("inference.run_inference.compute_metrics") as mock_metrics, \
             patch("inference.run_inference.Path.write_text"), \
             patch.dict("os.environ", {"HF_TOKEN": "my_secret_token"}):

            mock_load_hub.return_value = _make_test_dataset()
            mock_output_dir.return_value = Path("/tmp/test_output")
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_metrics.return_value = {
                "summary": {
                    "detection_micro_f1": 0.9, "detection_macro_f1": 0.9,
                    "avg_e2e_f1": 0.8, "unconditional_mean_iou": 0.6, "mean_iou": 0.75,
                    "char_f1": 0.85, "exact_match_rate": 0.7,
                    "spatial_char_f1": 0.8, "spatial_exact_match_rate": 0.7,
                    "hallucination_rate": 0.1, "miss_rate": 0.15, "format_compliance": 0.95,
                },
                "text_extraction": {
                    "detection": {
                        "micro": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
                        "per_label": {}, "macro_f1": 0.9, "macro_precision": 0.9,
                        "macro_recall": 0.9, "macro_f1_all": 0.9,
                    },
                    "span_exact": {
                        "micro": {"precision": 0.85, "recall": 0.85, "f1": 0.85},
                        "per_label": {}, "macro_f1": 0.85, "macro_precision": 0.85, "macro_recall": 0.85,
                    },
                    "char_f1": 0.85, "edit_distance": 0.1, "exact_match_rate": 0.7,
                },
                "bbox_localization": {
                    "avg_e2e_f1": 0.8, "mean_iou": 0.75, "unconditional_mean_iou": 0.6,
                    "spatial_char_f1": 0.8, "spatial_exact_match_rate": 0.7,
                    "end_to_end": {
                        "@0.5": {"micro": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
                                 "per_label": {}, "macro_f1": 0.8, "macro_precision": 0.8, "macro_recall": 0.8},
                    },
                },
                "hallucination_rate": {"overall": 0.1, "per_label": {}},
                "miss_rate": {"overall": 0.15, "per_label": {}},
                "coarse_f1": {}, "label_confusion": {}, "format_compliance": 0.95,
            }

            args = argparse.Namespace(
                input_dataset="dfreddi/multimodal-deid",
                input_split="base",
                from_hub=True,
                model="test-model",
                template="templates/deid_template.yaml",
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                max_new_tokens=2048,
                timeout=120,
                max_retries=3,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=None,
                enable_thinking=True,
                max_samples=None,
                iou_thresholds=[0.5],
                output_dir="output/inference",
                run_name="test_run",
                log_level="INFO",
                seed=42,
                wandb=False,
                batch_size=8,
                guided_json=False,
            )

            _run(args)

            # Verify login was called with the correct token
            mock_login.assert_called_once_with(token="my_secret_token")

    def test_no_hf_token_still_works(self):
        """Verify Hub loading works without HF token (for public datasets)."""
        from inference.run_inference import _run

        with patch("inference.run_inference.load_from_disk"), \
             patch("inference.run_inference.load_dataset") as mock_load_hub, \
             patch("inference.run_inference.login") as mock_login, \
             patch("inference.run_inference.init_logger"), \
             patch("inference.run_inference.set_seed"), \
             patch("inference.run_inference._init_wandb"), \
             patch("inference.run_inference.make_output_dir") as mock_output_dir, \
             patch("inference.run_inference.load_labels") as mock_labels, \
             patch("inference.run_inference.TemplateHandler"), \
             patch("inference.run_inference.LLMClient"), \
             patch("inference.run_inference.compute_metrics") as mock_metrics, \
             patch("inference.run_inference.Path.write_text"), \
             patch.dict("os.environ", {}, clear=True):

            mock_load_hub.return_value = _make_test_dataset()
            mock_output_dir.return_value = Path("/tmp/test_output")
            mock_labels.return_value = ["NAME:PATIENT", "DATETIME"]
            mock_metrics.return_value = {
                "summary": {
                    "detection_micro_f1": 0.9, "detection_macro_f1": 0.9,
                    "avg_e2e_f1": 0.8, "unconditional_mean_iou": 0.6, "mean_iou": 0.75,
                    "char_f1": 0.85, "exact_match_rate": 0.7,
                    "spatial_char_f1": 0.8, "spatial_exact_match_rate": 0.7,
                    "hallucination_rate": 0.1, "miss_rate": 0.15, "format_compliance": 0.95,
                },
                "text_extraction": {
                    "detection": {
                        "micro": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
                        "per_label": {}, "macro_f1": 0.9, "macro_precision": 0.9,
                        "macro_recall": 0.9, "macro_f1_all": 0.9,
                    },
                    "span_exact": {
                        "micro": {"precision": 0.85, "recall": 0.85, "f1": 0.85},
                        "per_label": {}, "macro_f1": 0.85, "macro_precision": 0.85, "macro_recall": 0.85,
                    },
                    "char_f1": 0.85, "edit_distance": 0.1, "exact_match_rate": 0.7,
                },
                "bbox_localization": {
                    "avg_e2e_f1": 0.8, "mean_iou": 0.75, "unconditional_mean_iou": 0.6,
                    "spatial_char_f1": 0.8, "spatial_exact_match_rate": 0.7,
                    "end_to_end": {
                        "@0.5": {"micro": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
                                 "per_label": {}, "macro_f1": 0.8, "macro_precision": 0.8, "macro_recall": 0.8},
                    },
                },
                "hallucination_rate": {"overall": 0.1, "per_label": {}},
                "miss_rate": {"overall": 0.15, "per_label": {}},
                "coarse_f1": {}, "label_confusion": {}, "format_compliance": 0.95,
            }

            args = argparse.Namespace(
                input_dataset="dfreddi/multimodal-deid",
                input_split="base",
                from_hub=True,
                model="test-model",
                template="templates/deid_template.yaml",
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                max_new_tokens=2048,
                timeout=120,
                max_retries=3,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=None,
                enable_thinking=True,
                max_samples=None,
                iou_thresholds=[0.5],
                output_dir="output/inference",
                run_name="test_run",
                log_level="INFO",
                seed=42,
                wandb=False,
                batch_size=8,
                guided_json=False,
            )

            _run(args)

            # Verify load_dataset was still called
            mock_load_hub.assert_called_once_with("dfreddi/multimodal-deid", split="base")

            # Verify login was NOT called (no token available)
            mock_login.assert_not_called()


class TestCLIParser:
    """Test CLI argument parsing."""

    def test_from_hub_flag_parsing(self):
        """Verify --from_hub flag is correctly parsed."""
        from inference.run_inference import _build_parser

        parser = _build_parser()

        # Test with --from_hub flag
        args = parser.parse_args([
            "--model", "test-model",
            "--input_dataset", "dfreddi/multimodal-deid",
            "--input_split", "base",
            "--from_hub"
        ])

        assert args.from_hub is True
        assert args.input_dataset == "dfreddi/multimodal-deid"
        assert args.input_split == "base"

    def test_from_hub_default_false(self):
        """Verify --from_hub defaults to False for backward compatibility."""
        from inference.run_inference import _build_parser

        parser = _build_parser()

        # Test without --from_hub flag
        args = parser.parse_args([
            "--model", "test-model",
            "--input_dataset", "data/test_dataset",
            "--input_split", "base"
        ])

        assert args.from_hub is False
        assert args.input_dataset == "data/test_dataset"

    def test_all_dataset_args_parsed(self):
        """Verify all dataset-related arguments are parsed correctly."""
        from inference.run_inference import _build_parser

        parser = _build_parser()

        args = parser.parse_args([
            "--model", "test-model",
            "--input_dataset", "my-org/my-dataset",
            "--input_split", "train",
            "--from_hub",
            "--output_dir", "custom/output"
        ])

        assert args.input_dataset == "my-org/my-dataset"
        assert args.input_split == "train"
        assert args.from_hub is True
        assert args.output_dir == "custom/output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
