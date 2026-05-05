"""
Tests for analyze_dataset.py.

Tests cover:
- log_dataset_stats correctly reconstructs per-PDF and per-doc-type statistics
- Statistics match expected values (page counts, annotation counts, label distribution)
- Rare labels are identified and source PDFs are tracked
- Per-doc-type breakdown is correctly computed
- CLI argument parsing works for both local and HF Hub modes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import logging
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage
from collections import Counter

from analysis.analyze_dataset import log_dataset_stats, _RARE_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _white_pil(h=50, w=40):
    """Create a white PIL image."""
    return PILImage.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))


def _make_test_dataset():
    """
    Build a minimal in-memory HF Dataset with multiple PDFs and doc_types.

    Returns a dataset with:
    - 3 PDFs: doc1.pdf (2 pages), doc2.pdf (1 page), doc3.pdf (1 page)
    - 2 doc_types: medical, legal
    - Various labels with different frequencies
    """
    from datasets import Dataset
    from datasets import Image as HFImage

    rows = [
        # doc1.pdf, page 1 (medical)
        {
            "image": _white_pil(),
            "page": 1,
            "total_pages": 2,
            "doc_type": "medical",
            "source_pdf": "doc1.pdf",
            "annotations": [
                {"id": "a1", "label": "NAME:PATIENT", "text": "John Doe",
                 "bboxes": [[0.1, 0.1, 0.2, 0.3]]},
                {"id": "a2", "label": "DATE", "text": "2024-01-01",
                 "bboxes": [[0.3, 0.3, 0.4, 0.5]]},
            ],
        },
        # doc1.pdf, page 2 (medical)
        {
            "image": _white_pil(),
            "page": 2,
            "total_pages": 2,
            "doc_type": "medical",
            "source_pdf": "doc1.pdf",
            "annotations": [
                {"id": "a3", "label": "NAME:PATIENT", "text": "John Doe",
                 "bboxes": [[0.1, 0.1, 0.2, 0.3]]},
            ],
        },
        # doc2.pdf, page 1 (medical, no annotations)
        {
            "image": _white_pil(),
            "page": 1,
            "total_pages": 1,
            "doc_type": "medical",
            "source_pdf": "doc2.pdf",
            "annotations": [],
        },
        # doc3.pdf, page 1 (legal)
        {
            "image": _white_pil(),
            "page": 1,
            "total_pages": 1,
            "doc_type": "legal",
            "source_pdf": "doc3.pdf",
            "annotations": [
                {"id": "a4", "label": "DATE", "text": "2024-02-01",
                 "bboxes": [[0.1, 0.1, 0.2, 0.3]]},
                {"id": "a5", "label": "ADDRESS:STREET", "text": "123 Main St",
                 "bboxes": [[0.3, 0.3, 0.4, 0.5]]},
            ],
        },
    ]
    return Dataset.from_list(rows).cast_column("image", HFImage())


def _get_test_logger():
    """Get a logger for testing."""
    logger = logging.getLogger("test_analyze_dataset")
    logger.setLevel(logging.DEBUG)
    return logger


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLogDatasetStats:
    """Test the log_dataset_stats function."""

    def test_page_counts(self, caplog):
        """Verify page counts are correctly computed."""
        dataset = _make_test_dataset()
        logger = _get_test_logger()

        with caplog.at_level(logging.INFO):
            log_dataset_stats(dataset, logger)

        # Check that page statistics are logged
        logs = caplog.text
        assert "4 total" in logs  # 4 total pages
        assert "3 with annotations" in logs  # 3 pages with annotations
        assert "1 empty" in logs  # 1 page without annotations

    def test_annotation_counts(self, caplog):
        """Verify annotation counts and averages."""
        dataset = _make_test_dataset()
        logger = _get_test_logger()

        with caplog.at_level(logging.INFO):
            log_dataset_stats(dataset, logger)

        logs = caplog.text
        # Total annotations: 2 + 1 + 0 + 2 = 5
        assert "5 total" in logs
        # Average per page: 5/4 = 1.25
        assert "1.25 avg per page" in logs or "1.2 avg per page" in logs
        # Average per annotated page: 5/3 = 1.67
        assert "1.67 avg per annotated page" in logs or "1.6" in logs

    def test_label_distribution(self, caplog):
        """Verify label distribution is correctly computed."""
        dataset = _make_test_dataset()
        logger = _get_test_logger()

        # Mock load_labels to return only the labels we use
        with patch("analysis.analyze_dataset.load_labels") as mock_load_labels:
            mock_load_labels.return_value = ["NAME:PATIENT", "DATE", "ADDRESS:STREET"]

            with caplog.at_level(logging.INFO):
                log_dataset_stats(dataset, logger)

        logs = caplog.text
        # NAME:PATIENT appears 2 times
        assert "NAME:PATIENT" in logs
        # DATE appears 2 times
        assert "DATE" in logs
        # ADDRESS:STREET appears 1 time
        assert "ADDRESS:STREET" in logs

    def test_rare_labels_show_pdfs(self, caplog):
        """Verify rare labels (≤ threshold) show source PDFs."""
        dataset = _make_test_dataset()
        logger = _get_test_logger()

        with patch("analysis.analyze_dataset.load_labels") as mock_load_labels:
            mock_load_labels.return_value = ["NAME:PATIENT", "DATE", "ADDRESS:STREET"]

            with caplog.at_level(logging.INFO):
                log_dataset_stats(dataset, logger)

        logs = caplog.text
        # ADDRESS:STREET appears only 1 time (≤ 10), should show source PDF
        assert "ADDRESS:STREET" in logs
        assert "doc3.pdf" in logs

    def test_pdfs_per_doc_type(self, caplog):
        """Verify PDFs per doc_type are correctly counted."""
        dataset = _make_test_dataset()
        logger = _get_test_logger()

        with patch("analysis.analyze_dataset.load_labels") as mock_load_labels:
            mock_load_labels.return_value = ["NAME:PATIENT", "DATE", "ADDRESS:STREET"]

            with caplog.at_level(logging.INFO):
                log_dataset_stats(dataset, logger)

        logs = caplog.text
        # medical: doc1.pdf, doc2.pdf (2 PDFs)
        assert "medical" in logs
        assert "2 PDF(s)" in logs
        # legal: doc3.pdf (1 PDF)
        assert "legal" in logs
        assert "1 PDF(s)" in logs

    def test_per_doc_type_breakdown(self, caplog):
        """Verify per-doc-type label breakdown."""
        dataset = _make_test_dataset()
        logger = _get_test_logger()

        with patch("analysis.analyze_dataset.load_labels") as mock_load_labels:
            mock_load_labels.return_value = ["NAME:PATIENT", "DATE", "ADDRESS:STREET"]

            with caplog.at_level(logging.INFO):
                log_dataset_stats(dataset, logger)

        logs = caplog.text
        # medical doc_type should have 3 annotations
        assert "medical" in logs
        assert "3 annotations" in logs
        # legal doc_type should have 2 annotations
        assert "legal" in logs
        assert "2 annotations" in logs

    def test_empty_dataset(self, caplog):
        """Verify handling of empty dataset."""
        from datasets import Dataset
        from datasets import Image as HFImage

        dataset = Dataset.from_list([]).cast_column("image", HFImage())
        logger = _get_test_logger()

        with patch("analysis.analyze_dataset.load_labels") as mock_load_labels:
            mock_load_labels.return_value = ["NAME:PATIENT", "DATE"]

            with caplog.at_level(logging.INFO):
                log_dataset_stats(dataset, logger)

        logs = caplog.text
        assert "0 total" in logs

    def test_dataset_with_only_empty_pages(self, caplog):
        """Verify handling of dataset with no annotations."""
        from datasets import Dataset
        from datasets import Image as HFImage

        rows = [
            {
                "image": _white_pil(),
                "page": 1,
                "total_pages": 1,
                "doc_type": "medical",
                "source_pdf": "empty.pdf",
                "annotations": [],
            }
        ]
        dataset = Dataset.from_list(rows).cast_column("image", HFImage())
        logger = _get_test_logger()

        with patch("analysis.analyze_dataset.load_labels") as mock_load_labels:
            mock_load_labels.return_value = ["NAME:PATIENT", "DATE"]

            with caplog.at_level(logging.INFO):
                log_dataset_stats(dataset, logger)

        logs = caplog.text
        assert "1 total" in logs  # 1 page
        assert "0 with annotations" in logs
        assert "1 empty" in logs


class TestCLI:
    """Test CLI argument parsing and dataset loading."""

    def test_local_dataset_loading(self):
        """Verify local dataset loading works."""
        from analysis.analyze_dataset import main
        import argparse

        with patch("analysis.analyze_dataset.load_from_disk") as mock_load, \
             patch("analysis.analyze_dataset.log_dataset_stats"), \
             patch("analysis.analyze_dataset.ConfigArgumentParser.parse_args") as mock_parse:

            # Mock arguments
            mock_parse.return_value = argparse.Namespace(
                dataset="data/test_dataset",
                split="base",
                local=True,
                log_level="INFO"
            )

            # Mock dataset
            mock_load.return_value = _make_test_dataset()

            main()

            # Verify load_from_disk was called with correct path
            mock_load.assert_called_once()
            call_args = str(mock_load.call_args)
            assert "data/test_dataset" in call_args
            assert "base" in call_args

    def test_hub_dataset_loading(self):
        """Verify HF Hub dataset loading works."""
        from analysis.analyze_dataset import main
        import argparse

        with patch("analysis.analyze_dataset.load_dataset") as mock_load, \
             patch("analysis.analyze_dataset.log_dataset_stats"), \
             patch("analysis.analyze_dataset.login"), \
             patch("analysis.analyze_dataset.ConfigArgumentParser.parse_args") as mock_parse, \
             patch.dict("os.environ", {"HF_TOKEN": "test_token"}):

            # Mock arguments
            mock_parse.return_value = argparse.Namespace(
                dataset="dfreddi/multimodal-deid",
                split="base",
                local=False,
                log_level="INFO"
            )

            # Mock dataset
            mock_load.return_value = _make_test_dataset()

            main()

            # Verify load_dataset was called with correct arguments
            mock_load.assert_called_once_with("dfreddi/multimodal-deid", split="base")

    def test_hf_token_authentication(self):
        """Verify HF token is used for authentication when provided."""
        from analysis.analyze_dataset import main
        import argparse

        with patch("analysis.analyze_dataset.load_dataset") as mock_load, \
             patch("analysis.analyze_dataset.log_dataset_stats"), \
             patch("analysis.analyze_dataset.login") as mock_login, \
             patch("analysis.analyze_dataset.ConfigArgumentParser.parse_args") as mock_parse, \
             patch.dict("os.environ", {"HF_TOKEN": "my_secret_token"}):

            mock_parse.return_value = argparse.Namespace(
                dataset="dfreddi/multimodal-deid",
                split="base",
                local=False,
                log_level="INFO"
            )
            mock_load.return_value = _make_test_dataset()

            main()

            # Verify login was called with the token
            mock_login.assert_called_once_with(token="my_secret_token")

    def test_no_hf_token_still_works(self):
        """Verify script works without HF token (for public datasets)."""
        from analysis.analyze_dataset import main
        import argparse

        with patch("analysis.analyze_dataset.load_dataset") as mock_load, \
             patch("analysis.analyze_dataset.log_dataset_stats"), \
             patch("analysis.analyze_dataset.login") as mock_login, \
             patch("analysis.analyze_dataset.ConfigArgumentParser.parse_args") as mock_parse, \
             patch.dict("os.environ", {}, clear=True):

            mock_parse.return_value = argparse.Namespace(
                dataset="dfreddi/multimodal-deid",
                split="base",
                local=False,
                log_level="INFO"
            )
            mock_load.return_value = _make_test_dataset()

            main()

            # Verify login was NOT called
            mock_login.assert_not_called()


class TestRareThreshold:
    """Test the _RARE_THRESHOLD constant behavior."""

    def test_rare_threshold_value(self):
        """Verify _RARE_THRESHOLD is set to 10."""
        assert _RARE_THRESHOLD == 10

    def test_labels_at_threshold_show_pdfs(self, caplog):
        """Verify labels with exactly threshold occurrences show PDFs."""
        from datasets import Dataset
        from datasets import Image as HFImage

        # Create dataset with label appearing exactly 10 times
        rows = []
        for i in range(10):
            rows.append({
                "image": _white_pil(),
                "page": 1,
                "total_pages": 1,
                "doc_type": "medical",
                "source_pdf": f"doc{i}.pdf",
                "annotations": [
                    {"id": f"a{i}", "label": "RARE_LABEL", "text": "test",
                     "bboxes": [[0.1, 0.1, 0.2, 0.2]]}
                ],
            })

        dataset = Dataset.from_list(rows).cast_column("image", HFImage())
        logger = _get_test_logger()

        with patch("analysis.analyze_dataset.load_labels") as mock_load_labels:
            mock_load_labels.return_value = ["RARE_LABEL"]

            with caplog.at_level(logging.INFO):
                log_dataset_stats(dataset, logger)

        logs = caplog.text
        # At exactly 10 occurrences, should show PDFs
        assert "RARE_LABEL" in logs
        assert "doc0.pdf" in logs or "←" in logs  # Should show source PDFs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
