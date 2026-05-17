"""
Tests for analyze_dataset.py.

Tests cover:
- collect_stats correctly reconstructs per-PDF and per-doc-type statistics
- build_report produces a markdown report with correct content
- Rare labels are identified and source PDFs are included
- Per-doc-type breakdown is correctly computed
- CLI argument parsing works for both local and HF Hub modes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import argparse
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage
from collections import Counter

from analysis.analyze_dataset import collect_stats, build_report, _RARE_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _white_pil(h=50, w=40):
    return PILImage.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))


def _make_test_dataset():
    """
    Minimal in-memory HF Dataset with multiple PDFs and doc_types.

    - 3 PDFs: doc1.pdf (2 pages), doc2.pdf (1 page), doc3.pdf (1 page)
    - 2 doc_types: medical, legal
    - Various labels with different frequencies
    """
    from datasets import Dataset
    from datasets import Image as HFImage

    rows = [
        {
            "image": _white_pil(), "page": 1, "total_pages": 2,
            "doc_type": "medical", "source_pdf": "doc1.pdf",
            "annotations": [
                {"id": "a1", "label": "NAME:PATIENT", "text": "John Doe",
                 "bboxes": [[0.1, 0.1, 0.2, 0.3]]},
                {"id": "a2", "label": "DATETIME", "text": "2024-01-01",
                 "bboxes": [[0.3, 0.3, 0.4, 0.5]]},
            ],
        },
        {
            "image": _white_pil(), "page": 2, "total_pages": 2,
            "doc_type": "medical", "source_pdf": "doc1.pdf",
            "annotations": [
                {"id": "a3", "label": "NAME:PATIENT", "text": "John Doe",
                 "bboxes": [[0.1, 0.1, 0.2, 0.3]]},
            ],
        },
        {
            "image": _white_pil(), "page": 1, "total_pages": 1,
            "doc_type": "medical", "source_pdf": "doc2.pdf",
            "annotations": [],
        },
        {
            "image": _white_pil(), "page": 1, "total_pages": 1,
            "doc_type": "legal", "source_pdf": "doc3.pdf",
            "annotations": [
                {"id": "a4", "label": "DATETIME", "text": "2024-02-01",
                 "bboxes": [[0.1, 0.1, 0.2, 0.3]]},
                {"id": "a5", "label": "ADDRESS:STREET", "text": "123 Main St",
                 "bboxes": [[0.3, 0.3, 0.4, 0.5]]},
            ],
        },
    ]
    return Dataset.from_list(rows).cast_column("image", HFImage())


def _run(dataset, labels=None):
    """Collect stats and build a report, returning both."""
    canonical = labels or ["NAME:PATIENT", "DATETIME", "ADDRESS:STREET"]
    with patch("analysis.analyze_dataset.load_labels", return_value=canonical):
        stats  = collect_stats(dataset, canonical)
        report = build_report(stats, "test", "test_dataset")
    return stats, report


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCollectStats:

    def test_page_counts(self):
        stats, report = _run(_make_test_dataset())
        assert stats["total_pages"] == 4
        assert stats["pages_with_annos"] == 3
        assert "Total pages | 4" in report
        assert "Annotated pages | 3" in report
        assert "Empty pages | 1" in report

    def test_annotation_counts(self):
        stats, report = _run(_make_test_dataset())
        assert stats["total_annos"] == 5
        assert "Total annotations | 5" in report
        assert "1.25" in report   # avg per page
        assert "1.67" in report   # avg per annotated page

    def test_label_distribution(self):
        _, report = _run(_make_test_dataset())
        assert "NAME:PATIENT" in report
        assert "DATETIME" in report
        assert "ADDRESS:STREET" in report

    def test_rare_labels_show_pdfs(self):
        _, report = _run(_make_test_dataset())
        # ADDRESS:STREET appears once (≤ threshold) — source PDF must be listed
        assert "ADDRESS:STREET" in report
        assert "doc3.pdf" in report

    def test_pdfs_per_doc_type(self):
        _, report = _run(_make_test_dataset())
        assert "medical" in report
        assert "legal" in report
        assert "2 PDFs" in report   # medical has 2 PDFs
        assert "1 PDFs" in report   # legal has 1 PDF

    def test_per_doc_type_annotation_counts(self):
        _, report = _run(_make_test_dataset())
        assert "3 annotations" in report   # medical
        assert "2 annotations" in report   # legal

    def test_empty_dataset(self):
        from datasets import Dataset
        from datasets import Image as HFImage
        dataset = Dataset.from_list([]).cast_column("image", HFImage())
        stats, report = _run(dataset, ["NAME:PATIENT", "DATETIME"])
        assert stats["total_pages"] == 0
        assert "Total pages | 0" in report

    def test_dataset_with_only_empty_pages(self):
        from datasets import Dataset
        from datasets import Image as HFImage
        rows = [{
            "image": _white_pil(), "page": 1, "total_pages": 1,
            "doc_type": "medical", "source_pdf": "empty.pdf",
            "annotations": [],
        }]
        dataset = Dataset.from_list(rows).cast_column("image", HFImage())
        stats, report = _run(dataset, ["NAME:PATIENT", "DATETIME"])
        assert "Total pages | 1" in report
        assert "Annotated pages | 0" in report
        assert "Empty pages | 1" in report


class TestCLI:

    _EMPTY_STATS = {
        "total_pages": 0, "pdf_names": set(), "total_annos": 0,
        "label_counts": Counter(),
    }

    def _mock_args(self, **kwargs):
        defaults = dict(dataset="data/test", split="base", local=True,
                        log_level="INFO", output_dir="output")
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_local_dataset_loading(self):
        from analysis.analyze_dataset import main
        with patch("analysis.analyze_dataset.load_from_disk") as mock_load, \
             patch("analysis.analyze_dataset.collect_stats", return_value=self._EMPTY_STATS), \
             patch("analysis.analyze_dataset.build_report", return_value=""), \
             patch("analysis.analyze_dataset.load_labels", return_value=[]), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.write_text"), \
             patch("analysis.analyze_dataset.ConfigArgumentParser.parse_args",
                   return_value=self._mock_args(local=True)):
            mock_load.return_value = _make_test_dataset()
            main()
            mock_load.assert_called_once()
            assert "data/test" in str(mock_load.call_args)
            assert "base" in str(mock_load.call_args)

    def test_hub_dataset_loading(self):
        from analysis.analyze_dataset import main
        with patch("analysis.analyze_dataset.load_dataset") as mock_load, \
             patch("analysis.analyze_dataset.collect_stats", return_value=self._EMPTY_STATS), \
             patch("analysis.analyze_dataset.build_report", return_value=""), \
             patch("analysis.analyze_dataset.load_labels", return_value=[]), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.write_text"), \
             patch("analysis.analyze_dataset.login"), \
             patch.dict("os.environ", {"HF_TOKEN": "test_token"}), \
             patch("analysis.analyze_dataset.ConfigArgumentParser.parse_args",
                   return_value=self._mock_args(local=False, dataset="dfreddi/multimodal-deid")):
            mock_load.return_value = _make_test_dataset()
            main()
            mock_load.assert_called_once_with("dfreddi/multimodal-deid", split="base")

    def test_hf_token_authentication(self):
        from analysis.analyze_dataset import main
        with patch("analysis.analyze_dataset.load_dataset", return_value=_make_test_dataset()), \
             patch("analysis.analyze_dataset.collect_stats", return_value=self._EMPTY_STATS), \
             patch("analysis.analyze_dataset.build_report", return_value=""), \
             patch("analysis.analyze_dataset.load_labels", return_value=[]), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.write_text"), \
             patch("analysis.analyze_dataset.login") as mock_login, \
             patch.dict("os.environ", {"HF_TOKEN": "my_secret_token"}), \
             patch("analysis.analyze_dataset.ConfigArgumentParser.parse_args",
                   return_value=self._mock_args(local=False)):
            main()
            mock_login.assert_called_once_with(token="my_secret_token")

    def test_no_hf_token_skips_login(self):
        from analysis.analyze_dataset import main
        with patch("analysis.analyze_dataset.load_dataset", return_value=_make_test_dataset()), \
             patch("analysis.analyze_dataset.collect_stats", return_value=self._EMPTY_STATS), \
             patch("analysis.analyze_dataset.build_report", return_value=""), \
             patch("analysis.analyze_dataset.load_labels", return_value=[]), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.write_text"), \
             patch("analysis.analyze_dataset.login") as mock_login, \
             patch.dict("os.environ", {}, clear=True), \
             patch("analysis.analyze_dataset.ConfigArgumentParser.parse_args",
                   return_value=self._mock_args(local=False)):
            main()
            mock_login.assert_not_called()


class TestRareThreshold:

    def test_rare_threshold_value(self):
        assert _RARE_THRESHOLD == 10

    def test_labels_at_threshold_show_pdfs(self):
        from datasets import Dataset
        from datasets import Image as HFImage
        rows = [
            {
                "image": _white_pil(), "page": 1, "total_pages": 1,
                "doc_type": "medical", "source_pdf": f"doc{i}.pdf",
                "annotations": [
                    {"id": f"a{i}", "label": "RARE_LABEL", "text": "test",
                     "bboxes": [[0.1, 0.1, 0.2, 0.2]]}
                ],
            }
            for i in range(10)
        ]
        dataset = Dataset.from_list(rows).cast_column("image", HFImage())
        _, report = _run(dataset, ["RARE_LABEL"])
        assert "RARE_LABEL" in report
        assert "doc0.pdf" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
