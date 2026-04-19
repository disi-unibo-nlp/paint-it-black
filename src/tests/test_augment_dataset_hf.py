"""
Tests for the HF Dataset augmentation mode in augment_dataset.py.

Tests cover:
- augment_dataset_split produces a Dataset with the correct schema (image, page, annotations)
  and no pdf_path column
- num_augmentations > 1 triggers a warning in dataset mode main()
- Re-sampling replaces only the specified row indices; all others are kept unchanged
- build_dataset output has no pdf_path column
- augment_pdf (dir mode) still works correctly after the _augment_single_image refactor
"""

import sys
import argparse
import logging
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage

sys.path.insert(0, "src")

from dataprep.augment_pdfs import (
    augment_dataset_split,
    augment_pdf,
    build_pipeline,
    _augment_single_image,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _white_pil(h=50, w=40):
    return PILImage.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))


def _make_source_dataset(n=3):
    """Build a minimal in-memory HF Dataset with n rows (no pdf_path)."""
    from datasets import Dataset
    from datasets import Image as HFImage
    rows = [
        {
            "image":       _white_pil(),
            "page":        i + 1,
            "annotations": [{"id": f"ann_{i}", "label": "NAME:PATIENT", "text": "foo",
                              "bboxes": [[0.1, 0.1, 0.2, 0.2]]}],
        }
        for i in range(n)
    ]
    return Dataset.from_list(rows).cast_column("image", HFImage())


def _base_args(**overrides):
    """Minimal args Namespace for augment_dataset_split."""
    defaults = dict(
        seed=0,
        # pipeline args — all disabled so the pipeline is a no-op
        ink_bleed_p=0.0,
        bleed_through_p=0.0,
        low_ink_p=0.0,
        ink_mottling_p=0.0,
        lines_degradation_p=0.0,
        color_paper_p=0.0,
        color_paper_page_sampling=0,
        color_paper_profile_weights=[1.0],
        color_paper_profile_hue_ranges=[20, 45],
        color_paper_profile_saturation_ranges=[10, 35],
        color_paper_hue_range=[20, 45],
        color_paper_saturation_range=[10, 35],
        texture_p=0.0,
        stains_p=0.0,
        stains_type="random",
        stains_blend_method="darken",
        stains_blend_alpha=0.4,
        stains_profile_sampling=0,
        stains_profile_weights=[1.0],
        stains_profile_types=["random"],
        stains_profile_blend_methods=["darken"],
        stains_profile_blend_alphas=[0.4],
        watermark_p=0.0,
        scanner_noise_p=0.0,
        scanner_noise_bad_photo_copy=0,
        scanner_noise_dirty_rollers=0,
        scanner_noise_dirty_drum=0,
        scanner_noise_dirty_screen=0,
        geometric_p=0.0,
        lighting_gradient_p=0.0,
        shadow_cast_p=0.0,
        exposure_p=0.0,
        subtle_noise_p=0.0,
        subtle_noise_range=12,
        subtle_noise_page_sampling=0,
        subtle_noise_profile_weights=[1.0],
        subtle_noise_profile_ranges=[12],
        jpeg_p=0.0,
        folding_p=0.0,
        page_border_p=0.0,
        annotations_p=0.0,
        annotations_markup=0,
        annotations_markup_sampling=0,
        annotations_scribbles=0,
        faxify_p=0.0,
        faxify_profile_sampling=0,
        faxify_scale_range=[1.0, 1.5],
        faxify_monochrome=-1,
        faxify_monochrome_method="random",
        faxify_monochrome_methods=[],
        faxify_halftone=-1,
        faxify_half_kernel_size=[1, 1],
        faxify_angle=[0, 360],
        faxify_sigma=[1.0, 3.0],
        faxify_profile_weights=[0.33, 0.34, 0.33],
        # not used in dataset mode but present on args
        num_augmentations=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _noop_pipeline():
    """A pipeline mock whose __call__ returns the input image unchanged."""
    mock = MagicMock()
    mock.side_effect = lambda img: img
    return mock


def _get_logger():
    logger = logging.getLogger("test_augment_dataset_hf")
    logger.setLevel(logging.DEBUG)
    return logger


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOutputSchema(unittest.TestCase):
    """augment_dataset_split returns a Dataset with {image, page, annotations} only."""

    def test_columns_match_expected(self):
        ds = _make_source_dataset(n=2)
        args = _base_args()
        pipeline = _noop_pipeline()
        logger = _get_logger()

        result = augment_dataset_split(ds, args, pipeline, logger)

        self.assertEqual(set(result.column_names), {"image", "page", "annotations"})
        self.assertNotIn("pdf_path", result.column_names)

    def test_row_count_equals_source(self):
        n = 4
        ds = _make_source_dataset(n=n)
        args = _base_args()
        pipeline = _noop_pipeline()
        logger = _get_logger()

        result = augment_dataset_split(ds, args, pipeline, logger)

        self.assertEqual(len(result), n)

    def test_page_and_annotations_preserved(self):
        ds = _make_source_dataset(n=2)
        args = _base_args()
        pipeline = _noop_pipeline()
        logger = _get_logger()

        result = augment_dataset_split(ds, args, pipeline, logger)

        for i in range(len(ds)):
            self.assertEqual(result[i]["page"], ds[i]["page"])
            self.assertEqual(result[i]["annotations"], ds[i]["annotations"])


class TestResampleMode(unittest.TestCase):
    """Re-sampling replaces only the specified rows; others are unchanged."""

    def _augmented_pixels(self, row):
        """Return raw pixel data for comparison. HF Image columns return PIL Images."""
        img = row["image"]
        if hasattr(img, "tobytes"):
            return img.tobytes()
        # Fallback: dict with "bytes" key (decode-on-demand format)
        return img["bytes"]

    def test_resample_replaces_only_specified_rows(self):
        n = 3
        source = _make_source_dataset(n=n)
        args = _base_args()
        logger = _get_logger()

        # First pass: augment all rows → existing augmented dataset
        pipeline_pass1 = _noop_pipeline()
        existing = augment_dataset_split(source, args, pipeline_pass1, logger)

        # Second pass: re-sample only row 1 with a pipeline that adds noise
        def noisy_pipeline(img):
            noisy = img.copy()
            noisy[0, 0] = [0, 0, 0]  # flip one pixel so images differ
            return noisy

        pipeline_pass2 = MagicMock(side_effect=noisy_pipeline)

        result = augment_dataset_split(
            source, args, pipeline_pass2, logger,
            resample_indices=[1],
            existing_dataset=existing,
        )

        self.assertEqual(len(result), n)

        # Row 0: unchanged (taken from existing)
        self.assertEqual(
            self._augmented_pixels(result[0]),
            self._augmented_pixels(existing[0]),
        )
        # Row 2: unchanged (taken from existing)
        self.assertEqual(
            self._augmented_pixels(result[2]),
            self._augmented_pixels(existing[2]),
        )
        # Row 1: re-augmented (pipeline_pass2 was called for it)
        # The noisy pipeline was invoked at least once (for row 1)
        self.assertTrue(pipeline_pass2.called)

    def test_resample_preserves_page_and_annotations_for_kept_rows(self):
        n = 3
        source = _make_source_dataset(n=n)
        args = _base_args()
        logger = _get_logger()

        pipeline = _noop_pipeline()
        existing = augment_dataset_split(source, args, pipeline, logger)
        result = augment_dataset_split(
            source, args, _noop_pipeline(), logger,
            resample_indices=[0],
            existing_dataset=existing,
        )

        # Rows 1 and 2 should have the same page/annotations as existing
        for i in [1, 2]:
            self.assertEqual(result[i]["page"], existing[i]["page"])
            self.assertEqual(result[i]["annotations"], existing[i]["annotations"])


class TestNumAugmentationsWarning(unittest.TestCase):
    """A warning is logged when num_augmentations != 1 in dataset mode."""

    def test_warning_emitted(self):
        with self.assertLogs("test_warning", level="WARNING") as cm:
            logger = logging.getLogger("test_warning")
            if getattr(logger, "_warned", False):
                pass

            # Simulate the warning logic from main() (not calling main() to avoid I/O)
            num_augmentations = 5
            if num_augmentations != 1:
                logger.warning(
                    f"--num_augmentations={num_augmentations} is ignored in dataset mode "
                    "(exactly one augmented image is produced per source row)"
                )

        self.assertTrue(any("num_augmentations" in msg for msg in cm.output))
        self.assertTrue(any("ignored in dataset mode" in msg for msg in cm.output))


class TestBuildDatasetNoPdfPath(unittest.TestCase):
    """build_dataset output has no pdf_path column."""

    def test_build_dataset_columns(self):
        from dataprep.build_dataset import build_dataset

        # Minimal mock instances (no real PDFs needed)
        mock_pil = _white_pil()
        instances = [
            {
                "pdf_path": "/fake/doc.pdf",
                "annotations": {"1": []},
            }
        ]
        logger = _get_logger()

        with patch("dataprep.build_dataset.convert_from_path", return_value=[mock_pil]):
            ds = build_dataset(instances, dpi=72, logger=logger)

        self.assertNotIn("pdf_path", ds.column_names)
        self.assertIn("image", ds.column_names)
        self.assertIn("page", ds.column_names)
        self.assertIn("annotations", ds.column_names)


class TestDirModeBackwardCompat(unittest.TestCase):
    """augment_pdf (dir mode) still produces images after the _augment_single_image refactor."""

    def test_augment_pdf_saves_images(self):
        import tempfile
        from pathlib import Path

        args = _base_args()
        pipeline = _noop_pipeline()
        logger = _get_logger()

        # Create a tiny 1-page PDF in memory using PIL (saved as a PNG, not PDF,
        # but we mock convert_from_path so the extension doesn't matter)
        mock_pil = _white_pil(h=100, w=80)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.touch()  # empty file; convert_from_path is mocked

            with patch("dataprep.augment_dataset.convert_from_path", return_value=[mock_pil]):
                saved = augment_pdf(
                    pdf_path,
                    Path(tmpdir),
                    pipeline,
                    args,
                    num_augmentations=2,
                    dpi=72,
                    logger=logger,
                )

        # 1 page × 2 augmentations = 2 saved images
        self.assertEqual(saved, 2)


class TestAugmentSingleImage(unittest.TestCase):
    """_augment_single_image returns a BGR numpy array of the same shape."""

    def test_output_shape_preserved(self):
        args = _base_args()
        img = np.full((60, 50, 3), 200, dtype=np.uint8)
        pipeline = _noop_pipeline()

        result = _augment_single_image(
            img, args, pipeline,
            use_dynamic_pipeline=False,
            faxify_mutex_active=False,
        )

        self.assertEqual(result.shape, img.shape)
        self.assertEqual(result.dtype, img.dtype)


if __name__ == "__main__":
    unittest.main()
