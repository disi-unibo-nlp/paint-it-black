"""
Tests for the page-level sampling feature in augment_pdfs.py.

Tests cover:
- ColorPaper and SubtleNoise are excluded from the augraphy pipeline when page sampling is on
- _build_per_image_pipeline samples a ColorPaper profile and places it in the paper phase
- _apply_page_sampled_augmentations handles SubtleNoise profile selection
- Weight normalization (weights that don't sum to 1.0 are accepted)
- No-op behavior when page sampling is disabled
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, "src")

from dataprep.augment_pdfs import (
    _apply_page_sampled_augmentations,
    _build_paper_phase,
    _build_post_phase,
    _build_ink_phase,
    _build_per_image_pipeline,
)
from augraphy import ColorPaper, SubtleNoise, AugmentationSequence


# ── Helpers ───────────────────────────────────────────────────────────────────

def white_image(h=100, w=100):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def base_args(**overrides):
    """Minimal Namespace covering every arg touched by the tested functions."""
    defaults = dict(
        # color_paper
        color_paper_p=1.0,
        color_paper_hue_range=[20, 45],
        color_paper_saturation_range=[10, 35],
        color_paper_page_sampling=0,
        color_paper_num_profiles=1,
        color_paper_profile_weights=[1.0],
        color_paper_profile_hue_ranges=[20, 45],
        color_paper_profile_saturation_ranges=[10, 35],
        # subtle_noise
        subtle_noise_p=1.0,
        subtle_noise_range=12,
        subtle_noise_page_sampling=0,
        subtle_noise_num_profiles=1,
        subtle_noise_profile_weights=[1.0],
        subtle_noise_profile_ranges=[12],
        # other paper-phase args (needed by _build_paper_phase)
        texture_p=0.0,
        stains_p=0.0,
        watermark_p=0.0,
        # ink-phase args (needed by _build_ink_phase)
        ink_bleed_p=0.0,
        bleed_through_p=0.0,
        low_ink_p=0.0,
        ink_mottling_p=0.0,
        lines_degradation_p=0.0,
        # other post-phase args (needed by _build_post_phase)
        scanner_noise_p=0.0,
        geometric_p=0.0,
        lighting_gradient_p=0.0,
        shadow_cast_p=0.0,
        exposure_p=0.0,
        jpeg_p=0.0,
        folding_p=0.0,
        folding_backdrop_color=[255, 255, 255],
        page_border_p=0.0,
        annotations_p=0.0,
        annotations_markup=1,
        annotations_markup_type="random",
        annotations_markup_num_lines_range=[1, 4],
        annotations_markup_length_range=[0.5, 1.0],
        annotations_markup_pencil_thickness_range=[1, 3],
        annotations_markup_pen_thickness_range=[1, 3],
        annotations_markup_marker_thickness_range=[1, 3],
        annotations_markup_highlighter_thickness_range=[1, 3],
        annotations_markup_sampling=0,
        annotations_markup_large_word_mode=1,
        annotations_markup_single_word_mode=0,
        annotations_markup_repetitions=[1, 1],
        annotations_markup_type_weights=[0.25, 0.25, 0.25, 0.25],
        annotations_scribbles_size_range=[400, 600],
        faxify_p=0.0,
        faxify_profile_sampling=0,
        faxify_profile_weights=[0.33, 0.34, 0.33],
        faxify_scale_range=[1.0, 1.5],
        faxify_monochrome=-1,
        faxify_monochrome_method="random",
        faxify_halftone=-1,
        faxify_half_kernel_size=[1, 1],
        faxify_angle=[0, 360],
        faxify_sigma=[1.0, 3.0],
        annotations_markup_ink="random",
        annotations_markup_ink_weights=[0.25, 0.25, 0.25, 0.25],
        annotations_markup_pencil_colors=["random"],
        annotations_markup_pen_colors=["random"],
        annotations_markup_marker_colors=["random"],
        annotations_markup_highlighter_colors=["random"],
        annotations_markup_control_points_range=[2, 3],
        annotations_markup_line_offset=3,
        # seed (needed by _build_per_image_pipeline)
        seed=42,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _flatten_phase(phase):
    """Recursively collect augmentation types from a phase list (unwrap AugmentationSequence)."""
    types = []
    for aug in phase:
        if isinstance(aug, AugmentationSequence):
            types.extend(_flatten_phase(aug.augmentations))
        else:
            types.append(type(aug))
    return types


# ── Pipeline inclusion tests ──────────────────────────────────────────────────

def test_page_sampling_off_color_paper_in_pipeline():
    args = base_args(color_paper_page_sampling=0, color_paper_p=0.9)
    phase = _build_paper_phase(args)
    types = [type(aug) for aug in phase]
    assert ColorPaper in types, "ColorPaper should be in pipeline when page_sampling=0"
    print("PASS test_page_sampling_off_color_paper_in_pipeline")


def test_page_sampling_on_color_paper_not_in_pipeline():
    args = base_args(color_paper_page_sampling=1, color_paper_p=0.9)
    phase = _build_paper_phase(args)
    types = [type(aug) for aug in phase]
    assert ColorPaper not in types, "ColorPaper should NOT be in pipeline when page_sampling=1"
    print("PASS test_page_sampling_on_color_paper_not_in_pipeline")


def test_page_sampling_off_subtle_noise_in_pipeline():
    args = base_args(subtle_noise_page_sampling=0, subtle_noise_p=0.9)
    phase = _build_post_phase(args)
    types = [type(aug) for aug in phase]
    assert SubtleNoise in types, "SubtleNoise should be in pipeline when page_sampling=0"
    print("PASS test_page_sampling_off_subtle_noise_in_pipeline")


def test_page_sampling_on_subtle_noise_not_in_pipeline():
    args = base_args(subtle_noise_page_sampling=1, subtle_noise_p=0.9)
    phase = _build_post_phase(args)
    types = [type(aug) for aug in phase]
    assert SubtleNoise not in types, "SubtleNoise should NOT be in pipeline when page_sampling=1"
    print("PASS test_page_sampling_on_subtle_noise_not_in_pipeline")


# ── _build_per_image_pipeline tests ──────────────────────────────────────────

def test_per_image_pipeline_paper_phase_has_color_paper():
    """ColorPaper should appear in the paper phase of the per-image pipeline."""
    args = base_args(
        color_paper_page_sampling=1,
        color_paper_p=1.0,
        color_paper_num_profiles=1,
        color_paper_profile_weights=[1.0],
        color_paper_profile_hue_ranges=[20, 25],
        color_paper_profile_saturation_ranges=[80, 100],
    )
    pipeline = _build_per_image_pipeline(
        args,
        _build_ink_phase(args),
        _build_paper_phase(args),
        _build_post_phase(args),
    )
    paper_types = _flatten_phase(pipeline.paper_phase.augmentations)
    assert ColorPaper in paper_types, "ColorPaper should be in paper phase of per-image pipeline"
    print("PASS test_per_image_pipeline_paper_phase_has_color_paper")


def test_per_image_pipeline_color_paper_first_profile():
    """weights=[1,0] → always picks profile 0 (high saturation) → white image changes."""
    np.random.seed(0)
    args = base_args(
        color_paper_page_sampling=1,
        color_paper_p=1.0,
        color_paper_num_profiles=2,
        color_paper_profile_weights=[1.0, 0.0],
        color_paper_profile_hue_ranges=[20, 25, 90, 95],
        color_paper_profile_saturation_ranges=[80, 100, 80, 100],
        subtle_noise_page_sampling=0,
    )
    img = white_image()
    pipeline = _build_per_image_pipeline(
        args,
        _build_ink_phase(args),
        _build_paper_phase(args),
        _build_post_phase(args),
    )
    result = pipeline(img)
    assert not np.array_equal(img, result), "ColorPaper with high saturation should change a white image"
    print("PASS test_per_image_pipeline_color_paper_first_profile")


def test_per_image_pipeline_color_paper_second_profile():
    """weights=[0,1] → always picks profile 1 (different hue range) → image also changes."""
    np.random.seed(0)
    args = base_args(
        color_paper_page_sampling=1,
        color_paper_p=1.0,
        color_paper_num_profiles=2,
        color_paper_profile_weights=[0.0, 1.0],
        color_paper_profile_hue_ranges=[20, 25, 90, 95],
        color_paper_profile_saturation_ranges=[80, 100, 80, 100],
        subtle_noise_page_sampling=0,
    )
    img = white_image()
    pipeline = _build_per_image_pipeline(
        args,
        _build_ink_phase(args),
        _build_paper_phase(args),
        _build_post_phase(args),
    )
    result = pipeline(img)
    assert not np.array_equal(img, result), "ColorPaper profile 1 with high saturation should change a white image"
    print("PASS test_per_image_pipeline_color_paper_second_profile")


# ── _apply_page_sampled_augmentations tests ───────────────────────────────────

def test_page_sampling_is_noop_when_disabled():
    args = base_args(color_paper_page_sampling=0, subtle_noise_page_sampling=0)
    img = white_image()
    result = _apply_page_sampled_augmentations(img, args)
    assert np.array_equal(img, result), "Should be no-op when both page_sampling flags are 0"
    print("PASS test_page_sampling_is_noop_when_disabled")


def test_subtle_noise_low_range_changes_image():
    """Profile with subtle_range=5 (weights=[1,0]) always selected → image changes."""
    np.random.seed(0)
    args = base_args(
        subtle_noise_page_sampling=1,
        subtle_noise_p=1.0,
        subtle_noise_num_profiles=2,
        subtle_noise_profile_weights=[1.0, 0.0],
        subtle_noise_profile_ranges=[5, 30],
        color_paper_page_sampling=0,
    )
    img = white_image()
    result = _apply_page_sampled_augmentations(img, args)
    assert not np.array_equal(img, result), "subtle_range=5 should change the image"
    print("PASS test_subtle_noise_low_range_changes_image")


def test_subtle_noise_high_range_changes_image():
    """Profile with subtle_range=30 (weights=[0,1]) always selected → image changes."""
    np.random.seed(0)
    args = base_args(
        subtle_noise_page_sampling=1,
        subtle_noise_p=1.0,
        subtle_noise_num_profiles=2,
        subtle_noise_profile_weights=[0.0, 1.0],
        subtle_noise_profile_ranges=[5, 30],
        color_paper_page_sampling=0,
    )
    img = white_image()
    result = _apply_page_sampled_augmentations(img, args)
    assert not np.array_equal(img, result), "subtle_range=30 should change the image"
    print("PASS test_subtle_noise_high_range_changes_image")


def test_weight_normalization():
    """Weights [2.0, 2.0] (sum=4) should be accepted and both profiles selectable."""
    np.random.seed(42)
    args = base_args(
        color_paper_page_sampling=1,
        color_paper_p=1.0,
        color_paper_num_profiles=2,
        color_paper_profile_weights=[2.0, 2.0],
        color_paper_profile_hue_ranges=[20, 25, 90, 95],
        color_paper_profile_saturation_ranges=[10, 20, 10, 20],
        subtle_noise_page_sampling=0,
    )
    ink = _build_ink_phase(args)
    paper = _build_paper_phase(args)
    post = _build_post_phase(args)
    # Just verify it runs without error 20 times
    for _ in range(20):
        _build_per_image_pipeline(args, ink, paper, post)
    print("PASS test_weight_normalization")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_page_sampling_off_color_paper_in_pipeline()
    test_page_sampling_on_color_paper_not_in_pipeline()
    test_page_sampling_off_subtle_noise_in_pipeline()
    test_page_sampling_on_subtle_noise_not_in_pipeline()
    test_per_image_pipeline_paper_phase_has_color_paper()
    test_per_image_pipeline_color_paper_first_profile()
    test_per_image_pipeline_color_paper_second_profile()
    test_page_sampling_is_noop_when_disabled()
    test_subtle_noise_low_range_changes_image()
    test_subtle_noise_high_range_changes_image()
    test_weight_normalization()
    print("\nAll tests passed.")
