"""
augment_pdfs.py — Apply document augmentations to a folder of PDFs using augraphy.

For each PDF in input_dir, each page is converted to an image and augmented
num_augmentations times. Results are saved as PNGs under output_dir.

Output structure:
    output_dir/<pdf_stem>/page_<N>_aug_<M>.png

Usage:
    python3 src/dataprep/augment_pdfs.py --config dataprep/augment_config
    python3 src/dataprep/augment_pdfs.py --config dataprep/augment_config --geometric_p 0.0
    python3 src/dataprep/augment_pdfs.py --config dataprep/augment_config --faxify_p 0.2

Every augmentation parameter is individually settable from the config file or CLI.
Set any <aug>_p to 0.0 to disable that augmentation entirely.
Set any OneOf membership flag (e.g. scanner_noise_dirty_drum) to 0 to exclude it from the pool.
Different presets are different config files — there is no --pipeline choice.
"""

import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from augraphy import (
    AugraphyPipeline,
    AugmentationSequence,
    OneOf,
    # Ink phase
    InkBleed,
    BleedThrough,
    LowInkRandomLines,
    LowInkPeriodicLines,
    InkMottling,
    LinesDegradation,
    # Paper phase
    ColorPaper,
    NoiseTexturize,
    BrightnessTexturize,
    Stains,
    WaterMark,
    # Post phase
    BadPhotoCopy,
    DirtyDrum,
    DirtyRollers,
    DirtyScreen,
    Geometric,
    LightingGradient,
    ShadowCast,
    Brightness,
    Gamma,
    SubtleNoise,
    Jpeg,
    Folding,
    PageBorder,
    Markup,
    Scribbles,
    Faxify,
)

from core.utils import ConfigArgumentParser, init_logger, set_seed


# ── Phase builders ────────────────────────────────────────────────────────────

def _build_ink_phase(args) -> list:
    phase = []

    if args.ink_bleed_p > 0:
        phase.append(InkBleed(
            intensity_range=tuple(args.ink_bleed_intensity_range),
            kernel_size=tuple(args.ink_bleed_kernel_size),
            severity=tuple(args.ink_bleed_severity),
            p=args.ink_bleed_p,
        ))

    if args.bleed_through_p > 0:
        phase.append(BleedThrough(
            intensity_range=tuple(args.bleed_through_intensity_range),
            color_range=tuple(args.bleed_through_color_range),
            ksize=tuple(args.bleed_through_ksize),
            sigmaX=args.bleed_through_sigma_x,
            alpha=args.bleed_through_alpha,
            offsets=tuple(args.bleed_through_offsets),
            p=args.bleed_through_p,
        ))

    if args.low_ink_p > 0:
        members = []
        if args.low_ink_random_lines:
            members.append(LowInkRandomLines(
                count_range=tuple(args.low_ink_random_lines_count_range),
                use_consistent_lines=bool(args.low_ink_random_lines_consistent),
                noise_probability=args.low_ink_random_lines_noise_prob,
                p=1.0,
            ))
        if args.low_ink_periodic_lines:
            members.append(LowInkPeriodicLines(
                count_range=tuple(args.low_ink_periodic_lines_count_range),
                period_range=tuple(args.low_ink_periodic_lines_period_range),
                use_consistent_lines=bool(args.low_ink_periodic_lines_consistent),
                noise_probability=args.low_ink_periodic_lines_noise_prob,
                p=1.0,
            ))
        if members:
            phase.append(OneOf(members, p=args.low_ink_p))

    if args.ink_mottling_p > 0:
        phase.append(InkMottling(
            ink_mottling_alpha_range=tuple(args.ink_mottling_alpha_range),
            ink_mottling_noise_scale_range=tuple(args.ink_mottling_noise_scale_range),
            ink_mottling_gaussian_kernel_range=tuple(args.ink_mottling_gaussian_kernel_range),
            p=args.ink_mottling_p,
        ))

    if args.lines_degradation_p > 0:
        phase.append(LinesDegradation(
            line_gradient_range=tuple(args.lines_degradation_gradient_range),
            line_split_probability=tuple(args.lines_degradation_split_probability),
            line_replacement_value=tuple(args.lines_degradation_replacement_value),
            line_replacement_probability=tuple(args.lines_degradation_replacement_probability),
            p=args.lines_degradation_p,
        ))

    return phase


def _build_paper_phase(args) -> list:
    phase = []

    if args.color_paper_p > 0:
        phase.append(ColorPaper(
            hue_range=tuple(args.color_paper_hue_range),
            saturation_range=tuple(args.color_paper_saturation_range),
            p=args.color_paper_p,
        ))

    if args.texture_p > 0:
        phase.append(AugmentationSequence(
            [
                NoiseTexturize(
                    sigma_range=tuple(args.texture_noise_sigma_range),
                    turbulence_range=tuple(args.texture_noise_turbulence_range),
                    p=1.0,
                ),
                BrightnessTexturize(
                    texturize_range=tuple(args.texture_brightness_texturize_range),
                    deviation=args.texture_brightness_deviation,
                    p=1.0,
                ),
            ],
            p=args.texture_p,
        ))

    if args.stains_p > 0:
        phase.append(Stains(
            stains_type=args.stains_type,
            stains_blend_method=args.stains_blend_method,
            stains_blend_alpha=args.stains_blend_alpha,
            p=args.stains_p,
        ))

    if args.watermark_p > 0:
        phase.append(WaterMark(
            watermark_word=args.watermark_word,
            watermark_font_size=tuple(args.watermark_font_size),
            watermark_font_thickness=tuple(args.watermark_font_thickness),
            watermark_rotation=tuple(args.watermark_rotation),
            watermark_location=args.watermark_location,
            watermark_color=args.watermark_color,
            watermark_method=args.watermark_method,
            p=args.watermark_p,
        ))

    return phase


def _build_post_phase(args) -> list:
    phase = []

    if args.scanner_noise_p > 0:
        members = []
        if args.scanner_noise_bad_photo_copy:
            members.append(BadPhotoCopy(
                noise_type=args.scanner_noise_bad_photo_copy_noise_type,
                noise_side=args.scanner_noise_bad_photo_copy_noise_side,
                noise_iteration=tuple(args.scanner_noise_bad_photo_copy_noise_iter),
                noise_size=tuple(args.scanner_noise_bad_photo_copy_noise_size),
                noise_value=tuple(args.scanner_noise_bad_photo_copy_noise_value),
                noise_sparsity=tuple(args.scanner_noise_bad_photo_copy_sparsity),
                noise_concentration=tuple(args.scanner_noise_bad_photo_copy_concentration),
                blur_noise=args.scanner_noise_bad_photo_copy_blur_noise,
                wave_pattern=args.scanner_noise_bad_photo_copy_wave_pattern,
                edge_effect=args.scanner_noise_bad_photo_copy_edge_effect,
                p=1.0,
            ))
        if args.scanner_noise_dirty_rollers:
            members.append(DirtyRollers(
                line_width_range=tuple(args.scanner_noise_dirty_rollers_line_width),
                scanline_type=args.scanner_noise_dirty_rollers_scanline_type,
                p=1.0,
            ))
        if args.scanner_noise_dirty_drum:
            members.append(DirtyDrum(
                line_width_range=tuple(args.scanner_noise_dirty_drum_line_width),
                line_concentration=args.scanner_noise_dirty_drum_line_concentration,
                direction=args.scanner_noise_dirty_drum_direction,
                noise_intensity=args.scanner_noise_dirty_drum_noise_intensity,
                noise_value=tuple(args.scanner_noise_dirty_drum_noise_value),
                ksize=tuple(args.scanner_noise_dirty_drum_ksize),
                p=1.0,
            ))
        if args.scanner_noise_dirty_screen:
            members.append(DirtyScreen(
                n_clusters=tuple(args.scanner_noise_dirty_screen_n_clusters),
                n_samples=tuple(args.scanner_noise_dirty_screen_n_samples),
                std_range=tuple(args.scanner_noise_dirty_screen_std_range),
                value_range=tuple(args.scanner_noise_dirty_screen_value_range),
                p=1.0,
            ))
        if members:
            phase.append(OneOf(members, p=args.scanner_noise_p))

    if args.geometric_p > 0:
        phase.append(Geometric(
            rotate_range=tuple(args.geometric_rotate_range),
            padding_type="fill",
            padding_value=tuple(args.geometric_padding_value),
            p=args.geometric_p,
        ))

    if args.lighting_gradient_p > 0:
        phase.append(LightingGradient(
            light_position=None,
            direction=None,
            max_brightness=args.lighting_gradient_max_brightness,
            min_brightness=args.lighting_gradient_min_brightness,
            mode=args.lighting_gradient_mode,
            transparency=None,
            p=args.lighting_gradient_p,
        ))

    if args.shadow_cast_p > 0:
        phase.append(ShadowCast(
            shadow_side=args.shadow_cast_side,
            shadow_vertices_range=tuple(args.shadow_cast_vertices_range),
            shadow_width_range=tuple(args.shadow_cast_width_range),
            shadow_height_range=tuple(args.shadow_cast_height_range),
            shadow_color=(0, 0, 0),
            shadow_opacity_range=tuple(args.shadow_cast_opacity_range),
            shadow_iterations_range=tuple(args.shadow_cast_iterations_range),
            shadow_blur_kernel_range=tuple(args.shadow_cast_blur_kernel_range),
            p=args.shadow_cast_p,
        ))

    if args.exposure_p > 0:
        members = []
        if args.exposure_brightness:
            members.append(Brightness(brightness_range=tuple(args.exposure_brightness_range), p=1.0))
        if args.exposure_gamma:
            members.append(Gamma(gamma_range=tuple(args.exposure_gamma_range), p=1.0))
        if members:
            phase.append(OneOf(members, p=args.exposure_p))

    if args.subtle_noise_p > 0:
        phase.append(SubtleNoise(subtle_range=args.subtle_noise_range, p=args.subtle_noise_p))

    if args.jpeg_p > 0:
        phase.append(Jpeg(quality_range=tuple(args.jpeg_quality_range), p=args.jpeg_p))

    if args.folding_p > 0:
        phase.append(Folding(
            fold_count=args.folding_fold_count,
            fold_noise=args.folding_fold_noise,
            gradient_width=tuple(args.folding_gradient_width),
            gradient_height=tuple(args.folding_gradient_height),
            p=args.folding_p,
        ))

    if args.page_border_p > 0:
        phase.append(PageBorder(
            page_rotation_angle_range=tuple(args.page_border_rotation_range),
            same_page_border=1,
            p=args.page_border_p,
        ))

    if args.annotations_p > 0:
        members = []
        if args.annotations_markup:
            members.append(Markup(
                num_lines_range=tuple(args.annotations_markup_num_lines_range),
                markup_type="random",
                markup_ink="random",
                markup_color="random",
                p=1.0,
            ))
        if args.annotations_scribbles:
            members.append(Scribbles(
                scribbles_type="random",
                scribbles_ink="random",
                scribbles_count_range=tuple(args.annotations_scribbles_count_range),
                scribbles_thickness_range=tuple(args.annotations_scribbles_thickness_range),
                p=1.0,
            ))
        if members:
            phase.append(OneOf(members, p=args.annotations_p))

    if args.faxify_p > 0:
        phase.append(Faxify(
            scale_range=tuple(args.faxify_scale_range),
            monochrome=args.faxify_monochrome,
            halftone=args.faxify_halftone,
            p=args.faxify_p,
        ))

    return phase


def build_pipeline(args) -> AugraphyPipeline:
    return AugraphyPipeline(
        ink_phase=_build_ink_phase(args),
        paper_phase=_build_paper_phase(args),
        post_phase=_build_post_phase(args),
        random_seed=args.seed,
        log=False,
    )


# ── I/O ───────────────────────────────────────────────────────────────────────

def augment_pdf(pdf_path: Path, output_dir: Path, pipeline, num_augmentations: int, dpi: int, logger, flat: bool = False) -> int:
    """Augment all pages of a single PDF. Returns total images saved."""
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    saved = 0
    out_subdir = output_dir if flat else output_dir / pdf_path.stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    for page_idx, pil_page in enumerate(pages):
        img = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
        for aug_idx in range(num_augmentations):
            augmented = pipeline(img)
            out_path = out_subdir / f"page_{page_idx:03d}_aug_{aug_idx:03d}.png"
            cv2.imwrite(str(out_path), augmented)
            saved += 1
            logger.debug(f"  Saved {out_path.name}")

    return saved


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = ConfigArgumentParser(description="Augment PDFs with augraphy")

    # General
    parser.add_argument("--run_name", type=str, default="augment_run")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)

    # I/O
    parser.add_argument("--input_dir", type=str, default="./data/input")
    parser.add_argument("--output_dir", type=str, default="./data/augmented")
    parser.add_argument("--num_augmentations", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--flat", action="store_true",
                        help="Write images directly into output_dir (skip per-doc subdir)")

    # ── Ink phase ────────────────────────────────────────────────────────────

    # ink_bleed
    parser.add_argument("--ink_bleed_p", type=float, default=0.5)
    parser.add_argument("--ink_bleed_intensity_range", type=float, nargs=2, default=[0.4, 0.6])
    parser.add_argument("--ink_bleed_kernel_size", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--ink_bleed_severity", type=float, nargs=2, default=[0.2, 0.35])

    # bleed_through
    parser.add_argument("--bleed_through_p", type=float, default=0.4)
    parser.add_argument("--bleed_through_intensity_range", type=float, nargs=2, default=[0.1, 0.5])
    parser.add_argument("--bleed_through_color_range", type=int, nargs=2, default=[0, 224])
    parser.add_argument("--bleed_through_ksize", type=int, nargs=2, default=[17, 17])
    parser.add_argument("--bleed_through_sigma_x", type=int, default=1)
    parser.add_argument("--bleed_through_alpha", type=float, default=0.15)
    parser.add_argument("--bleed_through_offsets", type=int, nargs=2, default=[15, 15])

    # low_ink (OneOf: random_lines / periodic_lines)
    parser.add_argument("--low_ink_p", type=float, default=0.4)
    parser.add_argument("--low_ink_random_lines", type=int, default=1)
    parser.add_argument("--low_ink_random_lines_count_range", type=int, nargs=2, default=[3, 8])
    parser.add_argument("--low_ink_random_lines_consistent", type=int, default=1)
    parser.add_argument("--low_ink_random_lines_noise_prob", type=float, default=0.1)
    parser.add_argument("--low_ink_periodic_lines", type=int, default=1)
    parser.add_argument("--low_ink_periodic_lines_count_range", type=int, nargs=2, default=[2, 4])
    parser.add_argument("--low_ink_periodic_lines_period_range", type=int, nargs=2, default=[15, 40])
    parser.add_argument("--low_ink_periodic_lines_consistent", type=int, default=1)
    parser.add_argument("--low_ink_periodic_lines_noise_prob", type=float, default=0.1)

    # ink_mottling
    parser.add_argument("--ink_mottling_p", type=float, default=0.35)
    parser.add_argument("--ink_mottling_alpha_range", type=float, nargs=2, default=[0.15, 0.3])
    parser.add_argument("--ink_mottling_noise_scale_range", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--ink_mottling_gaussian_kernel_range", type=int, nargs=2, default=[3, 5])

    # lines_degradation
    parser.add_argument("--lines_degradation_p", type=float, default=0.3)
    parser.add_argument("--lines_degradation_gradient_range", type=int, nargs=2, default=[32, 200])
    parser.add_argument("--lines_degradation_split_probability", type=float, nargs=2, default=[0.2, 0.4])
    parser.add_argument("--lines_degradation_replacement_value", type=int, nargs=2, default=[245, 255])
    parser.add_argument("--lines_degradation_replacement_probability", type=float, nargs=2, default=[0.3, 0.5])

    # ── Paper phase ──────────────────────────────────────────────────────────

    # color_paper
    parser.add_argument("--color_paper_p", type=float, default=0.7)
    parser.add_argument("--color_paper_hue_range", type=int, nargs=2, default=[20, 45])
    parser.add_argument("--color_paper_saturation_range", type=int, nargs=2, default=[10, 35])

    # texture (AugmentationSequence: noise_texturize + brightness_texturize)
    parser.add_argument("--texture_p", type=float, default=0.6)
    parser.add_argument("--texture_noise_sigma_range", type=int, nargs=2, default=[3, 8])
    parser.add_argument("--texture_noise_turbulence_range", type=int, nargs=2, default=[2, 4])
    parser.add_argument("--texture_brightness_texturize_range", type=float, nargs=2, default=[0.85, 0.99])
    parser.add_argument("--texture_brightness_deviation", type=float, default=0.06)

    # stains
    parser.add_argument("--stains_p", type=float, default=0.25)
    parser.add_argument("--stains_type", type=str, default="random")
    parser.add_argument("--stains_blend_method", type=str, default="darken")
    parser.add_argument("--stains_blend_alpha", type=float, default=0.4)

    # watermark
    parser.add_argument("--watermark_p", type=float, default=0.25)
    parser.add_argument("--watermark_word", type=str, default="random")
    parser.add_argument("--watermark_font_size", type=int, nargs=2, default=[10, 15])
    parser.add_argument("--watermark_font_thickness", type=int, nargs=2, default=[20, 25])
    parser.add_argument("--watermark_rotation", type=int, nargs=2, default=[0, 360])
    parser.add_argument("--watermark_location", type=str, default="random")
    parser.add_argument("--watermark_color", type=str, default="random")
    parser.add_argument("--watermark_method", type=str, default="darken")

    # ── Post phase ───────────────────────────────────────────────────────────

    # scanner_noise (OneOf: bad_photo_copy / dirty_rollers / dirty_drum / dirty_screen)
    parser.add_argument("--scanner_noise_p", type=float, default=0.55)
    parser.add_argument("--scanner_noise_bad_photo_copy", type=int, default=1)
    parser.add_argument("--scanner_noise_bad_photo_copy_noise_type", type=int, default=-1)
    parser.add_argument("--scanner_noise_bad_photo_copy_noise_side", type=str, default="random")
    parser.add_argument("--scanner_noise_bad_photo_copy_noise_iter", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--scanner_noise_bad_photo_copy_noise_size", type=int, nargs=2, default=[1, 3])
    parser.add_argument("--scanner_noise_bad_photo_copy_noise_value", type=int, nargs=2, default=[32, 96])
    parser.add_argument("--scanner_noise_bad_photo_copy_sparsity", type=float, nargs=2, default=[0.3, 0.6])
    parser.add_argument("--scanner_noise_bad_photo_copy_concentration", type=float, nargs=2, default=[0.1, 0.5])
    parser.add_argument("--scanner_noise_bad_photo_copy_blur_noise", type=int, default=-1)
    parser.add_argument("--scanner_noise_bad_photo_copy_wave_pattern", type=int, default=-1)
    parser.add_argument("--scanner_noise_bad_photo_copy_edge_effect", type=int, default=-1)
    parser.add_argument("--scanner_noise_dirty_rollers", type=int, default=1)
    parser.add_argument("--scanner_noise_dirty_rollers_line_width", type=int, nargs=2, default=[8, 14])
    parser.add_argument("--scanner_noise_dirty_rollers_scanline_type", type=int, default=0)
    parser.add_argument("--scanner_noise_dirty_drum", type=int, default=1)
    parser.add_argument("--scanner_noise_dirty_drum_line_width", type=int, nargs=2, default=[1, 4])
    parser.add_argument("--scanner_noise_dirty_drum_line_concentration", type=float, default=0.08)
    parser.add_argument("--scanner_noise_dirty_drum_direction", type=int, default=-1)
    parser.add_argument("--scanner_noise_dirty_drum_noise_intensity", type=float, default=0.4)
    parser.add_argument("--scanner_noise_dirty_drum_noise_value", type=int, nargs=2, default=[0, 30])
    parser.add_argument("--scanner_noise_dirty_drum_ksize", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--scanner_noise_dirty_screen", type=int, default=1)
    parser.add_argument("--scanner_noise_dirty_screen_n_clusters", type=int, nargs=2, default=[40, 80])
    parser.add_argument("--scanner_noise_dirty_screen_n_samples", type=int, nargs=2, default=[2, 15])
    parser.add_argument("--scanner_noise_dirty_screen_std_range", type=int, nargs=2, default=[1, 5])
    parser.add_argument("--scanner_noise_dirty_screen_value_range", type=int, nargs=2, default=[150, 250])

    # geometric
    parser.add_argument("--geometric_p", type=float, default=0.6)
    parser.add_argument("--geometric_rotate_range", type=int, nargs=2, default=[-3, 3])
    parser.add_argument("--geometric_padding_value", type=int, nargs=3, default=[255, 255, 255])

    # lighting_gradient
    parser.add_argument("--lighting_gradient_p", type=float, default=0.4)
    parser.add_argument("--lighting_gradient_max_brightness", type=int, default=255)
    parser.add_argument("--lighting_gradient_min_brightness", type=int, default=0)
    parser.add_argument("--lighting_gradient_mode", type=str, default="gaussian")

    # shadow_cast
    parser.add_argument("--shadow_cast_p", type=float, default=0.25)
    parser.add_argument("--shadow_cast_side", type=str, default="random")
    parser.add_argument("--shadow_cast_vertices_range", type=int, nargs=2, default=[1, 10])
    parser.add_argument("--shadow_cast_width_range", type=float, nargs=2, default=[0.2, 0.5])
    parser.add_argument("--shadow_cast_height_range", type=float, nargs=2, default=[0.2, 0.5])
    parser.add_argument("--shadow_cast_opacity_range", type=float, nargs=2, default=[0.2, 0.5])
    parser.add_argument("--shadow_cast_iterations_range", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--shadow_cast_blur_kernel_range", type=int, nargs=2, default=[101, 301])

    # exposure (OneOf: brightness / gamma)
    parser.add_argument("--exposure_p", type=float, default=0.5)
    parser.add_argument("--exposure_brightness", type=int, default=1)
    parser.add_argument("--exposure_brightness_range", type=float, nargs=2, default=[0.7, 1.3])
    parser.add_argument("--exposure_gamma", type=int, default=1)
    parser.add_argument("--exposure_gamma_range", type=float, nargs=2, default=[0.6, 1.4])

    # subtle_noise
    parser.add_argument("--subtle_noise_p", type=float, default=0.5)
    parser.add_argument("--subtle_noise_range", type=int, default=12)

    # jpeg
    parser.add_argument("--jpeg_p", type=float, default=0.4)
    parser.add_argument("--jpeg_quality_range", type=int, nargs=2, default=[55, 92])

    # folding
    parser.add_argument("--folding_p", type=float, default=0.25)
    parser.add_argument("--folding_fold_count", type=int, default=2)
    parser.add_argument("--folding_fold_noise", type=float, default=0.01)
    parser.add_argument("--folding_gradient_width", type=float, nargs=2, default=[0.1, 0.2])
    parser.add_argument("--folding_gradient_height", type=float, nargs=2, default=[0.01, 0.02])

    # page_border
    parser.add_argument("--page_border_p", type=float, default=0.3)
    parser.add_argument("--page_border_rotation_range", type=float, nargs=2, default=[-2.0, 2.0])

    # annotations (OneOf: markup / scribbles)
    parser.add_argument("--annotations_p", type=float, default=0.3)
    parser.add_argument("--annotations_markup", type=int, default=1)
    parser.add_argument("--annotations_markup_num_lines_range", type=int, nargs=2, default=[1, 4])
    parser.add_argument("--annotations_scribbles", type=int, default=1)
    parser.add_argument("--annotations_scribbles_count_range", type=int, nargs=2, default=[1, 3])
    parser.add_argument("--annotations_scribbles_thickness_range", type=int, nargs=2, default=[1, 2])

    # faxify (disabled by default — set faxify_p > 0 to enable)
    parser.add_argument("--faxify_p", type=float, default=0.0)
    parser.add_argument("--faxify_scale_range", type=float, nargs=2, default=[1.0, 1.5])
    parser.add_argument("--faxify_monochrome", type=int, default=-1)
    parser.add_argument("--faxify_halftone", type=int, default=-1)

    args = parser.parse_args()

    logger = init_logger(__name__, level=args.log_level)
    set_seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) in {input_dir}")
    logger.info(f"DPI: {args.dpi} | Augmentations per page: {args.num_augmentations}")
    logger.info(f"Output: {output_dir}")

    pipeline = build_pipeline(args)

    total_saved = 0
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name} ...")
        saved = augment_pdf(pdf_path, output_dir, pipeline, args.num_augmentations, args.dpi, logger, flat=args.flat)
        logger.info(f"  -> {saved} image(s) saved")
        total_saved += saved

    logger.info(f"Done. Total images saved: {total_saved}")


if __name__ == "__main__":
    main()
