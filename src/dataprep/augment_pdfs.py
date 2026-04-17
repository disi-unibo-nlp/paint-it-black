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

import random
import warnings
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from augraphy.augmentations.lib import smooth
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


# ── Constants ────────────────────────────────────────────────────────────────

_MARKUP_TYPES = ["strikethrough", "crossed", "underline", "highlight"]


# ── Helpers ───────────────────────────────────────────────────────────────────

_INK_TYPES = ["pencil", "pen", "marker", "highlighter"]


def _resolve_markup_ink(args):
    """Return a concrete ink type string, sampling by weight when ink='random'."""
    if args.annotations_markup_ink != "random":
        return args.annotations_markup_ink
    weights = np.array(args.annotations_markup_ink_weights, dtype=float)
    return _INK_TYPES[np.random.choice(len(weights), p=weights / weights.sum())]


def _resolve_markup_color(color_arg):
    """Resolve a per-ink color arg to a value accepted by Markup(markup_color=...).

    color_arg is a list from argparse nargs="+":
      ["random"] or ["contrast"]   → string passthrough to augraphy
      [255, 220, 0]                → fixed color, RGB→BGR converted to (0, 220, 255)
      [255,220,0, 255,170,0]       → sample uniformly from the two RGB triples, then BGR-convert

    Colors are specified as RGB in the config/CLI (natural for humans) and
    converted to BGR here because augraphy operates on OpenCV images.
    """
    if isinstance(color_arg[0], str) and not color_arg[0].lstrip("-").isdigit():
        return color_arg[0]
    ints = [int(v) for v in color_arg]
    triples = [tuple(ints[i:i + 3]) for i in range(0, len(ints), 3)]
    r, g, b = triples[np.random.randint(len(triples))]
    return (b, g, r)  # RGB → BGR



class RealisticMarkup(Markup):
    """Markup subclass with two fixes:

    1. distribute_line: configurable control-point count and y-jitter (instead of
       augraphy's hardcoded randint(3,6) points and offset=6 passed from __call__).
    2. __call__: corrected markup_length formula so values >1.0 center the extended
       annotation over the original text line, rather than shifting it off to the left.
    """

    def __init__(self, *args, control_points_range=(2, 3), line_offset=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_points_range = control_points_range
        self.line_offset = line_offset

    def distribute_line(self, starting_point, ending_point, offset):
        points_count = random.randint(self.control_points_range[0], self.control_points_range[1])
        points_x = np.linspace(starting_point[0], ending_point[0], points_count)
        points_y = [starting_point[1] + random.uniform(-self.line_offset, self.line_offset) for _ in points_x]
        return smooth(np.column_stack((points_x, points_y)).astype("float"), 6)

    def __call__(self, image, layer=None, mask=None, keypoints=None, bounding_boxes=None, force=False):
        # Verbatim copy of Markup.__call__ with one fix:
        # The markup_length x-shift formula is corrected to use original_w so that
        # markup_length > 1.0 centers the annotation instead of pushing it off-screen.

        has_alpha = 0
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            has_alpha = 1
            image, image_alpha = image[:, :, :3], image[:, :, 3]

        markup_image = image.copy()

        if self.markup_color == "random":
            markup_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif self.markup_color == "contrast":
            single_color = cv2.resize(image, (1, 1), interpolation=cv2.INTER_AREA)
            markup_color = 255 - single_color[0][0]
            markup_color = markup_color.tolist()
        else:
            markup_color = self.markup_color

        if self.large_word_mode == "random":
            large_word_mode = random.choice([True, False])
        else:
            large_word_mode = self.large_word_mode

        if self.markup_type == "random":
            markup_type = random.choice(["strikethrough", "crossed", "underline", "highlight"])
        else:
            markup_type = self.markup_type

        num_lines = random.randint(self.num_lines_range[0], self.num_lines_range[1])

        binary_image = self._preprocess(image)

        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        heights = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            heights.append(h)

        bins = np.unique(heights)
        hist, bin_edges = np.histogram(heights, bins=bins, density=False)
        if len(bin_edges) > 1 and np.max(hist) > 20:
            character_height_min = bin_edges[np.argmax(hist)]
            character_height_max = bin_edges[np.argmax(hist) + 1]
            character_height_average = int((character_height_max + character_height_min) / 2)
            height_range = ((character_height_max - character_height_min) / 2) + 1
        else:
            character_height_average = -1
            height_range = -1

        lines_coordinates = []

        if len(contours) > 0:
            contours = list(contours)
            random.shuffle(contours)

        for cnt in contours:
            choice = random.choice([False, True])
            x, y, w, h = cv2.boundingRect(cnt)

            if character_height_average == -1:
                check_height = h > 10
            else:
                check_height = (h > character_height_average - height_range) and (
                    h < character_height_average + height_range
                )

            if large_word_mode:
                # Add area and width guards to prevent page-spanning contours
                # (e.g. table borders, merged paragraph blocks) from being selected.
                # Limits are more lenient than non-large_word_mode (50% width vs 20%).
                conditions = (
                    check_height
                    and w * h < (markup_image.shape[0] * markup_image.shape[1]) / 10
                    and w < int(markup_image.shape[1] / 2)
                )
            else:
                conditions = (
                    choice
                    and (w > h * 2)
                    and (w * h < (markup_image.shape[0] * markup_image.shape[1]) / 10)
                    and w < int(markup_image.shape[1] / 5)
                    and check_height
                )

            if conditions:
                if num_lines == 0:
                    break
                num_lines = num_lines - 1
                markup_length = random.uniform(
                    self.markup_length_range[0],
                    self.markup_length_range[1],
                )
                # FIX: save original_w before scaling so the centering shift is correct
                # for markup_length > 1.0 (augraphy's original formula uses new w, which
                # pushes the annotation off-screen to the left for values > 1).
                original_w = w
                w = int(original_w * markup_length)
                x = int(x + (original_w - w) // 2)

                offset = 6

                if markup_type == "strikethrough" or markup_type == "highlight":
                    starting_point = [x, int(y + (h / 2))]
                    ending_point = [x + w, int(y + (h / 2))]
                elif markup_type == "crossed":
                    starting_point = [x, y]
                    ending_point = [x + w, y + h]
                else:
                    starting_point = [x, y + h]
                    ending_point = [x + w, y + h]

                for i in range(self.repetitions):
                    if markup_type == "crossed":
                        ysize, xsize = markup_image.shape[:2]
                        p1_x = np.clip(starting_point[0] + random.randint(-offset * 5, offset * 5), 0, xsize)
                        p1_y = np.clip(starting_point[1] + random.randint(-offset * 1, offset * 1), 0, ysize)
                        p2_x = np.clip(ending_point[0] + random.randint(-offset * 5, offset * 5), 0, xsize)
                        p2_y = np.clip(ending_point[1] + random.randint(-offset * 1, offset * 1), 0, ysize)
                        p1 = (p1_x, p1_y)
                        p2 = (p2_x, p2_y)
                        lines_coordinates.append(np.array([p1, p2]))
                        p1_x = np.clip(ending_point[0] + random.randint(-offset * 5, offset * 5), 0, xsize)
                        p1_y = np.clip(starting_point[1] + random.randint(-offset * 1, offset * 1), 0, ysize)
                        p2_x = np.clip(starting_point[0] + random.randint(-offset * 5, offset * 5), 0, xsize)
                        p2_y = np.clip(ending_point[1] + random.randint(-offset * 1, offset * 1), 0, ysize)
                        p1 = (p1_x, p1_y)
                        p2 = (p2_x, p2_y)
                        lines_coordinates.append(np.array([p1, p2]))
                    else:
                        points_list = self.distribute_line(
                            starting_point,
                            ending_point,
                            offset,
                        ).astype("int")
                        lines_coordinates.append(points_list)

        if lines_coordinates:
            if self.markup_ink == "random":
                markup_ink = random.choice(["pencil", "pen", "marker", "highlighter"])
            else:
                markup_ink = self.markup_ink

            if self.markup_type == "highlight":
                markup_thickness_range = (self.markup_thickness_range[0] + 5, self.markup_thickness_range[1] + 5)
            else:
                markup_thickness_range = self.markup_thickness_range

            from augraphy.utilities.inkgenerator import InkGenerator
            ink_generator = InkGenerator(
                ink_type=markup_ink,
                ink_draw_method="lines",
                ink_draw_iterations=(1, 1),
                ink_location="random",
                ink_background=markup_image,
                ink_background_size=None,
                ink_background_color=None,
                ink_color=markup_color,
                ink_min_brightness=1,
                ink_min_brightness_value_range=(150, 200),
                ink_draw_size_range=None,
                ink_thickness_range=markup_thickness_range,
                ink_brightness_change=[0],
                ink_skeletonize=0,
                ink_skeletonize_iterations_range=(1, 1),
                ink_text=None,
                ink_text_font=None,
                ink_text_rotate_range=None,
                ink_lines_coordinates=lines_coordinates,
                ink_lines_stroke_count_range=(1, 1),
            )
            markup_image = ink_generator.generate_ink()

        if has_alpha:
            markup_image = np.dstack((markup_image, image_alpha))

        outputs_extra = []
        if mask is not None or keypoints is not None or bounding_boxes is not None:
            outputs_extra = [mask, keypoints, bounding_boxes]

        if outputs_extra:
            return [markup_image] + outputs_extra
        else:
            return markup_image


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
            line_gradient_direction=tuple(args.lines_degradation_gradient_direction),
            line_split_probability=tuple(args.lines_degradation_split_probability),
            line_replacement_value=tuple(args.lines_degradation_replacement_value),
            line_replacement_probability=tuple(args.lines_degradation_replacement_probability),
            line_min_length=tuple(args.lines_degradation_min_length),
            line_long_to_short_ratio=tuple(args.lines_degradation_long_to_short_ratio),
            line_replacement_thickness=tuple(args.lines_degradation_replacement_thickness),
            p=args.lines_degradation_p,
        ))

    return phase


def _build_per_image_pipeline(args, ink_phase, base_paper_phase, post_phase):
    """Return a new AugraphyPipeline with a freshly sampled ColorPaper in the paper phase.

    Called once per augmented image when color_paper_page_sampling=1. ColorPaper is placed
    at the front of the paper phase so it only colours the blank paper canvas — before ink
    is composited — preserving hue of coloured document elements.
    """
    hue_ranges = list(zip(
        args.color_paper_profile_hue_ranges[0::2],
        args.color_paper_profile_hue_ranges[1::2],
    ))
    sat_ranges = list(zip(
        args.color_paper_profile_saturation_ranges[0::2],
        args.color_paper_profile_saturation_ranges[1::2],
    ))
    weights = np.array(args.color_paper_profile_weights, dtype=float)
    idx = np.random.choice(len(weights), p=weights / weights.sum())
    sampled_color_paper = ColorPaper(
        hue_range=hue_ranges[idx],
        saturation_range=sat_ranges[idx],
        p=args.color_paper_p,
    )
    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=[sampled_color_paper] + list(base_paper_phase),
        post_phase=post_phase,
        random_seed=None,  # do NOT reset global RNG — seeded once in main() via set_seed()
        log=False,
    )


def _apply_page_sampled_augmentations(img, args):
    """Apply per-image profile-sampled augmentations (post-pipeline).

    SubtleNoise: adds small uniform channel offsets, safe post-pipeline.
    Faxify: samples one of three discrete modes (mono-only, halftone-only, both).
    ColorPaper is handled separately via _build_per_image_pipeline (paper phase).
    When all profile_sampling flags are disabled this is a no-op.
    """
    result = img

    # NOTE: augraphy augmentations return None when their internal probability check fails.
    # All augmentations called here use p=1.0 and handle the _p probability manually so we
    # never assign None back to result.

    # Order: markup → subtle_noise → faxify
    # Markup detects text contours on the cleanest possible image.
    # SubtleNoise adds paper-level grain to the annotated document.
    # Faxify binarizes/halftones last so it does not destroy annotation colors before they are drawn.
    if args.annotations_markup and args.annotations_markup_sampling and args.annotations_p > 0:
        if np.random.random() < args.annotations_p:
            weights = np.array(args.annotations_markup_type_weights, dtype=float)
            markup_type = _MARKUP_TYPES[np.random.choice(len(weights), p=weights / weights.sum())]
            markup_ink = _resolve_markup_ink(args)
            color_args = {
                "pencil":      args.annotations_markup_pencil_colors,
                "pen":         args.annotations_markup_pen_colors,
                "marker":      args.annotations_markup_marker_colors,
                "highlighter": args.annotations_markup_highlighter_colors,
            }[markup_ink]
            thickness_range = {
                "pencil":      args.annotations_markup_pencil_thickness_range,
                "pen":         args.annotations_markup_pen_thickness_range,
                "marker":      args.annotations_markup_marker_thickness_range,
                "highlighter": args.annotations_markup_highlighter_thickness_range,
            }[markup_ink]
            _swm = args.annotations_markup_single_word_mode
            result = RealisticMarkup(
                num_lines_range=tuple(args.annotations_markup_num_lines_range),
                markup_length_range=tuple(args.annotations_markup_length_range),
                markup_thickness_range=tuple(thickness_range),
                markup_type=markup_type,
                markup_ink=markup_ink,
                markup_color=_resolve_markup_color(color_args),
                large_word_mode={-1: "random", 0: False, 1: True}[args.annotations_markup_large_word_mode],
                single_word_mode=(bool(np.random.randint(2)) if _swm == -1 else bool(_swm)),
                repetitions=np.random.randint(args.annotations_markup_repetitions[0], args.annotations_markup_repetitions[1] + 1),
                control_points_range=tuple(args.annotations_markup_control_points_range),
                line_offset=args.annotations_markup_line_offset,
                p=1.0,
            )(result)

    if args.subtle_noise_page_sampling and args.subtle_noise_p > 0:
        if np.random.random() < args.subtle_noise_p:
            weights = np.array(args.subtle_noise_profile_weights, dtype=float)
            idx = np.random.choice(len(weights), p=weights / weights.sum())
            result = SubtleNoise(
                subtle_range=args.subtle_noise_profile_ranges[idx],
                p=1.0,
            )(result)

    if args.faxify_profile_sampling and args.faxify_p > 0:
        if np.random.random() < args.faxify_p:
            # Profiles: 0=mono-only, 1=halftone-only, 2=both
            weights = np.array(args.faxify_profile_weights, dtype=float)
            idx = np.random.choice(len(weights), p=weights / weights.sum())
            mono = 1 if idx in (0, 2) else 0
            halftone = 1 if idx in (1, 2) else 0
            result = Faxify(
                scale_range=tuple(args.faxify_scale_range),
                monochrome=mono,
                monochrome_method=_resolve_monochrome_method(args),
                halftone=halftone,
                half_kernel_size=tuple(args.faxify_half_kernel_size),
                angle=tuple(args.faxify_angle),
                sigma=tuple(args.faxify_sigma),
                p=1.0,
            )(result)

    return result


def _build_paper_phase(args) -> list:
    phase = []

    if args.color_paper_p > 0 and not args.color_paper_page_sampling:
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


def _resolve_monochrome_method(args):
    methods = getattr(args, "faxify_monochrome_methods", [])
    if methods:
        return random.choice(methods)
    return args.faxify_monochrome_method


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
            _ksize_raw = tuple(args.scanner_noise_dirty_drum_ksize)
            _ksize = tuple(max(1, v if v % 2 == 1 else v + 1) for v in _ksize_raw)
            if _ksize != _ksize_raw:
                warnings.warn(
                    f"scanner_noise_dirty_drum_ksize {_ksize_raw} has even component(s); "
                    f"clamped to {_ksize} (cv2.GaussianBlur requires odd dimensions)",
                    UserWarning,
                )
            members.append(DirtyDrum(
                line_width_range=tuple(args.scanner_noise_dirty_drum_line_width),
                line_concentration=args.scanner_noise_dirty_drum_line_concentration,
                direction=args.scanner_noise_dirty_drum_direction,
                noise_intensity=args.scanner_noise_dirty_drum_noise_intensity,
                noise_value=tuple(args.scanner_noise_dirty_drum_noise_value),
                ksize=_ksize,
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
            transparency=args.lighting_gradient_transparency,
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

    if args.subtle_noise_p > 0 and not args.subtle_noise_page_sampling:
        phase.append(SubtleNoise(subtle_range=args.subtle_noise_range, p=args.subtle_noise_p))

    if args.jpeg_p > 0:
        phase.append(Jpeg(quality_range=tuple(args.jpeg_quality_range), p=args.jpeg_p))

    if args.folding_p > 0:
        phase.append(Folding(
            fold_count=args.folding_fold_count,
            fold_noise=args.folding_fold_noise,
            fold_angle_range=tuple(args.folding_fold_angle_range),
            fold_deviation=tuple(args.folding_fold_deviation),
            gradient_width=tuple(args.folding_gradient_width),
            gradient_height=tuple(args.folding_gradient_height),
            backdrop_color=tuple(args.folding_backdrop_color),
            p=args.folding_p,
        ))

    if args.page_border_p > 0:
        phase.append(PageBorder(
            page_rotation_angle_range=tuple(args.page_border_rotation_range),
            page_rotate_angle_in_order=args.page_border_rotate_in_order,
            page_border_color=tuple(args.page_border_color),
            curve_frequency=tuple(args.page_border_curve_frequency),
            curve_height=tuple(args.page_border_curve_height),
            curve_length_one_side=tuple(args.page_border_curve_length),
            same_page_border=1,
            p=args.page_border_p,
        ))

    if args.annotations_p > 0:
        # Markup and Scribbles fire independently so a Markup detection failure
        # doesn't silently produce a blank result.
        # When markup_sampling=1, Markup is suppressed here and applied manually
        # in _apply_page_sampled_augmentations so per-ink color can be resolved per image.
        if args.annotations_markup and not args.annotations_markup_sampling:
            _swm = args.annotations_markup_single_word_mode
            _ink = _resolve_markup_ink(args)
            _color_args = {
                "pencil":      args.annotations_markup_pencil_colors,
                "pen":         args.annotations_markup_pen_colors,
                "marker":      args.annotations_markup_marker_colors,
                "highlighter": args.annotations_markup_highlighter_colors,
            }[_ink]
            _thickness_range = {
                "pencil":      args.annotations_markup_pencil_thickness_range,
                "pen":         args.annotations_markup_pen_thickness_range,
                "marker":      args.annotations_markup_marker_thickness_range,
                "highlighter": args.annotations_markup_highlighter_thickness_range,
            }[_ink]
            phase.append(RealisticMarkup(
                num_lines_range=tuple(args.annotations_markup_num_lines_range),
                markup_length_range=tuple(args.annotations_markup_length_range),
                markup_thickness_range=tuple(_thickness_range),
                markup_type=args.annotations_markup_type,
                markup_ink=_ink,
                markup_color=_resolve_markup_color(_color_args),
                large_word_mode={-1: "random", 0: False, 1: True}[args.annotations_markup_large_word_mode],
                single_word_mode=(bool(np.random.randint(2)) if _swm == -1 else bool(_swm)),
                repetitions=np.random.randint(args.annotations_markup_repetitions[0], args.annotations_markup_repetitions[1] + 1),
                control_points_range=tuple(args.annotations_markup_control_points_range),
                line_offset=args.annotations_markup_line_offset,
                p=args.annotations_p,
            ))
        if args.annotations_scribbles:
            phase.append(Scribbles(
                scribbles_type="random",
                scribbles_ink="random",
                scribbles_size_range=tuple(args.annotations_scribbles_size_range),
                scribbles_count_range=tuple(args.annotations_scribbles_count_range),
                scribbles_thickness_range=tuple(args.annotations_scribbles_thickness_range),
                p=args.annotations_p,
            ))

    if args.faxify_p > 0 and not args.faxify_profile_sampling:
        phase.append(Faxify(
            scale_range=tuple(args.faxify_scale_range),
            monochrome=args.faxify_monochrome,
            monochrome_method=_resolve_monochrome_method(args),
            halftone=args.faxify_halftone,
            half_kernel_size=tuple(args.faxify_half_kernel_size),
            angle=tuple(args.faxify_angle),
            sigma=tuple(args.faxify_sigma),
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

def augment_pdf(pdf_path: Path, output_dir: Path, pipeline, args, num_augmentations: int, dpi: int, logger, flat: bool = False) -> int:
    """Augment all pages of a single PDF. Returns total images saved."""
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    saved = 0
    out_subdir = output_dir if flat else output_dir / pdf_path.stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Pre-build reusable phases for per-image pipeline construction (color_paper page sampling).
    # ColorPaper is sampled once per augmented image and placed in the paper phase so it
    # only colours the blank canvas — before ink compositing.
    _use_per_image_pipeline = args.color_paper_page_sampling and args.color_paper_p > 0
    if _use_per_image_pipeline:
        _ink_phase = _build_ink_phase(args)
        _base_paper_phase = _build_paper_phase(args)  # ColorPaper already excluded
        _post_phase = _build_post_phase(args)

    for page_idx, pil_page in enumerate(pages):
        img = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
        for aug_idx in range(num_augmentations):
            if _use_per_image_pipeline:
                aug_pipeline = _build_per_image_pipeline(args, _ink_phase, _base_paper_phase, _post_phase)
            else:
                aug_pipeline = pipeline
            augmented = aug_pipeline(img)
            augmented = _apply_page_sampled_augmentations(augmented, args)
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
    parser.add_argument("--lines_degradation_gradient_direction", type=int, nargs=2, default=[0, 2])
    parser.add_argument("--lines_degradation_split_probability", type=float, nargs=2, default=[0.2, 0.4])
    parser.add_argument("--lines_degradation_replacement_value", type=int, nargs=2, default=[245, 255])
    parser.add_argument("--lines_degradation_replacement_probability", type=float, nargs=2, default=[0.3, 0.5])
    parser.add_argument("--lines_degradation_min_length", type=int, nargs=2, default=[30, 40])
    parser.add_argument("--lines_degradation_long_to_short_ratio", type=int, nargs=2, default=[5, 7])
    parser.add_argument("--lines_degradation_replacement_thickness", type=int, nargs=2, default=[1, 3])

    # ── Paper phase ──────────────────────────────────────────────────────────

    # color_paper
    parser.add_argument("--color_paper_p", type=float, default=0.7)
    parser.add_argument("--color_paper_hue_range", type=int, nargs=2, default=[20, 45])
    parser.add_argument("--color_paper_saturation_range", type=int, nargs=2, default=[10, 35])
    parser.add_argument("--color_paper_page_sampling", type=int, default=0)
    parser.add_argument("--color_paper_num_profiles", type=int, default=1)
    parser.add_argument("--color_paper_profile_weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--color_paper_profile_hue_ranges", type=int, nargs="+", default=[20, 45])
    parser.add_argument("--color_paper_profile_saturation_ranges", type=int, nargs="+", default=[10, 35])

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
    parser.add_argument("--lighting_gradient_transparency", type=float, default=None)

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
    parser.add_argument("--subtle_noise_page_sampling", type=int, default=0)
    parser.add_argument("--subtle_noise_num_profiles", type=int, default=1)
    parser.add_argument("--subtle_noise_profile_weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--subtle_noise_profile_ranges", type=int, nargs="+", default=[12])

    # jpeg
    parser.add_argument("--jpeg_p", type=float, default=0.4)
    parser.add_argument("--jpeg_quality_range", type=int, nargs=2, default=[55, 92])

    # folding
    parser.add_argument("--folding_p", type=float, default=0.25)
    parser.add_argument("--folding_fold_count", type=int, default=2)
    parser.add_argument("--folding_fold_noise", type=float, default=0.01)
    parser.add_argument("--folding_fold_angle_range", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--folding_fold_deviation", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--folding_gradient_width", type=float, nargs=2, default=[0.1, 0.2])
    parser.add_argument("--folding_gradient_height", type=float, nargs=2, default=[0.01, 0.02])
    parser.add_argument("--folding_backdrop_color", type=int, nargs=3, default=[255, 255, 255])

    # page_border
    parser.add_argument("--page_border_p", type=float, default=0.3)
    parser.add_argument("--page_border_rotation_range", type=float, nargs=2, default=[-2.0, 2.0])
    parser.add_argument("--page_border_rotate_in_order", type=int, default=1)
    parser.add_argument("--page_border_color", type=int, nargs=3, default=[0, 0, 0])
    parser.add_argument("--page_border_curve_frequency", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--page_border_curve_height", type=int, nargs=2, default=[2, 4])
    parser.add_argument("--page_border_curve_length", type=int, nargs=2, default=[50, 100])

    # annotations
    parser.add_argument("--annotations_p", type=float, default=0.3)
    parser.add_argument("--annotations_markup", type=int, default=1)
    parser.add_argument("--annotations_markup_type", type=str, default="random")
    parser.add_argument("--annotations_markup_num_lines_range", type=int, nargs=2, default=[1, 4])
    parser.add_argument("--annotations_markup_length_range", type=float, nargs=2, default=[0.5, 1.0])
    parser.add_argument("--annotations_markup_pencil_thickness_range",      type=int, nargs=2, default=[1, 3])
    parser.add_argument("--annotations_markup_pen_thickness_range",         type=int, nargs=2, default=[1, 3])
    parser.add_argument("--annotations_markup_marker_thickness_range",      type=int, nargs=2, default=[1, 3])
    parser.add_argument("--annotations_markup_highlighter_thickness_range", type=int, nargs=2, default=[1, 3])
    parser.add_argument("--annotations_markup_ink", type=str, default="random")
    # Weights for [pencil, pen, marker, highlighter] when markup_ink="random" — sampled per image
    parser.add_argument("--annotations_markup_ink_weights", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25])
    # Controls which contours Markup considers eligible for marking.
    # -1 = library default ("random" — 50% chance of silent no-op on full-width text lines)
    #  0 = strict mode: contour must be narrow (w < image_width/5) — rarely useful for PDFs
    #  1 = lenient mode: any contour that passes the height check qualifies (recommended)
    parser.add_argument("--annotations_markup_large_word_mode", type=int, default=1)
    # -1=random (coin flip per image), 0=False (line-level, default): dilation kernel (20,1) merges words into full lines
    # 1=True (word-level): dilation kernel (10,1) marks individual words; forces markup_length_range=(1,1)
    parser.add_argument("--annotations_markup_single_word_mode", type=int, default=0)
    # Range [min, max] of strokes drawn per qualifying contour — sampled uniformly per image
    parser.add_argument("--annotations_markup_repetitions", type=int, nargs=2, default=[1, 1])
    # markup_sampling=1: suppress Markup from phase, sample type+color manually per image
    parser.add_argument("--annotations_markup_sampling", type=int, default=0)
    # Weights for [strikethrough, crossed, underline, highlight] — must have exactly 4 values
    parser.add_argument("--annotations_markup_type_weights", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25])
    # Per-ink color: "random"|"contrast" string, or flat RGB ints (3=fixed, 6+=sample equally)
    parser.add_argument("--annotations_markup_pencil_colors",      nargs="+", default=["random"])
    parser.add_argument("--annotations_markup_pen_colors",         nargs="+", default=["random"])
    parser.add_argument("--annotations_markup_marker_colors",      nargs="+", default=["random"])
    parser.add_argument("--annotations_markup_highlighter_colors", nargs="+", default=["random"])
    # Control point count per stroke [min, max] — fewer points = straighter lines
    parser.add_argument("--annotations_markup_control_points_range", type=int, nargs=2, default=[2, 3])
    # Y-jitter magnitude (pixels) applied to each control point — lower = straighter lines
    parser.add_argument("--annotations_markup_line_offset", type=int, default=3)
    parser.add_argument("--annotations_scribbles", type=int, default=1)
    parser.add_argument("--annotations_scribbles_size_range", type=int, nargs=2, default=[400, 600])
    parser.add_argument("--annotations_scribbles_count_range", type=int, nargs=2, default=[1, 3])
    parser.add_argument("--annotations_scribbles_thickness_range", type=int, nargs=2, default=[1, 2])

    # faxify (disabled by default — set faxify_p > 0 to enable)
    parser.add_argument("--faxify_p", type=float, default=0.0)
    parser.add_argument("--faxify_scale_range", type=float, nargs=2, default=[1.0, 1.5])
    parser.add_argument("--faxify_monochrome", type=int, default=-1)
    parser.add_argument("--faxify_monochrome_method", type=str, default="random")
    # Explicit inclusion list; when non-empty, overrides faxify_monochrome_method with a uniform random pick
    parser.add_argument("--faxify_monochrome_methods", type=str, nargs="+", default=[])
    parser.add_argument("--faxify_halftone", type=int, default=-1)
    parser.add_argument("--faxify_half_kernel_size", type=int, nargs=2, default=[1, 1])
    parser.add_argument("--faxify_angle", type=int, nargs=2, default=[0, 360])
    parser.add_argument("--faxify_sigma", type=float, nargs=2, default=[1.0, 3.0])
    parser.add_argument("--faxify_profile_sampling", type=int, default=0)
    parser.add_argument("--faxify_profile_weights", type=float, nargs=3, default=[0.33, 0.34, 0.33])

    args = parser.parse_args()

    logger = init_logger(__name__, level=args.log_level)
    set_seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / args.run_name
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
        saved = augment_pdf(pdf_path, output_dir, pipeline, args, args.num_augmentations, args.dpi, logger, flat=args.flat)
        logger.info(f"  -> {saved} image(s) saved")
        total_saved += saved

    logger.info(f"Done. Total images saved: {total_saved}")


if __name__ == "__main__":
    main()
