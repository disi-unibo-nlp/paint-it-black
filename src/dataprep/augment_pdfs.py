"""
augment_dataset.py — Apply document augmentations to a folder of PDFs or an HF Dataset split.

Dir mode (original):
    For each PDF in input_dir, each page is converted to an image and augmented
    num_augmentations times. Results are saved as PNGs under output_dir.

    Output structure:
        output_dir/<run_name>/<pdf_stem>/page_<N>_aug_<M>.png

    Usage:
        python3 src/dataprep/augment_dataset.py --config dataprep/augment_config
        python3 src/dataprep/augment_dataset.py --config dataprep/augment_config --geometric_p 0.0

Dataset mode (HF Dataset):
    Reads images from an HF Dataset split (local or Hub), augments each row once
    (num_augmentations is forced to 1), and writes the result as a new split.

    Usage:
        python3 src/dataprep/augment_dataset.py --config dataprep/low_noise_mix \\
            --input_dataset data/my_ds --input_split base --output_split augmented
        # Re-sample specific rows (e.g. after inspecting bad results):
        python3 src/dataprep/augment_dataset.py --config dataprep/low_noise_mix --seed 99 \\
            --input_dataset data/my_ds --input_split base \\
            --output_dataset data/my_ds --output_split augmented --resample_ids 5 12 47

Every augmentation parameter is individually settable from the config file or CLI.
Set any <aug>_p to 0.0 to disable that augmentation entirely.
Set any OneOf membership flag (e.g. scanner_noise_dirty_drum) to 0 to exclude it from the pool.
Different presets are different config files — there is no --pipeline choice.
"""

import copy
import random
import warnings
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datasets import Image as HFImage
from huggingface_hub import login
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

# Augraphy bug: InkGenerator.apply_highlighter_effect builds std_range with np.ceil,
# which returns numpy.float64, then passes it to random.randint which requires int.
from augraphy.utilities import inkgenerator as _ig
_orig_generate_noise_clusters = _ig.InkGenerator.generate_noise_clusters
def _patched_generate_noise_clusters(self, image, n_clusters=(200, 200), n_samples=(300, 300), std_range=(5, 10)):
    std_range = (int(std_range[0]), int(std_range[1]))
    return _orig_generate_noise_clusters(self, image, n_clusters, n_samples, std_range)
_ig.InkGenerator.generate_noise_clusters = _patched_generate_noise_clusters

# Augraphy bug: NoiseGenerator.generate_worley_noise uses sorted() inside nb.prange,
# which breaks numba parallel type inference on this numba version (AssertionError in
# typeinfer.py). Remap noise_type 4 (worley) to 1 (blobs) in generate_mask_main so the
# broken JIT function is never called. Random selection (-1) still yields 4 types instead
# of 5, which is acceptable.
from augraphy.utilities import noisegenerator as _ng
_orig_generate_mask_main = _ng.NoiseGenerator.generate_mask_main
def _patched_generate_mask_main(self, noise_type, noise_side, noise_value, noise_background,
                                noise_sparsity, noise_concentration, xsize, ysize):
    if noise_type == 4:
        noise_type = 1
    return _orig_generate_mask_main(self, noise_type, noise_side, noise_value, noise_background,
                                    noise_sparsity, noise_concentration, xsize, ysize)
_ng.NoiseGenerator.generate_mask_main = _patched_generate_mask_main


# ── Constants ────────────────────────────────────────────────────────────────

_MARKUP_TYPES = ["strikethrough", "crossed", "underline", "highlight"]


# ── Helpers ───────────────────────────────────────────────────────────────────

_INK_TYPES = ["pencil", "pen", "marker", "highlighter"]

# Augmentations suppressed when faxify fires (profile-sampling mutex).
# Only active when faxify_profile_sampling=1 and faxify_p > 0.
_FAXIFY_MUTEX_P_ATTRS = ["scanner_noise_p", "shadow_cast_p", "stains_p"]


def _maybe_downscale(pil_img: Image.Image, scale: float) -> Image.Image:
    if scale >= 1.0:
        return pil_img
    w, h = pil_img.size
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


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

    def precompute_lines(self, image):
        """Detect text lines on a (clean) image; return coordinate arrays for draw_precomputed.

        Designed to be called on the original document image before any augmentation so
        that line placement is determined from clean text contours, not from ink-bled,
        stain-covered, or noise-degraded versions of the page.

        Resolves markup_color and markup_type once and stores them on self
        (_precomputed_markup_color, _precomputed_markup_type) so draw_precomputed can
        use the same values.

        Returns a list of coordinate arrays (empty list if no qualifying contours found).
        """
        # Resolve markup_color from the clean image ("contrast" uses original background)
        if self.markup_color == "random":
            self._precomputed_markup_color = (
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
        elif self.markup_color == "contrast":
            single_color = cv2.resize(image, (1, 1), interpolation=cv2.INTER_AREA)
            self._precomputed_markup_color = (255 - single_color[0][0]).tolist()
        else:
            self._precomputed_markup_color = self.markup_color

        # Resolve markup_type (may be "random" when called from non-sampling path)
        if self.markup_type == "random":
            self._precomputed_markup_type = random.choice(
                ["strikethrough", "crossed", "underline", "highlight"]
            )
        else:
            self._precomputed_markup_type = self.markup_type

        if self.large_word_mode == "random":
            large_word_mode = random.choice([True, False])
        else:
            large_word_mode = self.large_word_mode

        num_lines = random.randint(self.num_lines_range[0], self.num_lines_range[1])
        markup_type = self._precomputed_markup_type

        # Normalise image to BGR 3-channel for _preprocess
        img_bgr = image
        if len(image.shape) < 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            img_bgr = image[:, :, :3]

        binary_image = self._preprocess(img_bgr)

        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        heights = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            heights.append(h)

        if not heights or len(np.unique(heights)) < 2:
            character_height_average = -1
            height_range = -1
        else:
            bins = np.unique(heights)
            hist, bin_edges = np.histogram(heights, bins=bins, density=False)
            if len(bin_edges) > 1 and np.max(hist) > 20:
                character_height_min = bin_edges[np.argmax(hist)]
                character_height_max = bin_edges[np.argmax(hist) + 1]
                character_height_average = int((character_height_max + character_height_min) / 2)
                height_range = ((character_height_max - character_height_min) / 2) + 1
                # Clean images produce many tiny binary artefacts (serifs, noise, 1-3px dots)
                # that dominate the histogram with h=2-5, making the height window (0, 6)
                # which rejects all actual text line contours.  When the modal height is
                # clearly sub-character, discard the histogram estimate and fall back to the
                # simple h>10 check so real text contours are not filtered out.
                if character_height_average < 10:
                    character_height_average = -1
                    height_range = -1
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
                conditions = (
                    check_height
                    and w * h < (img_bgr.shape[0] * img_bgr.shape[1]) / 10
                    and w < int(img_bgr.shape[1] / 2)
                )
            else:
                conditions = (
                    choice
                    and (w > h * 2)
                    and (w * h < (img_bgr.shape[0] * img_bgr.shape[1]) / 10)
                    and w < int(img_bgr.shape[1] / 5)
                    and check_height
                )

            if conditions:
                if num_lines == 0:
                    break
                num_lines = num_lines - 1
                markup_length = random.uniform(self.markup_length_range[0], self.markup_length_range[1])
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
                        ysize, xsize = img_bgr.shape[:2]
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

        return lines_coordinates

    def draw_precomputed(self, image, lines_coordinates):
        """Draw annotations on image using coordinates pre-computed by precompute_lines.

        Uses self._precomputed_markup_color and self._precomputed_markup_type so that
        color and type are consistent with the detection step.

        ink_min_brightness=0 disables the brightness floor that was washing out dark
        ink colours (e.g. dark blue pen) to near-white in the original __call__.

        Returns image unchanged when lines_coordinates is empty.
        """
        if not lines_coordinates:
            return image

        has_alpha = 0
        if len(image.shape) < 3:
            markup_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            has_alpha = 1
            markup_image = image[:, :, :3].copy()
            image_alpha = image[:, :, 3]
        else:
            markup_image = image.copy()

        if self.markup_ink == "random":
            markup_ink = random.choice(["pencil", "pen", "marker", "highlighter"])
        else:
            markup_ink = self.markup_ink

        markup_color = getattr(self, "_precomputed_markup_color", self.markup_color)
        markup_type = getattr(self, "_precomputed_markup_type", self.markup_type)

        if markup_type == "highlight":
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
            ink_min_brightness=0,  # disabled: was brightening dark ink (e.g. blue) to near-white
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


def _apply_page_sampled_augmentations(img, args, faxify_pre_rolled=None, precomputed_markup=None):
    """Apply per-image profile-sampled augmentations (post-pipeline).

    SubtleNoise: adds small uniform channel offsets, safe post-pipeline.
    Faxify: samples one of three discrete modes (mono-only, halftone-only, both).
    ColorPaper is handled separately via _build_per_image_pipeline (paper phase).
    When all profile_sampling flags are disabled this is a no-op.

    precomputed_markup: optional (markup_obj, lines_coordinates) tuple produced by
    augment_pdf before the pipeline runs. When supplied, draw_precomputed is called
    with the pre-detected coordinates from the original clean image, bypassing
    in-place contour detection on the (potentially degraded) augmented image.
    """
    result = img

    # NOTE: augraphy augmentations return None when their internal probability check fails.
    # All augmentations called here use p=1.0 and handle the _p probability manually so we
    # never assign None back to result.

    # Order: markup → subtle_noise → faxify
    # Markup draws on the cleanest image available before noise/binarization.
    # SubtleNoise adds paper-level grain to the annotated document.
    # Faxify binarizes/halftones last so it does not destroy annotation colors before they are drawn.
    if precomputed_markup is not None:
        # Primary path: augment_pdf set precomputed_markup.
        # Non-empty tuple → probability passed; draw using pre-detected coordinates from the clean image.
        # Empty tuple ()  → probability failed at augment_pdf level; skip to avoid a second roll below.
        if precomputed_markup:
            _markup_obj, _lines_coords = precomputed_markup
            result = _markup_obj.draw_precomputed(result, _lines_coords)
    elif args.annotations_markup and args.annotations_markup_sampling and args.annotations_p > 0:
        # Fallback: called standalone (not from augment_pdf); precomputed_markup was never set.
        # Detects and draws on whatever image arrives — appropriate for direct/test invocations.
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
        fire = faxify_pre_rolled if faxify_pre_rolled is not None else (np.random.random() < args.faxify_p)
        if fire:
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

    if getattr(args, "stains_profile_sampling", 0) and args.stains_p > 0:
        # Faxify mutex: suppress stains when faxify fired this iteration.
        # faxify_pre_rolled=True → faxify fired; False → faxify didn't fire; None → mutex inactive.
        if not faxify_pre_rolled:
            if np.random.random() < args.stains_p:
                weights = np.array(args.stains_profile_weights, dtype=float)
                idx = np.random.choice(len(weights), p=weights / weights.sum())
                result = Stains(
                    stains_type=args.stains_profile_types[idx],
                    stains_blend_method=args.stains_profile_blend_methods[idx],
                    stains_blend_alpha=args.stains_profile_blend_alphas[idx],
                    p=1.0,
                )(result)

    return result


def _augment_single_image(
    img_bgr,
    args,
    pipeline,
    use_dynamic_pipeline,
    faxify_mutex_active,
    ink_phase=None,
    base_paper_phase=None,
    post_phase=None,
    base_paper_phase_no_mutex=None,
    post_phase_no_mutex=None,
):
    """Augment a single BGR image. Returns an augmented BGR numpy array.

    Shared by augment_pdf (dir mode) and augment_dataset_split (HF dataset mode)
    so the two entry points cannot diverge silently.
    """
    faxify_pre_rolled = None
    if faxify_mutex_active:
        faxify_pre_rolled = np.random.random() < args.faxify_p

    precomputed_markup = None
    if args.annotations_markup and args.annotations_markup_sampling and args.annotations_p > 0:
        if np.random.random() < args.annotations_p:
            _weights = np.array(args.annotations_markup_type_weights, dtype=float)
            _markup_type = _MARKUP_TYPES[np.random.choice(len(_weights), p=_weights / _weights.sum())]
            _markup_ink = _resolve_markup_ink(args)
            _color_args = {
                "pencil":      args.annotations_markup_pencil_colors,
                "pen":         args.annotations_markup_pen_colors,
                "marker":      args.annotations_markup_marker_colors,
                "highlighter": args.annotations_markup_highlighter_colors,
            }[_markup_ink]
            _thickness_range = {
                "pencil":      args.annotations_markup_pencil_thickness_range,
                "pen":         args.annotations_markup_pen_thickness_range,
                "marker":      args.annotations_markup_marker_thickness_range,
                "highlighter": args.annotations_markup_highlighter_thickness_range,
            }[_markup_ink]
            _swm = args.annotations_markup_single_word_mode
            _markup_obj = RealisticMarkup(
                num_lines_range=tuple(args.annotations_markup_num_lines_range),
                markup_length_range=tuple(args.annotations_markup_length_range),
                markup_thickness_range=tuple(_thickness_range),
                markup_type=_markup_type,
                markup_ink=_markup_ink,
                markup_color=_resolve_markup_color(_color_args),
                large_word_mode={-1: "random", 0: False, 1: True}[args.annotations_markup_large_word_mode],
                single_word_mode=(bool(np.random.randint(2)) if _swm == -1 else bool(_swm)),
                repetitions=np.random.randint(
                    args.annotations_markup_repetitions[0],
                    args.annotations_markup_repetitions[1] + 1,
                ),
                control_points_range=tuple(args.annotations_markup_control_points_range),
                line_offset=args.annotations_markup_line_offset,
                p=1.0,
            )
            precomputed_markup = (_markup_obj, _markup_obj.precompute_lines(img_bgr))
        else:
            # Probability failed: use empty tuple sentinel so the fallback detection
            # path in _apply_page_sampled_augmentations is not triggered.
            precomputed_markup = ()

    if use_dynamic_pipeline:
        active_paper = base_paper_phase_no_mutex if faxify_pre_rolled else base_paper_phase
        active_post  = post_phase_no_mutex       if faxify_pre_rolled else post_phase
        aug_pipeline = _build_per_image_pipeline(args, ink_phase, active_paper, active_post)
    else:
        aug_pipeline = pipeline

    augmented = aug_pipeline(img_bgr)
    augmented = _apply_page_sampled_augmentations(
        augmented, args,
        faxify_pre_rolled=faxify_pre_rolled,
        precomputed_markup=precomputed_markup,
    )
    return augmented


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

    if args.stains_p > 0 and not getattr(args, "stains_profile_sampling", 0):
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


def _build_post_phase_no_mutex(args) -> list:
    """Build post phase with faxify-mutex augmentations disabled.

    When faxify fires and faxify_profile_sampling=1, scanner_noise, shadow_cast,
    and stains are suppressed for that image's pipeline so they never co-occur with
    faxify (which tends to make text unreadable when combined).
    """
    args_no_mutex = copy.copy(args)
    for attr in _FAXIFY_MUTEX_P_ATTRS:
        setattr(args_no_mutex, attr, 0.0)
    return _build_post_phase(args_no_mutex)


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
            shadow_color=tuple(args.shadow_cast_color) if args.shadow_cast_color != "random" else "random",
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

    # Pre-build reusable phases for per-image pipeline construction.
    # Two flags can trigger dynamic (per-image) pipeline building:
    #   color_paper_page_sampling: ColorPaper profile is sampled once per image.
    #   faxify_mutex_active: when faxify fires, scanner_noise/shadow_cast/stains are suppressed
    #     by swapping in a stripped-down post phase built without those augmentations.
    _use_per_image_pipeline = args.color_paper_page_sampling and args.color_paper_p > 0
    _faxify_mutex_active = bool(getattr(args, "faxify_profile_sampling", 0)) and args.faxify_p > 0
    _use_dynamic_pipeline = _use_per_image_pipeline or _faxify_mutex_active

    _ink_phase = _base_paper_phase = _post_phase = None
    _base_paper_phase_no_mutex = _post_phase_no_mutex = None

    if _use_dynamic_pipeline:
        _ink_phase = _build_ink_phase(args)
        _base_paper_phase = _build_paper_phase(args)  # ColorPaper already excluded
        _post_phase = _build_post_phase(args)
        if _faxify_mutex_active:
            # Build no-mutex variants for both phases: stains lives in the paper phase,
            # scanner_noise and shadow_cast live in the post phase.
            _args_no_mutex = copy.copy(args)
            for _attr in _FAXIFY_MUTEX_P_ATTRS:
                setattr(_args_no_mutex, _attr, 0.0)
            _base_paper_phase_no_mutex = _build_paper_phase(_args_no_mutex)
            _post_phase_no_mutex = _build_post_phase(_args_no_mutex)

    for page_idx, pil_page in enumerate(pages):
        img = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
        for aug_idx in range(num_augmentations):
            augmented = _augment_single_image(
                img, args, pipeline,
                _use_dynamic_pipeline, _faxify_mutex_active,
                _ink_phase, _base_paper_phase, _post_phase,
                _base_paper_phase_no_mutex if _faxify_mutex_active else None,
                _post_phase_no_mutex if _faxify_mutex_active else None,
            )
            out_path = out_subdir / f"page_{page_idx:03d}_aug_{aug_idx:03d}.png"
            cv2.imwrite(str(out_path), augmented)
            saved += 1
            logger.debug(f"  Saved {out_path.name}")

    return saved


# ── HF Dataset mode ──────────────────────────────────────────────────────────

def augment_dataset_split(
    source_dataset,
    args,
    pipeline,
    logger,
    resample_indices=None,
    existing_dataset=None,
):
    """Augment an HF Dataset split, producing exactly one augmented row per source row.

    source_dataset:   the clean base Dataset to read images from.
    resample_indices: if given (list/set of int row indices), only those rows are
                      re-augmented from source_dataset and merged into existing_dataset.
                      All other rows are taken unchanged from existing_dataset.
    existing_dataset: required when resample_indices is not None.
    """
    _use_per_image_pipeline = args.color_paper_page_sampling and args.color_paper_p > 0
    _faxify_mutex_active = bool(getattr(args, "faxify_profile_sampling", 0)) and args.faxify_p > 0
    _use_dynamic_pipeline = _use_per_image_pipeline or _faxify_mutex_active

    _ink_phase = _base_paper_phase = _post_phase = None
    _base_paper_phase_no_mutex = _post_phase_no_mutex = None

    if _use_dynamic_pipeline:
        _ink_phase = _build_ink_phase(args)
        _base_paper_phase = _build_paper_phase(args)
        _post_phase = _build_post_phase(args)
        if _faxify_mutex_active:
            _args_no_mutex = copy.copy(args)
            for _attr in _FAXIFY_MUTEX_P_ATTRS:
                setattr(_args_no_mutex, _attr, 0.0)
            _base_paper_phase_no_mutex = _build_paper_phase(_args_no_mutex)
            _post_phase_no_mutex = _build_post_phase(_args_no_mutex)

    resample_set = set(resample_indices) if resample_indices is not None else None
    n = len(source_dataset)
    rows = []

    for i in range(n):
        if resample_set is not None and i not in resample_set:
            # Keep existing augmented row unchanged — do NOT apply output_scale here,
            # the existing image is already at the correct scale from its original augmentation.
            existing_row = existing_dataset[i]
            rows.append({
                **{k: existing_row[k] for k in existing_row if k != "image"},
                "image": existing_row["image"],
            })
            continue

        row = source_dataset[i]
        img_bgr = cv2.cvtColor(np.array(row["image"]), cv2.COLOR_RGB2BGR)

        augmented = _augment_single_image(
            img_bgr, args, pipeline,
            _use_dynamic_pipeline, _faxify_mutex_active,
            _ink_phase, _base_paper_phase, _post_phase,
            _base_paper_phase_no_mutex if _faxify_mutex_active else None,
            _post_phase_no_mutex if _faxify_mutex_active else None,
        )

        pil_aug = _maybe_downscale(Image.fromarray(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)), args.output_scale)
        rows.append({
            **{k: row[k] for k in row if k != "image"},
            "image": pil_aug,
        })
        logger.debug(f"  Augmented row {i}")

    n_augmented = n if resample_set is None else len(resample_set)
    logger.info(f"Augmented {n_augmented} row(s); output split has {n} rows total")
    return Dataset.from_list(rows).cast_column("image", HFImage())


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
    parser.add_argument("--stains_profile_sampling", type=int, default=0)
    parser.add_argument("--stains_num_profiles", type=int, default=1)
    parser.add_argument("--stains_profile_weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--stains_profile_types", type=str, nargs="+", default=["random"])
    parser.add_argument("--stains_profile_blend_methods", type=str, nargs="+", default=["darken"])
    parser.add_argument("--stains_profile_blend_alphas", type=float, nargs="+", default=[0.4])

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
    parser.add_argument("--shadow_cast_color", nargs="+", default=[0, 0, 0],
                        help="BGR color of the shadow, or 'random'. E.g. [0,0,0] for black, [30,30,50] for dark brownish.")

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

    # ── HF Dataset mode ──────────────────────────────────────────────────────
    parser.add_argument("--input_dataset", type=str, default=None,
                        help="Local path or HF repo ID of the source dataset (activates dataset mode)")
    parser.add_argument("--input_split", type=str, default="base",
                        help="Split name to read from the source dataset")
    parser.add_argument("--output_dataset", type=str, default=None,
                        help="Local path or HF repo ID for the output dataset (defaults to --input_dataset)")
    parser.add_argument("--output_split", type=str, default=None,
                        help="Split name for the output dataset (required in dataset mode)")
    parser.add_argument("--resample_ids", type=int, nargs="+", default=None,
                        help="Row indices (0-based) in the output split to re-augment and replace")
    parser.add_argument("--output_scale", type=float, default=1.0,
                        help="Downscale factor applied to all images before saving (e.g. 0.667 = 200/300 DPI). Default 1.0 = no resize.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the output dataset to the HF Hub (dataset mode only)")

    args = parser.parse_args()

    logger = init_logger(__name__, level=args.log_level)
    set_seed(args.seed)

    # ── Dataset mode ─────────────────────────────────────────────────────────
    if args.input_dataset is not None:
        if args.output_split is None:
            logger.error("--output_split is required in dataset mode")
            return

        if args.num_augmentations != 1:
            logger.warning(
                f"--num_augmentations={args.num_augmentations} is ignored in dataset mode "
                "(exactly one augmented image is produced per source row)"
            )

        output_dataset_path = args.output_dataset or args.input_dataset

        # Load source split (local path or HF Hub)
        source_dir = Path(args.input_dataset) / args.input_split
        if source_dir.is_dir():
            from datasets import load_from_disk as _load_disk
            source = _load_disk(str(source_dir))
            logger.info(
                f"Loaded source split '{args.input_split}' from {args.input_dataset} ({len(source)} rows)"
            )
        else:
            from datasets import load_dataset as _load_hub
            source = _load_hub(args.input_dataset, split=args.input_split)
            logger.info(
                f"Loaded source split '{args.input_split}' from HF Hub: {args.input_dataset} ({len(source)} rows)"
            )

        # Load existing output split when re-sampling
        existing = None
        if args.resample_ids is not None:
            out_dir = Path(output_dataset_path) / args.output_split
            if not out_dir.is_dir():
                logger.error(
                    f"--resample_ids requires the output split to already exist at {out_dir}"
                )
                return
            from datasets import load_from_disk as _load_disk
            existing = _load_disk(str(out_dir))
            logger.info(
                f"Re-sampling {len(args.resample_ids)} row(s) in existing split "
                f"'{args.output_split}' at {output_dataset_path}"
            )

        pipeline = build_pipeline(args)
        augmented = augment_dataset_split(
            source, args, pipeline, logger,
            resample_indices=args.resample_ids,
            existing_dataset=existing,
        )

        if args.push_to_hub:
            import os
            from huggingface_hub import login
            token = os.environ.get("HF_TOKEN")
            if not token:
                logger.error("HF_TOKEN environment variable is not set — cannot push to Hub.")
                return
            login(token=token)
            try:
                current = dict(load_dataset(output_dataset_path))
                logger.info("Loaded existing splits from Hub: %s", list(current.keys()))
            except Exception:
                current = {}
            current[args.output_split] = augmented
            compatible = {k: v for k, v in current.items() if v.features == augmented.features}
            stale = set(current) - set(compatible)
            if stale:
                logger.warning("Dropping schema-incompatible splits (need rebuild): %s", sorted(stale))
            DatasetDict(compatible).push_to_hub(output_dataset_path, private=True)
            logger.info(f"Pushed split '{args.output_split}' to HF Hub: {output_dataset_path}")
        else:
            out_path = Path(output_dataset_path) / args.output_split
            augmented.save_to_disk(str(out_path))
            logger.info(f"Saved split '{args.output_split}' to: {out_path}")
        return

    # ── Dir mode (original behavior) ─────────────────────────────────────────
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
