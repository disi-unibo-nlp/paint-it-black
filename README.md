# multimodal-deid

## Overview

A benchmark pipeline for evaluating multimodal large language models on the task of **medical document de-identification**. The project covers the full workflow from raw annotated PDFs to structured evaluation results:

1. **Annotation** — browser-based tool for drawing PHI bounding boxes on PDF pages.
2. **Dataset building** — convert annotated PDFs into a HuggingFace `Dataset` with image + label columns.
3. **Augmentation** — simulate realistic hospital scan degradation (fax, stains, noise, skew, …) to create difficulty levels.
4. **Review** — Jupyter notebook for visual QA and rejection marking of augmented samples.
5. **Inference** — run any OpenAI-compatible vision-language model on each document page, extract predicted PHI entities, and compute a comprehensive suite of benchmark metrics.

---

## Project Structure

```
multimodal-deid/
├── config/
│   ├── dataprep/               # Augmentation configs (presets + per-augmentation examples)
│   └── inference/
│       ├── base.yaml           # Default inference config (local VLLM backend)
│       └── openai.yaml         # Remote OpenAI backend config
├── data/                       # Not tracked — populated by the user
│   ├── dev_annotation_cases/   # Raw PDF + annotation.json pairs for dataset building
│   └── test_ds/                # Output HF dataset (base / medium / hard splits)
├── docker/
│   ├── Dockerfile              # Based on vllm/vllm-openai; adds project deps
│   └── requirements.txt
├── experiments/
│   ├── augment_tuning.sh       # Run all augmentation example configs for visual tuning
│   ├── create_dataset.sh       # End-to-end dataset build + augmentation workflow
│   └── run_inference.sh        # Benchmark inference across splits
├── output/                     # Augmented PNGs and inference results (not tracked)
├── scripts/
│   ├── build_image.sh          # Build the Docker image
│   ├── run_cont.sh             # Run any command inside the container
│   ├── run_job.sh              # SLURM HPC submission wrapper
│   └── run_vllm_inference.sh   # Start VLLM serve + inference in one container
├── src/
│   ├── analysis/
│   │   └── review_augmentations.ipynb   # Jupyter QA notebook
│   ├── core/
│   │   ├── llm_client.py       # OpenAI-compatible client with retry logic
│   │   ├── template_handler.py # Prompt template loader + output parser (multimodal-aware)
│   │   └── utils.py            # ConfigArgumentParser, logging, seed, output dir helpers
│   ├── dataprep/
│   │   ├── annotation_app.html # Standalone browser annotation tool (no server needed)
│   │   ├── augment_pdfs.py     # PDF → augmented PNG pipeline
│   │   └── build_dataset.py    # Annotated PDFs → HuggingFace Dataset
│   ├── inference/
│   │   ├── metrics.py          # Full benchmark metric suite
│   │   └── run_inference.py    # Main inference + evaluation script
│   └── tests/
│       ├── test_augment_dataset_hf.py
│       ├── test_augment_page_sampling.py
│       ├── test_llm_client.py
│       └── test_metrics.py
└── templates/
    ├── deid_template.yaml      # Multimodal prompt template for de-identification
    └── placeholder_template.yaml
```

---

## Getting Started

### Prerequisites

- **Docker** with GPU support (`nvidia-container-toolkit`)
- **NVIDIA driver** compatible with CUDA 12.4
- **~30 GB disk** for the Docker image (VLLM base is large)
- `curl` on the host (used by `run_vllm_inference.sh` for the health check)

### Docker Build

```bash
./scripts/build_image.sh
```

The image is based on `vllm/vllm-openai:v0.8.5` with project-specific Python packages
(`augraphy`, `datasets`, `pdf2image`, `openai`, …) installed on top.
The first build downloads the full VLLM base image (~18 GB); subsequent builds are cached.

### Environment

Copy `.env.example` to `.env` (if present) and fill in any tokens needed for your setup
(e.g. `HF_TOKEN` for gated models, `OPENAI_API_KEY` for remote inference).
The `.env` file is sourced automatically by all `scripts/*.sh` wrappers.

---

## Running the Container

All commands run from the **project root**.

```bash
# Interactive shell
./scripts/run_cont.sh

# Run a single command
./scripts/run_cont.sh python3 src/dataprep/build_dataset.py --help

# Start Jupyter notebook server on port 8888 (VS Code Remote SSH: connect via Existing Jupyter Server)
./scripts/run_cont.sh -j

# Jupyter on a custom port
./scripts/run_cont.sh -j 8899
```

`run_cont.sh` uses `--network host` so the container can reach host services
(e.g. a VLLM server started separately). GPU device is configurable via
`CUDA_VISIBLE_DEVICES` in `.env` (default: `1`).

---

## Data Preparation

### 1 — Annotate

Open `src/dataprep/annotation_app.html` in a browser (no server required — it uses the
File System Access API). Click **Open Folder**, select a directory containing PDF files,
draw bounding boxes for each PHI entity, assign labels, and save. The tool writes an
`annotation.json` file into the same directory.

Labels follow a `PARENT:SUBLABEL` convention (e.g. `NAME:PATIENT`, `ID:DOCUMENT_ID`).
See the label list at the top of the HTML file.

### 2 — Build Dataset

```bash
./scripts/run_cont.sh python3 src/dataprep/build_dataset.py \
    --input_dir data/dev_annotation_cases \
    --output_dir data/test_ds \
    --split base \
    --dpi 200
```

Produces a HuggingFace `Dataset` saved to `data/test_ds/base/` with columns:
`image` (PIL), `page` (int), `annotations` (list of `{id, label, text, bboxes}`).
Bounding boxes are normalised `[y_min, x_min, y_max, x_max]` in `[0, 1]`.

### 3 — Augment

```bash
# Low-noise augmentation → medium split
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --input_dataset data/test_ds \
    --input_split base \
    --output_split medium \
    --config config/dataprep/low_noise_mix.yaml \
    --seed 1

# High-noise augmentation → hard split
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --input_dataset data/test_ds \
    --input_split base \
    --output_split hard \
    --config config/dataprep/high_noise_mix.yaml \
    --seed 1
```

Or run the full end-to-end workflow:
```bash
./experiments/create_dataset.sh
```

See the [Data Augmentation Reference](#data-augmentation-reference) section below for the
full config documentation.

### 4 — Review Augmented Samples

Start Jupyter and open the notebook:
```bash
./scripts/run_cont.sh -j
# In VS Code: Kernel → Existing Jupyter Server → http://localhost:8888
# Open: src/analysis/review_augmentations.ipynb
```

The notebook loads a dataset split, displays each page image with colour-coded bounding
boxes, and lets you navigate and mark samples as rejected. State is saved to
`data/test_ds/<split>/review.json` and can be resumed across sessions.

---

## Inference & Evaluation

The inference pipeline runs a vision-language model on each document page image,
extracts predicted PHI entities, and evaluates them against ground-truth annotations.

### Quick Start (local VLLM)

```bash
./scripts/run_vllm_inference.sh \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --config config/inference/base.yaml \
    --input_dataset data/test_ds \
    --input_split base \
    --run_name eval_base
```

`run_vllm_inference.sh` starts `vllm serve` and the inference script **inside the same
container** so `localhost:8000` is always reachable without any network configuration.

Results are written to `output/inference/<run_name>/results.json`.

### `run_vllm_inference.sh` Reference

```
Usage: ./scripts/run_vllm_inference.sh --model MODEL [vllm options] [inference options]

VLLM serve options (consumed by this script):
  --model MODEL                         Model name or HuggingFace path (required)
  --vllm_port PORT                      Port for VLLM serve (default: 8000)
  --vllm_gpu_memory_utilization FLOAT   GPU memory fraction (default: 0.9)
  --vllm_max_model_len INT              Max token length — must exceed image tokens
                                        + prompt tokens (default: 8192)
  --vllm_tensor_parallel_size INT       Number of GPUs for tensor parallelism (default: 1)

All other flags are forwarded verbatim to run_inference.py.
```

### Remote API Backend (OpenAI, etc.)

For remote APIs no VLLM is needed — use `run_cont.sh` directly:

```bash
./scripts/run_cont.sh python3 src/inference/run_inference.py \
    --config config/inference/openai.yaml \
    --input_dataset data/test_ds \
    --input_split base \
    --api_key $OPENAI_API_KEY \
    --run_name eval_gpt4o_base
```

### `run_inference.py` Reference

```
General:
  --run_name STR          Label for this run; becomes the output subdirectory (default: inference_run)
  --log_level STR         Logging verbosity (default: INFO)
  --seed INT              Random seed (default: 42)

Dataset:
  --input_dataset PATH    Root of the HF dataset (e.g. data/test_ds) [required]
  --input_split STR       Split to evaluate (default: base)
  --output_dir PATH       Output root (default: output/inference)

Template:
  --template PATH         Prompt template YAML (default: templates/deid_template.yaml)

Backend / API:
  --backend STR           openai | vllm | anthropic (default: vllm)
  --model STR             Model name [required]
  --base_url URL          API base URL (default: http://localhost:8000/v1)
  --api_key STR           API key; use EMPTY for local VLLM (default: EMPTY)
  --max_new_tokens INT    Max tokens to generate (default: 2048)
  --timeout INT           Request timeout in seconds (default: 120)
  --max_retries INT       Retry attempts on transient errors (default: 3)

Evaluation:
  --iou_thresholds FLOAT+ IoU thresholds for end-to-end F1 (default: 0.25 0.5 0.75)
```

Config files under `config/inference/` follow the same key names and can be used via
`--config config/inference/base.yaml`. CLI flags override config values.

### Prompt Templates

Templates live in `templates/` and are YAML files loaded by `TemplateHandler`. Content
fields can be strings (text-only) or lists of typed blocks for multimodal messages:

```yaml
messages:
  - role: system
    content: "You are a de-identification expert..."
  - role: user
    content:
      - type: image
        variable: page_image     # PIL Image passed from the inference loop → base64 JPEG
      - type: text
        text: "List all PHI entities in this document page."

output_fields:
  - name: entities
    pattern: null        # null = use the full model output as-is
    required: true
```

The default template (`templates/deid_template.yaml`) instructs the model to return a
JSON array of `{label, text, bboxes}` objects. Curly braces in template string content
must be doubled (`{{`, `}}`) to avoid Python's `.format()` interpreting them as slots.

### Metrics

Results in `results.json` include:

| Metric | Description |
|---|---|
| `entity_detection_f1` | P/R/F1 matching by label + text (char-F1 > 0.5), ignoring bbox. Micro + macro + per label. |
| `end_to_end_f1` | P/R/F1 matching by label + bbox IoU > threshold. Reported at each configured threshold. Micro + macro + per label. |
| `char_f1` | Mean SQuAD-style character-level F1 for text fields of matched pairs (IoU > 0.5). |
| `edit_distance` | Mean normalised Levenshtein distance for matched pairs (0 = identical). |
| `exact_match_rate` | Fraction of GT entities exactly reproduced (case-insensitive). |
| `mean_iou` | Mean bbox IoU for entities matched by text + label. |
| `hallucination_rate` | FP / (TP + FP) — predicted entities that don't correspond to any GT entity. Overall + per label. |
| `miss_rate` | FN / (TP + FN) — GT entities not found by the model. Overall + per label. |
| `coarse_f1` | Same as `entity_detection_f1` but with labels collapsed to their parent (e.g. `NAME:PATIENT` → `NAME`). |
| `per_label_breakdown` | Full P/R/F1/TP/FP/FN table for every label. |
| `label_confusion` | For FP predictions with bbox overlap against a GT entity of a different label: `{gt_label: {pred_label: count}}`. |
| `format_compliance` | Fraction of samples where the model returned parseable JSON (0–1). |

Multi-box entities (PHI spanning non-contiguous regions) are reduced to their
axis-aligned enclosing rectangle for IoU computation.

### Experiment Script

`experiments/run_inference.sh` contains pre-configured calls across all splits and
configurations. Edit `MODEL` and `DATASET_DIR` at the top, then:

```bash
./experiments/run_inference.sh
```

---

## Data Augmentation Reference

### PDF Augmentation

`src/dataprep/augment_pdfs.py` converts a folder of PDFs into augmented scan-like PNG images
using the [augraphy](https://github.com/sparkfish/augraphy) library. Each PDF page is
rasterised and passed through a configurable degradation pipeline that simulates realistic
hospital scanning artefacts.

**Output layout:**
```
data/augmented/
└── <pdf_stem>/
    ├── page_000_aug_000.png
    ├── page_000_aug_001.png
    └── page_001_aug_000.png
```

#### Configuration Reference

Config lives in `config/dataprep/augment_config.yaml`. Every key can be overridden on the
command line — CLI values always win over the file.

---

**General**

| Key | Type | Default | Description |
|---|---|---|---|
| `run_name` | `str` | `augment_run` | Human-readable label printed in logs. Useful for distinguishing multiple runs in the same output directory. |
| `log_level` | `str` | `INFO` | Logging verbosity. Use `DEBUG` to see every saved filename; `WARNING` to suppress progress lines in batch jobs. |
| `seed` | `int` | `42` | Global random seed passed to Python's `random`, NumPy, and PyTorch — and also used as augraphy's `random_seed`. The same seed on the same input produces identical augmented images, which is useful for reproducing a specific artefact combination during debugging. Change it (or the input documents) to generate different variants. |

---

**I/O**

| Key | Type | Default | Description |
|---|---|---|---|
| `input_dir` | `str` | `./data/input` | Path to the directory containing source PDF files. All `*.pdf` files in this directory are processed; subdirectories are ignored. |
| `output_dir` | `str` | `./output/augmented_docs` | Root output directory. Augmented PNGs are written to `<output_dir>/<pdf_stem>/page_NNN_aug_MMM.png`. The directory is created if it does not exist. |
| `num_augmentations` | `int` | `2` | Number of independently augmented variants to generate **per page**. For a 3-page PDF with `num_augmentations: 5`, the pipeline produces 15 images. Increase this to expand a small document set into a larger training corpus — 3–5 is a practical range before diversity plateaus. |

---

**Rendering**

| Key | Type | Default | Description |
|---|---|---|---|
| `dpi` | `int` | `200` | DPI at which PDF pages are rasterised before augmentation. Controls base image resolution. Higher DPI = sharper text and finer artefact detail at the cost of memory and time. See [DPI guidance](#tips--best-practices) below. |

---

**Augmentation parameters**

Every augmentation parameter has its own flat key following the naming pattern `<augmentation>_<param>`, e.g. `ink_bleed_p`, `ink_bleed_intensity_range`, `geometric_rotate_range`. The full set of standard keys is documented in `config/dataprep/augment_config.yaml` with inline comments.

Three special conventions:

- **Probability (`_p`)** — every augmentation has a `<name>_p` key (float 0.0–1.0). Set it to `0.0` to disable that augmentation entirely; `1.0` to make it always fire.
- **OneOf membership flags** — augmentations that compete in a `OneOf` pool (e.g. the four scanner noise types) each have an integer flag (`1` = included, `0` = excluded from the pool), independently of their parent group's `_p`.
- **Page-level and profile sampling** — some augmentations support sampling a discrete profile once per image rather than varying continuously within a range. These are custom features not in the augraphy library; their args are documented in [Page-level and profile sampling](#page-level-and-profile-sampling) below.

The pipeline structure is:

| Phase | Augmentations |
|---|---|
| **Ink** | `ink_bleed`, `bleed_through`, `low_ink` (OneOf: random/periodic lines), `ink_mottling`, `lines_degradation` |
| **Paper** | `color_paper` ¹, `texture` (noise + brightness sequence), `stains`, `watermark` |
| **Post** | `scanner_noise` (OneOf: bad_photo_copy/dirty_rollers/dirty_drum/dirty_screen), `geometric`, `lighting_gradient`, `shadow_cast`, `exposure` (OneOf: brightness/gamma), `subtle_noise` ¹, `jpeg`, `folding`, `page_border`, `annotations` (markup ¹, scribbles), `faxify` ¹ |

¹ When the corresponding sampling flag is enabled, the augmentation is suppressed from the
phase above and applied outside the pipeline so a discrete profile can be sampled per image.
See [Page-level and profile sampling](#page-level-and-profile-sampling).

**Presets are config files.** To use a different set of augmentation values, create a new config file and pass it via `--config`. See [Example Configs](#example-configs) below.

---

#### Page-level and profile sampling

Some augmentations support sampling a **discrete profile** once per image, before the
augraphy pipeline runs. This allows different augmented variants of the same page to have
systematically different paper colour, noise level, fax mode, or annotation style —
rather than drawing from a single continuous range each time.

When a sampling flag is set to `1`, the augmentation is removed from the augraphy phase
and applied manually: a profile is drawn from a weighted distribution, its parameters are
resolved, and the augmentation is called with `p=1.0` (probability is handled manually to
avoid augraphy returning `None` on a probability miss).

Four augmentations support this:

| Augmentation | Sampling flag | What is sampled |
|---|---|---|
| `color_paper` | `color_paper_page_sampling` | HSV profile (hue + saturation ranges) — applied in a freshly built per-image paper phase |
| `subtle_noise` | `subtle_noise_page_sampling` | Noise intensity (`subtle_range`) — applied after the pipeline |
| `faxify` | `faxify_profile_sampling` | Fax mode: mono-only / halftone-only / both — applied after the pipeline |
| `annotations` markup | `annotations_markup_sampling` | Markup type (highlight/strikethrough/underline/crossed) + per-type color — applied after the pipeline |

---

**Faxify mutex**

When `faxify_profile_sampling: 1` and `faxify_p > 0`, faxify is mutually exclusive
with `scanner_noise`, `shadow_cast`, and `stains`. If faxify fires for a given image,
those three augmentations are suppressed for that image's pipeline. Faxify probability
is preserved exactly; the actual rate of the other augmentations is reduced
proportionally (e.g. `scanner_noise_p: 0.1` with `faxify_p: 0.5` → effective scanner
noise rate ≈ 0.05). The mutex is inactive when `faxify_profile_sampling: 0`.

---

**color_paper page sampling**

| Key | Type | Default | Description |
|---|---|---|---|
| `color_paper_page_sampling` | `int` | `0` | `1` = sample a paper-colour profile once per image. |
| `color_paper_num_profiles` | `int` | `1` | Number of discrete paper-colour profiles. |
| `color_paper_profile_weights` | `float list` | `[1.0]` | Sampling weight per profile (normalised internally; need not sum to 1). |
| `color_paper_profile_hue_ranges` | `int list` | `[20, 45]` | Flat `[lo, hi, lo, hi, ...]` — one `[lo, hi]` pair per profile. |
| `color_paper_profile_saturation_ranges` | `int list` | `[10, 35]` | Flat `[lo, hi, lo, hi, ...]` — one `[lo, hi]` pair per profile. |

Example — 3 profiles (near-white / cream / aged yellow):
```yaml
color_paper_page_sampling: 1
color_paper_num_profiles: 3
color_paper_profile_weights: [0.30, 0.40, 0.30]
color_paper_profile_hue_ranges:        [15, 20,  15, 25,  20, 30]
color_paper_profile_saturation_ranges: [ 5, 15,  15, 40,  40, 80]
```

---

**subtle_noise page sampling**

| Key | Type | Default | Description |
|---|---|---|---|
| `subtle_noise_page_sampling` | `int` | `0` | `1` = sample a noise-level profile once per image. |
| `subtle_noise_num_profiles` | `int` | `1` | Number of discrete noise-level profiles. |
| `subtle_noise_profile_weights` | `float list` | `[1.0]` | Sampling weight per profile. |
| `subtle_noise_profile_ranges` | `int list` | `[12]` | One `subtle_range` value per profile. |

Example — 2 profiles (low / stronger):
```yaml
subtle_noise_page_sampling: 1
subtle_noise_num_profiles: 2
subtle_noise_profile_weights: [0.50, 0.50]
subtle_noise_profile_ranges: [18, 30]
```

---

**faxify profile sampling**

Three fixed profiles (mono-only / halftone-only / both) are always used; only their
weights are configurable. "Neither" is never sampled — if `faxify_p` fires, one of the
three modes is guaranteed.

| Key | Type | Default | Description |
|---|---|---|---|
| `faxify_profile_sampling` | `int` | `0` | `1` = sample one of three fax modes per image (suppresses Faxify from the pipeline phase). |
| `faxify_profile_weights` | `float list (3)` | `[0.33, 0.34, 0.33]` | Weights for profiles 0 (mono-only), 1 (halftone-only), 2 (both). |
| `faxify_monochrome_methods` | `str list` | `[]` | Inclusion list for monochrome methods. When non-empty, overrides `faxify_monochrome_method` with a uniform random pick per image. Valid values: `threshold_li`, `threshold_mean`, `threshold_otsu`, `threshold_sauvola`, `threshold_triangle`. Empty = use `faxify_monochrome_method` as-is. |

Example — equal probability for all three modes:
```yaml
faxify_profile_sampling: 1
faxify_profile_weights: [0.33, 0.34, 0.33]
```

---

**annotations markup sampling**

When `annotations_markup_sampling: 1`, the markup type and ink are both drawn per image.
The color is then resolved from the corresponding `annotations_markup_<ink>_colors` arg,
so the same ink type always gets the same color palette regardless of which geometric type
was selected.

| Key | Type | Default | Description |
|---|---|---|---|
| `annotations_markup_sampling` | `int` | `0` | `1` = sample markup type + ink + color per image. |
| `annotations_markup_type_weights` | `float list (4)` | `[0.25, 0.25, 0.25, 0.25]` | Sampling weights for `strikethrough`, `crossed`, `underline`, `highlight` (in that order). Normalised internally. |
| `annotations_markup_ink` | `str` | `random` | Ink texture: `random`, `pencil`, `pen`, `marker`, `highlighter`. |
| `annotations_markup_ink_weights` | `float[4]` | `[0.25, 0.25, 0.25, 0.25]` | Sampling weights for `[pencil, pen, marker, highlighter]` — used when `annotations_markup_ink="random"`. Need not sum to 1. |
| `annotations_markup_large_word_mode` | `int` | `1` | `-1` = library default (randomly picks strict or lenient per call — unreliable for PDFs). `0` = strict: only narrow contours (`w < image_width/5`) are marked. `1` = lenient: any contour passing the height check is marked (recommended). |
| `annotations_markup_single_word_mode` | `int` | `0` | `-1` = random per image (coin flip). `0` = line-level: dilation kernel `(20,1)` merges words into full lines. `1` = word-level: kernel `(10,1)` marks individual words and forces `markup_length_range=(1,1)`. |
| `annotations_markup_repetitions` | `int[2]` | `[1, 1]` | Range `[min, max]` of strokes drawn per qualifying contour — sampled uniformly per image. |
| `annotations_markup_pencil_thickness_range` | `int[2]` | `[1, 3]` | Thickness `[min, max]` used when ink is `pencil`. |
| `annotations_markup_pen_thickness_range` | `int[2]` | `[1, 3]` | Thickness `[min, max]` used when ink is `pen`. |
| `annotations_markup_marker_thickness_range` | `int[2]` | `[1, 3]` | Thickness `[min, max]` used when ink is `marker`. |
| `annotations_markup_highlighter_thickness_range` | `int[2]` | `[1, 3]` | Thickness `[min, max]` used when ink is `highlighter`. |
| `annotations_markup_pencil_colors` | color spec | `random` | Color used when ink is `pencil`. |
| `annotations_markup_pen_colors` | color spec | `random` | Color used when ink is `pen`. |
| `annotations_markup_marker_colors` | color spec | `random` | Color used when ink is `marker`. |
| `annotations_markup_highlighter_colors` | color spec | `random` | Color used when ink is `highlighter`. |

**Color spec encoding** — each `*_colors` arg accepts one of three forms:

| Form | Example | Behaviour |
|---|---|---|
| String `random` | `random` | Fully random RGB per call (augraphy default) |
| String `contrast` | `contrast` | Inverse of the dominant page colour |
| 3 integers `R G B` | `255 220 0` | Fixed colour |
| 6+ integers (multiples of 3) | `0 0 0  0 0 200` | Sample uniformly from the listed colours |

Example — realistic annotations (yellow highlighter, black pen/marker/pencil):
```yaml
annotations_markup_sampling: 1
annotations_markup_pencil_colors: [0, 0, 0]
annotations_markup_pen_colors: [0, 0, 0]
annotations_markup_marker_colors: [0, 0, 0]
annotations_markup_highlighter_colors: [255, 220, 0]
```

---

#### Example Configs

`config/dataprep/examples/` contains one config per augmentation type. Each enables only that augmentation (at `p: 1.0` so it always fires) and disables everything else, producing 5 variants per page for isolated visual comparison.

| Config | What it showcases |
|---|---|
| `ink_bleed.yaml` | Ink spreading into paper fibres |
| `bleed_through.yaml` | Back-page ghosting through thin paper |
| `low_ink.yaml` | Faded toner streaks (random and periodic banding) |
| `ink_mottling.yaml` | Patchy toner density |
| `lines_degradation.yaml` | Broken form rules and ruled lines |
| `color_paper.yaml` | Aged / yellowed paper |
| `texture.yaml` | Paper grain and uneven scanner brightness |
| `stains.yaml` | Coffee and water stains |
| `watermark.yaml` | Hospital stamps and overlays |
| `scanner_noise.yaml` | OneOf: bad_photo_copy / dirty_rollers / dirty_drum / dirty_screen |
| `geometric.yaml` | Scanner skew |
| `lighting_gradient.yaml` | Uneven lamp illumination |
| `shadow_cast.yaml` | Book-spine or hand shadow |
| `exposure.yaml` | Global brightness and gamma shift |
| `subtle_noise.yaml` | CCD sensor noise |
| `jpeg.yaml` | JPEG compression artefacts |
| `folding.yaml` | Fold creases |
| `page_border.yaml` | Scanner-lid border artefacts |
| `annotations.yaml` | Handwritten markup and scribbles |
| `faxify.yaml` | Fax transmission (monochrome + halftone) |

Run any example config with:
```bash
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --config dataprep/examples/scanner_noise
```

---

#### Usage Examples

All commands below assume you are in the project root and the Docker image has been built.

**Quickstart** — run with config file defaults:
```bash
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --config dataprep/augment_config
```

**High-quality rendering** — 300 DPI for final training data:
```bash
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --config dataprep/augment_config \
    --dpi 300
```

**Batch augmentation** — 5 variants per page for dataset expansion:
```bash
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --config dataprep/augment_config \
    --num_augmentations 5 \
    --output_dir ./data/augmented_batch
```

**Fax simulation** — enable fax on top of the hospital preset:
```bash
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --config dataprep/augment_config \
    --faxify_p 0.2 \
    --output_dir ./data/augmented_fax
```

**Disable a specific augmentation** — remove skew, keep everything else:
```bash
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --config dataprep/augment_config \
    --geometric_p 0.0
```

**Debug a single PDF** — verbose output, one augmentation per page:
```bash
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --config dataprep/augment_config \
    --input_dir ./data/dev_docs \
    --num_augmentations 1 \
    --log_level DEBUG
```

---

#### Tips & Best Practices

**DPI trade-offs**

| DPI | Use case | Notes |
|---|---|---|
| `150` | Fast iteration / visual spot-checks | Noticeably pixelated; fine for checking pipeline logic, not for training |
| `200` | Default — balanced quality and speed | Good for most development and moderate-scale training runs |
| `300` | Final training data | Sharp text, detailed artefacts; ~2× the memory and processing time of 200 DPI |

**Scanner skew**
Keep `geometric_rotate_range` within `[-5.0, 5.0]`. Real flatbed and sheet-fed hospital scanners produce skew in the 0–3° range — values above ±5° start to look like intentional rotation rather than a misaligned paper feed and can hurt model generalisation.

**Number of augmentations vs. document diversity**
Generating many variants of the same document (high `num_augmentations`) increases volume but not content diversity. For robust model training, prioritise adding more source PDFs over increasing `num_augmentations`. A ratio of 3–5 augmentations per page is a practical ceiling before returns diminish.

**Reproducibility with `seed`**
The seed locks the entire augmentation sequence — same seed, same input, same output every time. This is valuable when you want to isolate the effect of a single config change without other randomness interfering. To intentionally generate diverse variants across separate runs, change `seed` between runs rather than relying solely on `num_augmentations`.

**Tuning individual augmentations**
Use the example configs in `config/dataprep/examples/` to visualise each effect in isolation before tuning. Adjust the `_p` probability and parameter ranges in `augment_config.yaml`, then verify the change visually with `--num_augmentations 5 --log_level DEBUG` before running the full dataset.

**Faxify and readability**
Faxify heavily degrades text and should not co-occur with scanner noise, shadow cast, or stains. When `faxify_profile_sampling: 1`, these are automatically suppressed for any image where faxify fires (see [Faxify mutex](#faxify-mutex)). Set `faxify_p` to control the desired fax rate directly; the other augmentations' actual rates will be reduced proportionally.

**`max_model_len` for inference**
Image tokens depend on resolution. At 200 DPI a full A4 page typically tokenises to 4 000–6 000 tokens with Qwen-VL models. The default `--vllm_max_model_len 8192` covers this; increase to `16384` for higher-resolution inputs or very long system prompts.

---

## Development

### Core Utilities (`src/core/`)

**`ConfigArgumentParser`** (`utils.py`) — wraps `argparse` with YAML config file support.
Priority: explicit CLI args > config file values > `add_argument` defaults. Pass
`--config path/to/config.yaml` (relative to the `config_dir` set at construction time,
`.yaml` extension optional).

**`TemplateHandler`** (`template_handler.py`) — loads prompt templates from YAML, fills
`{placeholder}` slots in string content and `variable:` references in image blocks, and
extracts structured fields from model output via regex. Multimodal messages (image +
text) are built in OpenAI API format; PIL images are base64-encoded as JPEG automatically.

**`LLMClient`** (`llm_client.py`) — thin wrapper around `openai.OpenAI` with configurable
retry logic (exponential backoff). Works with any OpenAI-compatible endpoint including
local VLLM servers.

### Running Tests

```bash
./scripts/run_cont.sh python3 -m pytest src/tests/ -v
```

Tests cover augmentation page-sampling logic, HF dataset mode, metrics computation, and
the LLM client retry behaviour. All tests use mocks or synthetic data — no GPU or network
access required.

### SLURM / HPC

```bash
./scripts/run_job.sh <command>
```

Wraps `sbatch` with the project's standard resource request. See the script header for
configurable SLURM variables (`PARTITION`, `GPUS`, `MEM`, `TIME`).
