# multimodal-deid

## Overview

_TODO: brief description of the project — goals, scope, target models/tasks._

---

## Project Structure

_TODO: annotated directory tree._

---

## Getting Started

### Prerequisites

_TODO: list system requirements (CUDA version, Docker, Poppler, etc.)._

### Environment Setup

_TODO: copy `.env.example` to `.env`, fill in tokens (HF_TOKEN, WANDB_API_KEY, etc.)._

### Docker Build

_TODO: `./scripts/build_image.sh` walkthrough._

---

## Running the Pipeline

### Interactive container

_TODO: `./scripts/run_cont.sh` with no args drops into a shell._

### Single command

_TODO: `./scripts/run_cont.sh <command>` pattern._

### SLURM (HPC)

_TODO: `./scripts/run_job.sh` usage and relevant SLURM env vars._

---

## Data Preparation

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
| `output_dir` | `str` | `./data/augmented` | Root output directory. Augmented PNGs are written to `<output_dir>/<pdf_stem>/page_NNN_aug_MMM.png`. The directory is created if it does not exist. |
| `num_augmentations` | `int` | `1` | Number of independently augmented variants to generate **per page**. For a 3-page PDF with `num_augmentations: 5`, the pipeline produces 15 images. Increase this to expand a small document set into a larger training corpus — 3–5 is a practical range before diversity plateaus. |

---

**Rendering**

| Key | Type | Default | Description |
|---|---|---|---|
| `dpi` | `int` | `200` | DPI at which PDF pages are rasterised before augmentation. Controls base image resolution. Higher DPI = sharper text and finer artefact detail at the cost of memory and time. See [DPI guidance](#tips--best-practices) below. |

---

**Augmentation parameters**

Every augmentation parameter has its own flat key following the naming pattern `<augmentation>_<param>`, e.g. `ink_bleed_p`, `ink_bleed_intensity_range`, `geometric_rotate_range`. The full set of ~100 keys is documented in `config/dataprep/augment_config.yaml` with inline comments.

Two special conventions:

- **Probability (`_p`)** — every augmentation has a `<name>_p` key (float 0.0–1.0). Set it to `0.0` to disable that augmentation entirely; `1.0` to make it always fire.
- **OneOf membership flags** — augmentations that compete in a `OneOf` pool (e.g. the four scanner noise types) each have an integer flag (`1` = included, `0` = excluded from the pool), independently of their parent group's `_p`.

The pipeline structure is:

| Phase | Augmentations |
|---|---|
| **Ink** | `ink_bleed`, `bleed_through`, `low_ink` (OneOf: random/periodic lines), `ink_mottling`, `lines_degradation` |
| **Paper** | `color_paper`, `texture` (noise + brightness sequence), `stains`, `watermark` |
| **Post** | `scanner_noise` (OneOf: bad_photo_copy/dirty_rollers/dirty_drum/dirty_screen), `geometric`, `lighting_gradient`, `shadow_cast`, `exposure` (OneOf: brightness/gamma), `subtle_noise`, `jpeg`, `folding`, `page_border`, `annotations` (OneOf: markup/scribbles), `faxify` |

**Presets are config files.** To use a different set of augmentation values, create a new config file and pass it via `--config`. See [Example Configs](#example-configs) below.

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

You can also use the experiment wrapper, which passes any extra flags through:
```bash
./scripts/run_cont.sh bash experiments/augment.sh --num_augmentations 3
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

**Fax as a minority augmentation**
Set `faxify_p` to a small value (e.g. `0.15–0.2`) and generate fax variants into a separate output directory, then merge them at a controlled ratio (~10–15% of total samples). Fax images are heavily degraded and, if over-represented, can pull the model toward expecting low-resolution monochrome inputs.

**Output naming and dataset tracking**
The `page_NNN_aug_MMM.png` filename pattern encodes both the original page index and the augmentation index. This makes it straightforward to: (a) group all variants of the same source page, (b) link augmented images back to their originals for annotation purposes, and (c) implement stratified splits that keep all variants of a page in the same fold.

---

## Development

### Project Layout

_TODO: annotated module descriptions._

### Adding a Custom Augmentation Preset

_TODO: copy `config/dataprep/augment_config.yaml`, tune parameter values, and pass the new file via `--config`. No code changes needed._

### Core Utilities

_TODO: `ConfigArgumentParser`, `init_logger`, `set_seed` — usage notes._

---

## Roadmap

_TODO: planned modules (model training, inference, evaluation, de-identification)._
