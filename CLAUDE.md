# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo generates a synthetic clinical document benchmark dataset for evaluating multimodal AI PHI (Protected Health Information) de-identification. It uses Google Gemini to generate realistic HTML clinical documents, then converts them to PDF via WeasyPrint.

## Setup

```bash
pip install google-genai weasyprint pypdf beautifulsoup4 python-dotenv
```

Set `GOOGLE_API_KEY` in `.env` (auto-loaded via `python-dotenv`).

WeasyPrint requires system libraries — see `install_weasyprint.sh` for the apt packages needed on Ubuntu/Debian.

## Key Commands

```bash
# Preview generation plan (no API calls)
python3 generate_benchmark_all.py --dry-run

# Generate N random variants across all document types
python3 generate_benchmark_all.py --count 5

# Generate variants for specific document type(s)
python3 generate_benchmark_all.py --types lab ct --count 3

# Run the full grid (warning: very large — hundreds of variants)
python3 generate_benchmark_all.py

# Legacy single-type lab report generator
python3 generate_benchmark.py --count 5
```

Output goes to `benchmark_output/` by default; override with `--out-dir`.

## Architecture

### Main Script: `generate_benchmark_all.py`

The primary entry point. Supports 4 document types:

| Type | Prompt File | Pages | Param axes |
|------|-------------|-------|------------|
| `lab` | `prompts/prompt_lab_report.md` | 3 | country, sex, panels, style |
| `ct` | `prompts/prompt_ct_report.md` | 1 | country, sex, exam_type, indication, style |
| `mri` | `prompts/prompt_mri_report.md` | 1 | country, sex, exam_type, indication, style |
| `gynecology` | `prompts/prompt_gynecological_report.md` | 2 | country, age_range, scenario, style |

**Generation flow** (per variant):
1. Build a user message from the param combination
2. Call Gemini API with the document-type system prompt
3. Extract HTML (strip any markdown fences the model adds)
4. Validate: must start with `<!DOCTYPE`, must contain exactly N `<div class="page">` elements
5. Save `.html`, convert to PDF via WeasyPrint, validate PDF page count matches
6. Retry up to 3 times (10s wait) on any validation failure
7. Write `manifest.json` summarizing all results

### System Prompts (`.md` files)

Each prompt file is the full system instruction sent to Gemini for that document type. They encode:
- Required PHI categories (patient name, DOB, address, physician, hospital, IDs, etc.)
- WeasyPrint-compatible HTML/CSS constraints (mandatory — violations cause broken PDFs)
- Prose realism rules (flowing narrative, no bullet lists, minor typos in non-PHI text)
- Layout rules (near-monochrome, optional vertical page-division line, mandatory footer)

**Critical WeasyPrint constraint**: every page must be `<div class="page">` with fixed `210mm × 297mm`, `overflow: hidden`, `page-break-after: always`. No `@page` rules, no external CSS, no JS.

### Output Structure

Each benchmark run produces:
- `sample_NNNN_<type>_<country>_<sex/age>.html` — raw generated HTML
- `sample_NNNN_<type>_<country>_<sex/age>.pdf` — WeasyPrint-rendered PDF
- `manifest.json` — run metadata, params, and pass/fail status per sample

### Model

Currently using `gemini-3-flash-preview` (fast). Change `MODEL` constant in the script to `gemini-2.5-pro` for higher quality output.

## Prompt Development

The `.md` prompt files in `prompts/` are the active prompts. `old_prompts/` and `old_prompts_2/` contain earlier iterations — do not edit those. When refining prompts, edit the files in `prompts/` directly.

The `generate_benchmark.py` (singular) script is an older lab-only version that falls back to an inline system prompt if `variant_generation_prompt.md` is missing. Prefer `generate_benchmark_all.py` for new work.
