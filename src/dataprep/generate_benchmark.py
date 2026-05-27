"""
Clinical Document Benchmark Generator — 4 seed types
======================================================
Generates synthetic clinical document variants (HTML + PDF) for PHI
de-identification benchmarking across 4 document types:

  lab        — Clinical laboratory report        (templates/benchmark/lab_report.yaml)
  ct         — CT radiology report               (templates/benchmark/ct_report.yaml)
  mri        — MRI radiology report              (templates/benchmark/mri_report.yaml)
  gynecology — Gynaecological visit report       (templates/benchmark/gynecological_report.yaml)

Setup:
    pip install google-genai weasyprint pypdf beautifulsoup4
    export GOOGLE_API_KEY="..."

Usage:
    python3 src/dataprep/generate_benchmark.py --config dataprep/generate_benchmark
    python3 src/dataprep/generate_benchmark.py --config dataprep/generate_benchmark --types lab ct
    python3 src/dataprep/generate_benchmark.py --config dataprep/generate_benchmark --count 5
    python3 src/dataprep/generate_benchmark.py --config dataprep/generate_benchmark --dry_run
"""

import os
import re
import sys
import json
import time
import random
import itertools
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Allow imports from src/ when running the script directly or via run_cont.sh
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.utils import ConfigArgumentParser, init_logger
from core.template_handler import TemplateHandler

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

try:
    from weasyprint import HTML as _WeasyHTML, CSS
    _WEASYPRINT_AVAILABLE = True
except ImportError:
    _WEASYPRINT_AVAILABLE = False
    CSS = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page-context descriptions (vary depth by page count for each doc type)
# ---------------------------------------------------------------------------

_PAGE_CONTEXT_LAB = {
    1: "urgent or limited panel, concise single-visit workup",
    2: "standard outpatient workup with 2–3 panels",
    3: "comprehensive workup with 4–5 panels and interpretive comments",
    4: "extended multi-panel workup with full interpretive narrative and follow-up recommendations",
}

_PAGE_CONTEXT_CT = {
    1: "single-region routine exam, concise findings",
    2: "multi-region or complex exam with detailed findings",
    3: "comprehensive staging or multi-system exam with extensive findings and prior comparison",
    4: "whole-body or oncology restaging report with full history, prior comparison, and management plan",
}

_PAGE_CONTEXT_MRI = {
    1: "single-region routine exam, concise technique and findings",
    2: "complex multi-sequence exam with detailed technique and findings",
    3: "comprehensive multi-sequence exam with prior comparison and management plan",
    4: "multi-region or whole-spine exam with extensive findings, prior comparison, and detailed management plan",
}

_PAGE_CONTEXT_GYN = {
    1: "brief follow-up visit or results letter — content must fill the single page at least 80%",
    2: "standard specialist consultation — each page must be at least 80% full",
    3: "comprehensive first consultation — distribute content evenly, each page at least 80% full",
    4: "complex multi-system workup — each page carries a distinct section, all at least 80% full",
}

# ---------------------------------------------------------------------------
# Document type definitions
# Each entry defines:
#   template_file — TemplateHandler YAML path (relative to repo root)
#   page_context  — page-count → context-string mapping
#   param_grid    — axes to cross-product for this type
# ---------------------------------------------------------------------------

DOC_TYPES = {
    "lab": {
        "template_file": "templates/benchmark/lab_report.yaml",
        "page_context":  _PAGE_CONTEXT_LAB,
        "param_grid": {
            "pages":     [1, 2, 3, 4],
            "country":   ["US", "UK", "Australia", "Canada"],
            "sex":       ["Male", "Female"],
            "age_range": ["teens", "20s", "30s", "40s", "50s", "60s", "70s", "80s+"],
            "clinical_context": [
                "routine annual health check",
                "pre-operative assessment",
                "chronic kidney disease monitoring",
                "diabetic management follow-up",
                "suspected autoimmune disease workup",
                "oncology treatment monitoring",
                "suspected thyroid dysfunction",
                "acute illness workup — fever and malaise",
                "cardiovascular risk assessment",
                "medication toxicity monitoring",
                "post-surgical follow-up",
                "infertility and hormonal workup",
            ],
            "panels": [
                "CBC, Clinical Chemistry, Lipid Profile",
                "CBC, Liver Function Tests, Coagulation",
                "CBC, Thyroid Panel, Inflammatory Markers",
                "CBC, HbA1c, Renal Function, Urinalysis",
                "CBC, Lipid Profile, Tumor Markers, Coagulation",
                "CBC, Renal Function, Electrolytes, Urinalysis",
                "CBC, Inflammatory Markers, Coagulation, Blood Cultures",
                "CBC, Autoimmune Panel (ANA, ANCA, RF, anti-dsDNA), Complement Levels",
                "CBC, HbA1c, Lipid Profile, Renal Function, Liver Function Tests",
                "CBC, Thyroid Panel, Iron Studies, Vitamin B12 and Folate",
                "CBC, Toxicology Screen, Liver Function Tests, Renal Function",
                "CBC, Bone Profile (Calcium, Phosphate, ALP, PTH), Vitamin D, Renal Function",
            ],
            "style": [
                "plain institutional monochrome, single-column layout, patient info as compact label-value pairs, serif font",
                "near-monochrome with dark grey header bar, single-column, two-column patient info table, no color fills",
                "minimal institutional, wide left margin, underlined section labels, Times New Roman body text, ruled separators between sections",
                "hospital HIS printout style, single-column, monospaced-feel sans-serif, plain thin-border tables, no decoration",
                "outpatient letter format, left-aligned letterhead, single-column body, bold inline section labels, compact line spacing",
                "European hospital form style, thin vertical line dividing narrow left margin codes from main content, single-column right, serif font",
                "US academic medical centre EMR printout, patient banner header, single-column body, sans-serif, no decoration",
                "fax-ready monochrome, single-column, bold section labels, sans-serif, ruled lines between sections, no decoration",
            ],
        },
    },

    "ct": {
        "template_file": "templates/benchmark/ct_report.yaml",
        "page_context":  _PAGE_CONTEXT_CT,
        "param_grid": {
            "pages":     [1, 2],
            "country":   ["US", "UK", "Australia", "Canada"],
            "sex":       ["Male", "Female"],
            "age_range": ["teens", "20s", "30s", "40s", "50s", "60s", "70s", "80s+"],
            "exam_type": [
                "CT Abdomen/Pelvis with and without IV contrast",
                "CT Chest with contrast (pulmonary embolism protocol)",
                "CT Head without contrast",
                "CT Urography with contrast",
                "CT Neck with contrast",
                "CT Chest/Abdomen/Pelvis with contrast (oncology staging protocol)",
                "CT Pulmonary Angiography (CTPA)",
                "CT Abdomen without contrast (renal colic protocol)",
                "CT Cervical Spine without contrast",
                "CT Thorax without contrast (high-resolution HRCT protocol)",
                "CT Coronary Angiography (CTCA)",
                "CT Aortography with contrast",
            ],
            "indication": [
                "acute abdominal pain, fever, and elevated inflammatory markers",
                "sudden onset dyspnea and pleuritic chest pain, suspected PE",
                "severe headache, nausea, and visual disturbances",
                "haematuria and flank pain, rule out urolithiasis or renal mass",
                "neck mass and hoarseness, history of smoking",
                "known malignancy, staging and treatment response assessment",
                "acute chest pain with elevated D-dimer, rule out aortic dissection",
                "blunt abdominal trauma following road traffic accident",
                "chronic cough and weight loss, suspected primary lung malignancy",
                "pre-operative workup for planned colorectal surgery",
                "follow-up of previously identified pulmonary nodule (Fleischner protocol)",
                "acute neurological deficit with sudden onset, rule out ischaemic stroke",
            ],
            "style": [
                "fax-ready monochrome radiology report with FAX TRANSMISSION block prepended at top of page 1 (CONFIDENTIAL header, FROM / TO / DATE / PAGES fields above the hospital header); single-column below, bold section labels, sans-serif, ruled lines between sections, no decoration",
                "Canadian outpatient radiology fax style, single-column, sans-serif, burgundy header bar; no bold, no underline, no italics anywhere; section breaks indicated only by a plain label word (e.g. 'Findings:' or 'Impression:') with no formatting; patient info as compact single-row text string; raw undecorated clinical feel",
            ],
        },
    },

    "mri": {
        "template_file": "templates/benchmark/mri_report.yaml",
        "page_context":  _PAGE_CONTEXT_MRI,
        "param_grid": {
            "pages":     [1, 2, 3],
            "country":   ["US", "UK", "Australia", "Canada"],
            "sex":       ["Male", "Female"],
            "age_range": ["teens", "20s", "30s", "40s", "50s", "60s", "70s", "80s+"],
            "exam_type": [
                "MRI Pelvis without contrast (endometriosis protocol)",
                "MRI Brain without and with contrast",
                "MRI Lumbar Spine without contrast",
                "MRI Knee without contrast",
                "mpMRI Prostate without contrast",
                "MRI Cervical Spine without contrast",
                "MRI Shoulder without contrast",
                "MRI Abdomen with contrast (liver / HCC surveillance protocol)",
                "MRI Breast with contrast (staging protocol)",
                "MRI Brain without contrast (epilepsy protocol)",
                "MRI Cardiac without and with contrast (function and viability)",
                "MRI Whole Spine without contrast (cervical + thoracic + lumbar)",
            ],
            "indication": [
                "chronic pelvic pain, dysmenorrhea, and deep dyspareunia; suspected recurrent endometriosis",
                "progressive headache, memory impairment, and personality change",
                "lower back pain radiating to left leg, suspected L4-L5 disc herniation",
                "acute knee pain after sports injury, suspected meniscal or ligamentous tear",
                "elevated PSA (8.2 ng/mL) on routine screening, no prior biopsy",
                "neck pain with bilateral upper limb radiculopathy, suspected cervical myelopathy",
                "shoulder pain and restricted range of motion after fall, suspected rotator cuff tear",
                "known hepatocellular carcinoma, follow-up after locoregional treatment",
                "family history of BRCA1 mutation, annual high-risk breast screening",
                "first unprovoked generalised tonic-clonic seizure, new diagnosis workup",
                "palpitations and dyspnoea, reduced ejection fraction on echocardiogram, cardiomyopathy workup",
                "widespread neck and back pain with progressive lower limb weakness",
            ],
            "style": [
                # --- Group A: raw / no-decoration ---
                "plain institutional MRI printout, single-column, serif font, dark grey header bar (max 25mm); ALL section labels are plain unformatted inline text — no bold, no underline, no italics anywhere; patient info as raw colon-separated lines; looks like verbatim dictation output printed with no typographic treatment",
                "HIS terminal printout style, single-column, sans-serif font, slate grey header bar; section labels are plain CAPITAL words inline (e.g. 'TECHNIQUE:', 'FINDINGS:'), no bold, no underline, no decorative treatment anywhere; patient info block as plain colon-separated lines; tight spacing, raw clinical feel",
                "voice-recognition transcription printout, single-column, monospaced-feel font (Courier or similar), NO header bar — hospital name and radiology department as plain unformatted text lines only; NO section label words anywhere — sections separated only by a single blank line; patient info as plain multi-line block; looks like raw VR output before any editorial cleanup",
                "Australian public hospital MRI printout, single-column, sans-serif, NO header bar (hospital name + department as plain small-caps text, no background fill); section separators are thin horizontal rules ONLY — no section label text anywhere; patient info as compact single-line string (Name — DOB — MRN — Exam Date); completely undecorated minimal feel",
                "UK internal radiology memo style, single-column, serif font, burgundy header bar (max 20mm); section breaks indicated ONLY by a single blank line — no section label words, no rules, no decoration of any kind; patient info as plain label: value lines; document reads as one continuous flowing prose block with zero structural markup",
                # --- Group B: formatted institutional ---
                "NHS radiology letter format, left-aligned letterhead, single-column, Times New Roman, navy blue header bar; bold underlined section headers; patient info as two-column label-value table; standard 15mm margins; conventional formatted appearance",
                "US academic medical centre EMR printout, compact single-line patient banner (Name | DOB | MRN | Exam Date — one row, no tall coloured block), single-column body, sans-serif, dark teal accent lines under section labels; bold section headers; clean formatted look",
                "institutional MRI printout, single-column, serif font, charcoal header bar; section headers in slightly larger font (15px) but NO bold, NO underline, NO italics anywhere — visual hierarchy via font size only; patient info as bordered two-column label-value table; clean formatted appearance without typographic weight",
                "private imaging centre letter style, single-column, serif font, forest green header bar; section headers in italics ONLY (no bold, no underline) — gentle typographic distinction; patient info as vertical label: value list; formal but restrained institutional feel",
                "Canadian university hospital MRI style, single-column, serif font, deep indigo/purple header bar; section headers in small-caps (no bold, no underline, no italics) — typographic distinction via small-caps only; patient info as horizontal single-row label-value string; restrained academic institutional feel",
                "Australian private imaging centre style, single-column, sans-serif, warm terracotta/rust header bar; section headers in a slightly darker rust-colored text (no bold, no underline, no italics) — color only; patient info as compact two-column table with thin border; warm-toned institutional feel",
                # --- Group C: margin annotation / form ---
                "European radiology form style: narrow left margin column (~12% width) with a continuous vertical border-right line (1px, #aaaaaa); left margin contains right-aligned section codes (IND, TECH, FIND, IMP, SGN); main content single-column right, serif font, charcoal header bar; no section headers repeated in main content",
                "Italian ASL/AST radiology form style: narrow left margin column (~10% width) with a vertical border-right line; left margin shows plain exam codes and abbreviated section tags (TEC, RIS, IMP); main content right, sans-serif, plain institutional header with hospital name and radiology department only",
                "NHS outpatient form style: narrow left margin column (~12% width) with vertical border-right line; left margin contains right-aligned section codes (IND, TECH, FIND, IMP); main content right, Times New Roman, navy blue header bar; no section headers in main content",
                "US hospital form style: narrow left margin column (~12% width) with vertical border-right line; left margin shows abbreviated tags (IND, TECH, FIND, IMP, SGN); main content right, sans-serif, dark teal header bar; no section headers in main content",
                "Australian imaging form style: narrow left margin column (~10% width) with vertical border-right line; left margin shows plain section reference codes; main content right, sans-serif, warm rust/terracotta header bar; no section headers in main content",
                # --- Group D: fax / transmission ---
                "fax-ready monochrome MRI report with FAX TRANSMISSION block prepended at top of page 1 (CONFIDENTIAL header, FROM / TO / DATE / PAGES fields above the hospital header); single-column below, bold section labels, sans-serif, ruled lines between sections, no decoration",
                "Canadian outpatient MRI fax style, single-column, sans-serif, burgundy header bar; no bold, no underline, no italics anywhere; section breaks indicated only by a plain label word (e.g. 'Findings:' or 'Impression:') with no formatting; patient info as compact single-row text string; raw undecorated clinical feel",
                "UK NHS fax transmission style, single-column, Times New Roman, navy header bar; FAX TRANSMISSION block at top of page 1; bold section headers; patient info as compact two-column table; clean formatted appearance",
                "US hospital fax style, single-column, sans-serif, charcoal header bar; FAX TRANSMISSION block at top of page 1; section labels in plain ALL-CAPS with no other decoration; patient info as compact single-line string; raw institutional feel",
                "Australian imaging fax style, single-column, sans-serif, NO decorative header bar (hospital name and department as plain text only); FAX TRANSMISSION block at top of page 1; section separators are thin horizontal rules only — no section label text; patient info as plain Name — DOB — MRN line",
            ],
        },
    },

    "gynecology": {
        "template_file": "templates/benchmark/gynecological_report.yaml",
        "page_context":  _PAGE_CONTEXT_GYN,
        "param_grid": {
            "pages":     [1, 2, 3],
            "country":   ["US", "UK", "Australia", "Canada"],
            "age_range": ["20s", "30s", "40s", "50s", "60s"],
            "scenario": [
                "uterine fibromatosis with menometrorrhagia, failed medical therapy",
                "endometriosis stage III with dysmenorrhea, deep dyspareunia, and infertility",
                "right ovarian endometrioma (4 cm), found incidentally on ultrasound",
                "PCOS with oligomenorrhea, hyperandrogenism, and metabolic features",
                "postmenopausal uterine bleeding — endometrial thickening 9 mm on transvaginal USS",
                "primary infertility workup, suspected tubal factor, IVF counselling",
                "anterior and posterior pelvic organ prolapse, grade II cystocele",
                "adenomyosis with heavy menstrual bleeding, levonorgestrel IUS failure",
                "high-grade cervical dyskaryosis (HSIL) follow-up after LLETZ procedure",
                "premature ovarian insufficiency — secondary amenorrhoea aged 33, FSH 68 IU/L",
            ],
            "style": [
                "plain institutional monochrome, single-column, serif font, compact layout, patient info as label-value block, charcoal header bar (max 25mm total header height). No bold section labels, but clear visual separation.",
                "NHS outpatient letter format, left-aligned address window block, formal salutation line, Times New Roman, navy blue header bar; margins standard 15mm (NOT wide) on 1-page documents",
                "minimal hospital printout style, single-column, underlined section labels, Times New Roman, burgundy header accent (single underlined clinic name line, no decorative block)",
                "US academic medical center EMR printout, compact single-line patient banner header (name | DOB | MRN | date — one row, no tall coloured block), single-column body, sans-serif, dark teal accent",
                "Australian private rooms letterhead, left-aligned, single-column, provider number in footer, serif font, forest green header (letterhead max 2 lines)",
                "institutional HIS printout feel, single-column, plain thin-border patient info table, no decoration, slate grey header bar (max 20mm)",
                "private women's health clinic letter, centred clinic name, single-column body, italic section labels, serif font, deep purple header; left margin standard 15mm (NOT wide) on 1-page documents",
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# WeasyPrint CSS override (instantiated lazily so the module can be imported
# without WeasyPrint installed, e.g. during unit testing)
# ---------------------------------------------------------------------------

_WEASYPRINT_CSS_STRING = """
    @page { size: A4; margin: 0; }
    body  { margin: 0; padding: 0; background: white; }
    .page {
        width: 210mm;
        height: 297mm;
        box-sizing: border-box;
        overflow: hidden;
        page-break-after: always;
        break-after: page;
    }
"""

def _get_weasyprint_css():
    if not _WEASYPRINT_AVAILABLE:
        raise RuntimeError("weasyprint is not installed. Run inside the Docker container.")
    return CSS(string=_WEASYPRINT_CSS_STRING)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_html(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def count_html_pages(html_text: str) -> int:
    if not _BS4_AVAILABLE:
        raise RuntimeError("beautifulsoup4 is not installed. Run inside the Docker container.")
    soup = BeautifulSoup(html_text, "html.parser")
    return len(soup.find_all("div", class_="page"))


def html_to_pdf(html_text: str, pdf_path: Path) -> int:
    if not _WEASYPRINT_AVAILABLE or not _PYPDF_AVAILABLE:
        raise RuntimeError("weasyprint and pypdf are required. Run inside the Docker container.")
    html_obj = _WeasyHTML(string=html_text)
    html_obj.write_pdf(target=str(pdf_path), stylesheets=[_get_weasyprint_css()],
                       presentational_hints=True)
    return len(PdfReader(str(pdf_path)).pages)


def create_chat(client, system: str, model: str):
    """Create a stateful multi-turn chat session with the given system instruction."""
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(system_instruction=system),
    )


# ---------------------------------------------------------------------------
# Build flat job list from selected doc types
# ---------------------------------------------------------------------------

def build_jobs(selected_types: list) -> list:
    """Returns a flat list of job dicts, each with doc_type + all params."""
    jobs = []
    for doc_type in selected_types:
        cfg = DOC_TYPES[doc_type]
        grid = cfg["param_grid"]
        keys = list(grid.keys())
        for combo in itertools.product(*grid.values()):
            params = dict(zip(keys, combo))
            params["doc_type"] = doc_type
            jobs.append(params)
    return jobs


def job_key(job: dict) -> str:
    """Canonical JSON fingerprint for a job — used to detect duplicates."""
    return json.dumps({k: job[k] for k in sorted(job)}, sort_keys=True)


def load_existing_manifest(out_dir: Path) -> tuple:
    """Load manifest.json from out_dir if present.

    Returns:
        existing_results : list of sample dicts already recorded
        done_keys        : set of job_key strings for status=='ok' entries
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return [], set()

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = data.get("samples", [])
        done = set()
        for entry in existing:
            if entry.get("status") == "ok":
                job = dict(entry.get("params", {}))
                job["doc_type"] = entry["doc_type"]
                done.add(job_key(job))
        print(f"Loaded existing manifest: {len(existing)} entries, {len(done)} completed (ok).")
        return existing, done
    except Exception as e:
        print(f"Warning: could not load existing manifest: {e}")
        return [], set()


# ---------------------------------------------------------------------------
# Visual self-correction (PDF review → HTML fix)
# ---------------------------------------------------------------------------

_SELF_CORRECT_PROMPT = """\
The HTML you generated has been rendered to a PDF via WeasyPrint. The PDF is attached.
Check every page for these layout problems only — fix ONLY what is broken, change nothing else:

1. TEXT CLIPPED OR OVERLAPPING FOOTER — body text runs into or past the footer.
   Fix: increase padding-bottom on .page so content ends above the footer.

2. PAGE TOO DENSE — text is cramped or overflows the page boundary.
   Fix: reduce font-size by 1–2px, or reduce margin/padding on sections.
   Last resort only: trim a sentence of non-PHI prose. Never remove a section.

3. PAGE TOO SPARSE (multi-page documents only) — a page ends with large blank space while
   the next page starts with content that could have fit on the previous page.
   Fix: reduce excessive spacing between sections (margin/padding) so content flows up
   to fill the gap. Do NOT add dummy content and do NOT change the page count.

4. COLOURED SIDEBAR NOT FULL HEIGHT — sidebar background ends before the page bottom.
   Fix: use display:table on .page and display:table-cell on sidebar + content divs.

If the PDF looks correct, respond with exactly: OK
If you find any issue, respond with the COMPLETE corrected HTML only.
- Exactly {n_pages} <div class="page"> elements
- No markdown fences, no explanation — raw HTML only
"""


def self_correct_variant(chat, pdf_path: Path, expected_pages: int):
    """Continue the existing chat session by sending the rendered PDF for visual review.

    The model already has the HTML it generated in context, so we only need to
    attach the PDF and ask for a correction — no need to re-send the HTML source.

    Returns:
        (corrected_html, True)  — model found issues and returned fixed HTML
        (None, False)           — model said OK, or correction was unusable
    """
    prompt = _SELF_CORRECT_PROMPT.format(n_pages=expected_pages)
    try:
        response = chat.send_message([
            types.Part.from_bytes(
                data=pdf_path.read_bytes(),
                mime_type="application/pdf",
            ),
            prompt,
        ])
        result = extract_html(response.text.strip())
    except Exception as e:
        print(f"    self-correct API call failed: {e}")
        return None, False

    if result.strip().upper() == "OK":
        return None, False
    if result.lower().startswith("<!doctype"):
        return result, True
    # Unrecognised response — treat as no correction needed
    print(f"    self-correct: unrecognised response ({result[:80]!r}), skipping")
    return None, False


# ---------------------------------------------------------------------------
# Single variant generator
# ---------------------------------------------------------------------------

def generate_variant(
    client,
    job: dict,
    sample_id: str,
    out_dir: Path,
    args,
) -> dict:
    doc_type = job["doc_type"]
    cfg      = DOC_TYPES[doc_type]
    expected = job["pages"]

    # Load template and build system + user messages
    handler          = TemplateHandler.from_yaml(cfg["template_file"])
    page_context_str = cfg["page_context"].get(expected, "clinical document")
    format_kwargs    = {k: v for k, v in job.items() if k != "doc_type"}
    format_kwargs["page_context"] = page_context_str
    messages = handler.format(**format_kwargs)
    system   = next(m["content"] for m in messages if m["role"] == "system")
    user_msg = next(m["content"] for m in messages if m["role"] == "user")

    exc = None
    for attempt in range(1, args.retry_max + 1):
        print(f"  [attempt {attempt}/{args.retry_max}] calling API...", flush=True)
        try:
            # Each retry starts a fresh chat so the model has no stale context.
            chat      = create_chat(client, system, args.model)
            raw       = chat.send_message(user_msg).text
            html_text = extract_html(raw)

            if not html_text.lower().startswith("<!doctype"):
                raise ValueError("Response does not start with <!DOCTYPE html>")

            html_pages = count_html_pages(html_text)
            if html_pages != expected:
                raise ValueError(
                    f"HTML has {html_pages} <div class='page'> elements, expected {expected}"
                )

            html_path = out_dir / f"{sample_id}.html"
            html_path.write_text(html_text, encoding="utf-8")

            pdf_path  = out_dir / f"{sample_id}.pdf"
            pdf_pages = html_to_pdf(html_text, pdf_path)

            if pdf_pages != expected:
                raise ValueError(
                    f"PDF has {pdf_pages} pages, expected {expected} "
                    f"(HTML had {html_pages})"
                )

            print(f"  ✓ {sample_id}.html + .pdf  ({pdf_pages} pages)")

            # ------------------------------------------------------------------
            # Visual self-correction: show the model its own rendered PDF and
            # ask it to fix any layout/content issues (up to self_correct_max rounds)
            # ------------------------------------------------------------------
            sc_rounds_done = 0
            if args.self_correct_max > 0 and expected > 1:
                for sc_round in range(1, args.self_correct_max + 1):
                    print(f"  [self-correct {sc_round}/{args.self_correct_max}] reviewing PDF...",
                          flush=True)
                    corrected_html, was_corrected = self_correct_variant(
                        chat, pdf_path, expected
                    )

                    if not was_corrected:
                        print(f"  ✓ self-review OK")
                        break

                    # Validate the corrected HTML before accepting it
                    corrected_pages = count_html_pages(corrected_html)
                    if corrected_pages != expected:
                        print(f"  ↻ correction changed page count to {corrected_pages} "
                              f"(expected {expected}) — discarding, keeping previous version")
                        break

                    # Render to a temp file so the validated PDF is never overwritten
                    # before we confirm the corrected version is actually valid.
                    tmp_pdf = pdf_path.with_suffix(".sc_tmp.pdf")
                    try:
                        corrected_pdf_pages = html_to_pdf(corrected_html, tmp_pdf)
                    except Exception as render_err:
                        print(f"  ↻ correction render failed ({render_err}) — keeping previous")
                        if tmp_pdf.exists():
                            tmp_pdf.unlink()
                        break

                    if corrected_pdf_pages != expected:
                        print(f"  ↻ corrected PDF has {corrected_pdf_pages} pages "
                              f"(expected {expected}) — keeping previous")
                        tmp_pdf.unlink()
                        break

                    # Validation passed — atomically replace both files
                    html_path.write_text(corrected_html, encoding="utf-8")
                    tmp_pdf.replace(pdf_path)   # atomic rename; original is gone only now
                    html_text   = corrected_html
                    html_pages  = corrected_pages
                    pdf_pages   = corrected_pdf_pages
                    sc_rounds_done += 1
                    print(f"  ↻ correction applied (round {sc_round})")

            return {
                "sample_id":        sample_id,
                "doc_type":         doc_type,
                "status":           "ok",
                "html_pages":       html_pages,
                "pdf_pages":        pdf_pages,
                "self_corrections": sc_rounds_done,
                "params":           {k: v for k, v in job.items() if k != "doc_type"},
            }

        except Exception as e:
            exc = e
            print(f"  ✗ attempt {attempt} failed: {e}")
            if attempt < args.retry_max:
                print(f"  waiting {args.retry_wait}s before retry...")
                time.sleep(args.retry_wait)

    return {
        "sample_id": sample_id,
        "doc_type":  doc_type,
        "status":    "failed",
        "error":     str(exc),
        "params":    {k: v for k, v in job.items() if k != "doc_type"},
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> ConfigArgumentParser:
    all_types = list(DOC_TYPES.keys())
    parser = ConfigArgumentParser(
        description="Generate PHI benchmark clinical documents (4 seed types)",
        config_dir=Path("./config/dataprep"),
    )
    parser.add_argument("--run_name",         type=str, default="benchmark_generation",
                        help="Name of this run (for logging / manifest)")
    parser.add_argument("--log_level",        type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--model",            type=str, default="gemini-3-flash-preview",
                        help="Gemini model to use for generation")
    parser.add_argument("--output_dir",       type=str, default="output/benchmark",
                        help="Directory for generated HTML, PDF, and manifest.json")
    parser.add_argument("--retry_max",        type=int, default=3,
                        help="Max generation attempts per variant before marking as failed")
    parser.add_argument("--retry_wait",       type=int, default=10,
                        help="Seconds to wait between retry attempts")
    parser.add_argument("--self_correct_max", type=int, default=1,
                        help="Visual self-correction rounds after initial render (0 = disabled)")
    parser.add_argument("--types",            type=str, nargs="+",
                        choices=all_types, default=None, metavar="TYPE",
                        help=f"Document types to generate. Choices: {all_types}. Default: all.")
    parser.add_argument("--count",            type=int, default=None,
                        help="Generate N randomly sampled variants (across selected types)")
    parser.add_argument("--dry_run",          action="store_true", default=False,
                        help="Print generation plan without making any API calls")
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    args    = _build_parser().parse_args()
    logger  = init_logger(__name__, level=args.log_level)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_types = args.types if args.types else list(DOC_TYPES.keys())

    # Load existing manifest to skip already-completed combos
    existing_results, done_keys = load_existing_manifest(out_dir)

    jobs = build_jobs(selected_types)

    # Filter out combos that already succeeded
    original_count = len(jobs)
    jobs = [j for j in jobs if job_key(j) not in done_keys]
    skipped = original_count - len(jobs)

    if args.count:
        jobs = random.sample(jobs, min(args.count, len(jobs)))

    # Count per type for display
    type_counts: dict = {}
    for j in jobs:
        type_counts[j["doc_type"]] = type_counts.get(j["doc_type"], 0) + 1

    logger.info(
        f"{'DRY RUN — ' if args.dry_run else ''}Generating {len(jobs)} variant(s)"
        + (f"  ({skipped} skipped — already in manifest)" if skipped else "")
    )
    logger.info(f"Output dir : {out_dir.resolve()}")
    logger.info(f"Model      : {args.model}")
    if type_counts:
        logger.info("Breakdown  : " + "  |  ".join(f"{t}: {n}" for t, n in type_counts.items()))

    if not jobs:
        logger.info("Nothing to do — all requested variants are already completed.")
        return

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            print(f"  [{i:04d}] {job['doc_type']:<12} country={job.get('country','?')}  "
                  f"pages={job.get('pages','?')}")
        return

    if not _GENAI_AVAILABLE:
        sys.exit("Error: google-genai package is not installed. Run inside the Docker container.")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Error: GOOGLE_API_KEY environment variable not set.")

    client      = genai.Client(api_key=api_key)
    new_results = []
    start       = datetime.now()
    id_offset   = len(existing_results)  # new samples numbered after existing ones

    for i, job in enumerate(jobs, 1):
        doc_type  = job["doc_type"]
        country   = job.get("country", "XX")
        sex       = job.get("sex", job.get("age_range", "X")).lower().replace(" ", "")
        sample_id = f"sample_{id_offset + i:04d}_{doc_type}_{country.lower()}_{sex}"

        logger.info(f"[{i}/{len(jobs)}] {sample_id}")

        # Sub-type detail line
        detail = job.get("panels") or job.get("exam_type") or job.get("scenario") or ""
        if detail:
            logger.info(f"  detail : {detail[:80]}")

        result = generate_variant(client, job, sample_id, out_dir, args)
        new_results.append(result)

        if i < len(jobs):
            time.sleep(2)

    # Merge with existing results and save manifest
    all_results   = existing_results + new_results
    all_doc_types = sorted({r["doc_type"] for r in all_results})
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "generated_at": start.isoformat(),
        "model":        args.model,
        "total":        len(all_results),
        "ok":           sum(1 for r in all_results if r["status"] == "ok"),
        "failed":       sum(1 for r in all_results if r["status"] == "failed"),
        "by_type": {
            t: {
                "ok":     sum(1 for r in all_results if r["doc_type"] == t and r["status"] == "ok"),
                "failed": sum(1 for r in all_results if r["doc_type"] == t and r["status"] == "failed"),
            }
            for t in all_doc_types
        },
        "samples": all_results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed    = (datetime.now() - start).seconds
    new_ok     = sum(1 for r in new_results if r["status"] == "ok")
    new_failed = sum(1 for r in new_results if r["status"] == "failed")
    logger.info(f"{'='*55}")
    logger.info(f"Done in {elapsed}s  ({len(new_results)} new,  {skipped} skipped)")
    logger.info(f"  This run — OK: {new_ok}  Failed: {new_failed}")
    logger.info(f"  Manifest total — OK: {manifest['ok']}/{manifest['total']}")
    for t, counts in manifest["by_type"].items():
        logger.info(f"  {t:<12}: {counts['ok']} ok / {counts['failed']} failed")
    logger.info(f"  Manifest: {manifest_path.resolve()}")

    if new_failed:
        logger.warning("Failed samples (this run):")
        for r in new_results:
            if r["status"] == "failed":
                logger.warning(f"  - {r['sample_id']}: {r.get('error', '?')}")


if __name__ == "__main__":
    main()
