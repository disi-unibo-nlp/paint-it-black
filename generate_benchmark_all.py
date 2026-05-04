"""
Clinical Document Benchmark Generator — 4 seed types
======================================================
Generates synthetic clinical document variants (HTML + PDF) for PHI
de-identification benchmarking across 4 document types:

  lab       — Clinical laboratory report        (prompts/prompt_lab_report.md)
  ct        — CT radiology report               (prompts/prompt_ct_report.md)
  mri       — MRI radiology report              (prompts/prompt_mri_report.md)
  gynecology — Gynaecological visit report      (prompts/prompt_gynecological_report.md)

Setup:
    pip install google-genai weasyprint pypdf beautifulsoup4
    export GOOGLE_API_KEY="..."

Usage:
    python generate_benchmark.py                        # full grid, all types
    python generate_benchmark.py --types lab ct         # only lab + CT
    python generate_benchmark.py --count 5              # 5 random across all types
    python generate_benchmark.py --count 5 --types mri  # 5 random MRI variants
    python generate_benchmark.py --dry-run              # print plan, no API calls
"""

import os
import re
import sys
import json
import time
import random
import argparse
import itertools
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from weasyprint import HTML, CSS
from pypdf import PdfReader
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR        = Path("benchmark_output_mri_all")  # default output directory
MODEL             = "gemini-3-flash-preview" #"gemini-3-flash-preview" #"gemini-3.1-pro-preview"  # swap to gemini-2.5-pro for higher quality
RETRY_MAX         = 3
RETRY_WAIT        = 10
SELF_CORRECT_MAX  = 1   # visual self-correction rounds after initial render (0 = disabled)

# ---------------------------------------------------------------------------
# Document type definitions
# Each entry defines:
#   prompt_file  — system prompt markdown file to load
#   pages        — fixed page count for this doc type
#   param_grid   — axes to cross-product for this type
#   user_msg_fn  — function(params) -> user turn string
# ---------------------------------------------------------------------------

def _lab_user_msg(p):
    page_context = {
        1: "urgent or limited panel, concise single-visit workup",
        2: "standard outpatient workup with 2–3 panels",
        3: "comprehensive workup with 4–5 panels and interpretive comments",
        4: "extended multi-panel workup with full interpretive narrative and follow-up recommendations",
    }.get(p['pages'], "laboratory report")
    return (
        "Generate a clinical laboratory report HTML variant with the following parameters:\n\n"
        f"- Pages: {p['pages']} ({page_context})\n"
        f"- Country setting: {p['country']}\n"
        f"- Patient sex: {p['sex']}\n"
        f"- Patient age range: {p['age_range']}\n"
        f"- Clinical context: {p['clinical_context']}\n"
        f"- Lab panels to include: {p['panels']}\n"
        f"- Visual style: {p['style']}\n"
    )

def _ct_user_msg(p):
    page_context = {
        1: "single-region routine exam, concise findings",
        2: "multi-region or complex exam with detailed findings",
        3: "comprehensive staging or multi-system exam with extensive findings and prior comparison",
        4: "whole-body or oncology restaging report with full history, prior comparison, and management plan",
    }.get(p['pages'], "radiology report")
    return (
        "Generate a CT radiology report HTML variant with the following parameters:\n\n"
        f"- Pages: {p['pages']} ({page_context})\n"
        f"- Country setting: {p['country']}\n"
        f"- Patient sex: {p['sex']}\n"
        f"- Patient age range: {p['age_range']}\n"
        f"- CT exam type: {p['exam_type']}\n"
        f"- Clinical indication: {p['indication']}\n"
        f"- Visual style: {p['style']}\n"
    )

def _mri_user_msg(p):
    page_context = {
        1: "single-region routine exam, concise technique and findings",
        2: "complex multi-sequence exam with detailed technique and findings",
        3: "comprehensive multi-sequence exam with prior comparison and management plan",
        4: "multi-region or whole-spine exam with extensive findings, prior comparison, and detailed management plan",
    }.get(p['pages'], "radiology report")
    return (
        "Generate an MRI radiology report HTML variant with the following parameters:\n\n"
        f"- Pages: {p['pages']} ({page_context})\n"
        f"- Country setting: {p['country']}\n"
        f"- Patient sex: {p['sex']}\n"
        f"- Patient age range: {p['age_range']}\n"
        f"- MRI exam type: {p['exam_type']}\n"
        f"- Clinical indication: {p['indication']}\n"
        f"- Visual style: {p['style']}\n"
    )

def _gyn_user_msg(p):
    page_context = {
        1: "brief follow-up visit or results letter — content must fill the single page at least 80%",
        2: "standard specialist consultation — each page must be at least 80% full",
        3: "comprehensive first consultation — distribute content evenly, each page at least 80% full",
        4: "complex multi-system workup — each page carries a distinct section, all at least 80% full",
    }.get(p['pages'], "specialist consultation")
    return (
        "Generate a gynaecological visit report HTML variant with the following parameters:\n\n"
        f"- Pages: {p['pages']} ({page_context})\n"
        f"- Country setting: {p['country']}\n"
        f"- Patient age range: {p['age_range']}\n"
        f"- Clinical scenario: {p['scenario']}\n"
        f"- Visual style: {p['style']}\n\n"
        "REMINDER: Never refer to the patient by name in narrative text (history, findings, conclusions). "
        "Use 'the patient', 'she', 'her' only. Patient name appears only in the patient info block and page footer."
    )


DOC_TYPES = {
    "lab": {
        "prompt_file": "prompts/prompt_lab_report.md",
        "user_msg_fn": _lab_user_msg,
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
        "prompt_file": "prompts/prompt_ct_report.md",
        "user_msg_fn": _ct_user_msg,
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
                # --- Group A: raw / no-decoration styles ---
                # "plain institutional radiology printout, single-column, serif font, dark grey header bar (max 25mm); ALL section labels are plain unformatted inline text — no bold, no underline, no italics anywhere in the document; patient info as raw colon-separated lines;  looks like verbatim dictation output printed with no typographic treatment",
                # "HIS terminal printout style, single-column, sans-serif font, slate grey header bar; section labels are plain CAPITAL words inline (e.g. 'FINDINGS:', 'IMPRESSION:'), no bold, no underline, no decorative treatment anywhere; patient info block as plain colon-separated lines; tight spacing, raw clinical feel",
                # "voice-recognition transcription printout, single-column, monospaced-feel font (Courier or similar), NO header bar — hospital name and department as plain unformatted text lines only; NO section label words anywhere — sections separated only by a single blank line; patient info as plain multi-line block; looks like raw VR output before any editorial cleanup",
                # "Australian public hospital discharge printout style, single-column, sans-serif, NO header bar (hospital name + department as plain small-caps text, no background); section separators are thin horizontal rules ONLY — no section label text anywhere; patient info as compact single-line string (Name — DOB — MRN — Exam Date);  completely undecorated minimal feel",
                # "UK internal radiology memo style, single-column, serif font, burgundy header bar (max 20mm); section breaks indicated ONLY by a single blank line — no section label words, no rules, no decoration of any kind; patient info as plain label: value lines;  document reads as one continuous flowing prose block with zero structural markup",
                # --- Group B: formatted institutional styles ---
                # "NHS radiology letter format, left-aligned letterhead, single-column, Times New Roman, navy blue header bar; bold underlined section headers; patient info as two-column label-value table; standard 15mm margins; conventional formatted appearance",
                # "US academic medical centre EMR printout, compact single-line patient banner (Name | DOB | MRN | Exam Date — one row, no tall coloured block), single-column body, sans-serif, dark teal accent lines under section labels; bold section headers; clean formatted look",
                # "institutional radiology printout, single-column, serif font, charcoal header bar; section headers in slightly larger font (15px) but NO bold, NO underline, NO italics anywhere — visual hierarchy via font size only; patient info as bordered two-column label-value table; clean formatted appearance without typographic weight",
                # "outpatient radiology report, single-column, sans-serif, dark teal header bar; section labels in dark teal colored text (no bold, no underline, no italics) — color alone distinguishes section titles from body text; patient info as compact label-value block; structured and readable but entirely decoration-free in terms of weight",
                # "private radiology centre letter style, single-column, serif font, forest green header bar; section headers in italics ONLY (no bold, no underline) — gentle typographic distinction without heavy treatment; patient info as vertical label: value list; formal but restrained institutional feel",
                # "Australian private imaging centre style, single-column, sans-serif, warm terracotta/rust header bar; section headers in a slightly darker rust-colored text (no bold, no underline, no italics) — color only; patient info as compact two-column table with thin border; warm-toned institutional feel distinct from NHS/US styles",
                # "Canadian university hospital radiology style, single-column, serif font, deep indigo/purple header bar; section headers in small-caps (no bold, no underline, no italics) — typographic distinction via small-caps only; patient info as horizontal single-row label-value string; restrained academic institutional feel with distinctive header color",
                # # --- Group C: margin annotation / form styles ---
                # "European radiology form style: narrow left margin column (~12% width) with a MANDATORY continuous vertical line (border-right); left margin contains right-aligned section codes (IND, TECH, FIND, IMP, SGN) each vertically aligned to its corresponding section in the right column; main content single-column right, serif font, charcoal header bar; no section headers in main content",
                # "Italian ASL/AST radiology form style: narrow left margin column (~10% width) with a MANDATORY vertical border-right line; left margin shows plain exam codes and abbreviated section tags (TEC, RIS, IMP); main content right, sans-serif, plain institutional header with hospital name and radiology department only; no section headers in main content",
                # # --- Group D: fax / transmission styles ---
                "fax-ready monochrome radiology report with FAX TRANSMISSION block prepended at top of page 1 (CONFIDENTIAL header, FROM / TO / DATE / PAGES fields above the hospital header); single-column below, bold section labels, sans-serif, ruled lines between sections, no decoration",
                "Canadian outpatient radiology fax style, single-column, sans-serif, burgundy header bar; no bold, no underline, no italics anywhere; section breaks indicated only by a plain label word (e.g. 'Findings:' or 'Impression:') with no formatting; patient info as compact single-row text string; raw undecorated clinical feel",
            ],
        },
    },

    "mri": {
        "prompt_file": "prompts/prompt_mri_report.md",
        "user_msg_fn": _mri_user_msg,
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
        "prompt_file": "prompts/prompt_gynecological_report.md",
        "user_msg_fn": _gyn_user_msg,
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
                # # --- Single-column variants (varied header colors) ---
                "plain institutional monochrome, single-column, serif font, compact layout, patient info as label-value block, charcoal header bar (max 25mm total header height). No bold section labels, but clear visual separation.",
                "NHS outpatient letter format, left-aligned address window block, formal salutation line, Times New Roman, navy blue header bar; margins standard 15mm (NOT wide) on 1-page documents",
                "minimal hospital printout style, single-column, underlined section labels, Times New Roman, burgundy header accent (single underlined clinic name line, no decorative block)",
                "US academic medical center EMR printout, compact single-line patient banner header (name | DOB | MRN | date — one row, no tall coloured block), single-column body, sans-serif, dark teal accent",
                "Australian private rooms letterhead, left-aligned, single-column, provider number in footer, serif font, forest green header (letterhead max 2 lines)",
                "institutional HIS printout feel, single-column, plain thin-border patient info table, no decoration, slate grey header bar (max 20mm)",
                "private women's health clinic letter, centred clinic name, single-column body, italic section labels, serif font, deep purple header; left margin standard 15mm (NOT wide) on 1-page documents",
                # --- Single-column variants — NO bold, NO underline, raw plain text only ---
                # "plain institutional printout, single-column, serif font, charcoal header bar (max 25mm); ALL section labels are plain unformatted inline text — no bold, no underline, no italics anywhere in the document; patient info as raw label: value lines; body text 12px; everything looks like raw typewriter output",
                # "HIS terminal printout style, single-column, sans-serif font, slate grey header bar; section labels are plain capital letters inline with content (e.g. 'INDICATION:'), no bold, no underline, no decorative treatment anywhere; patient info block as plain colon-separated lines; tight spacing, clinical rawness",
                # "NHS internal memo style, single-column, Times New Roman, navy header bar; zero typographic emphasis anywhere — no bold, no underline, no italics; section headers written as plain ALL-CAPS inline words flowing into the text; patient info block plain single-line per field; document reads like raw dictation output printed verbatim",
                # "Canadian outpatient fax output style, single-column, sans-serif, burgundy header bar; no bold, no underline, no italics at all; section breaks indicated only by a single blank line and a plain label word (e.g. 'History:' or 'Findings:') with no formatting treatment; patient info as a compact single-row text string; raw undecorated clinical feel",
                # # --- Narrow left margin annotation column (Option B) ---
                # "European clinic form style: a narrow left margin column (~12% width) separated from the main content by a MANDATORY continuous vertical line (border-right); left margin contains bold right-aligned section codes (IND, HX, EX, LAB, CONC, RX) each vertically aligned to its corresponding section in the right column; header are then no longer needed in the main content; main content single-column right, serif font, navy blue header bar",
                # "NHS outpatient form style: a narrow left margin column (~12% width) separated from the main content by a MANDATORY continuous vertical line (border-right); left margin contains bold right-aligned section codes (HX, EX, CONC, PLAN) each vertically aligned to its corresponding section in the right column; header are then no longer needed in the main content; main content single-column right, Times New Roman, burgundy header bar",
                # #--- Full-height lateral sidebar with patient ID panel (Option C) ---
                #"full-height LEFT sidebar (navy blue background, 18% width) with department info panel (department name, unit/ward name, department head full name and title, department phone and fax, department email, floor/building location); sidebar empty (colour only) on pages 2+; main content right, sans-serif body, NO BOLD HEADERS, No underlined header lines, just raw clinical content.",
                # "full-height LEFT sidebar (navy blue background, 18% width) with department info panel (department name, unit/ward name, department head full name and title, department phone and fax, department email, floor/building location); sidebar empty (colour only) on pages 2+; main content right, sans-serif body, institutional header.",
                # "full-height LEFT sidebar (burgundy background, 17% width) with department info panel (department name, specialty unit, department director full name and title, department code, accreditation number, department contact hours); sidebar empty (colour only) on pages 2+; main content right, serif body font, compact header",
                # "full-height RIGHT sidebar (dark teal background, 18% width) with department info panel (department name, ward number, head of department full name and title, direct phone extension, fax number, hospital floor and building); sidebar empty (colour only) on pages 2+; main content left, sans-serif body, centred clinic header",
                # "full-height RIGHT sidebar (forest green background, 17% width) with department info panel (department name, specialty unit, chief of department full name and title, department phone, department fax, clinic provider number); sidebar empty (colour only) on pages 2+; main content left, serif body font, left-aligned letterhead header",
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# WeasyPrint CSS override
# ---------------------------------------------------------------------------

WEASYPRINT_CSS = CSS(string="""
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
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_system_prompt(filename: str) -> str:
    path = Path(filename)
    if path.exists():
        return path.read_text(encoding="utf-8")
    sys.exit(f"Error: system prompt file '{filename}' not found. "
             f"Make sure all 4 prompt .md files are present in the prompts/ directory.")


def extract_html(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def count_html_pages(html_text: str) -> int:
    soup = BeautifulSoup(html_text, "html.parser")
    return len(soup.find_all("div", class_="page"))


def html_to_pdf(html_text: str, pdf_path: Path) -> int:
    html_obj = HTML(string=html_text)
    html_obj.write_pdf(target=str(pdf_path), stylesheets=[WEASYPRINT_CSS],
                       presentational_hints=True)
    return len(PdfReader(str(pdf_path)).pages)


def create_chat(client: genai.Client, system: str):
    """Create a stateful multi-turn chat session with the given system instruction."""
    return client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=system),
    )


# ---------------------------------------------------------------------------
# Build flat job list from selected doc types
# ---------------------------------------------------------------------------

def build_jobs(selected_types):
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


def load_existing_manifest(out_dir: Path) -> tuple[list, set]:
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
    client: genai.Client,
    job: dict,
    sample_id: str,
    out_dir: Path,
) -> dict:
    doc_type    = job["doc_type"]
    cfg         = DOC_TYPES[doc_type]
    system      = load_system_prompt(cfg["prompt_file"])
    user_msg    = cfg["user_msg_fn"](job)
    expected    = job["pages"]

    exc = None
    for attempt in range(1, RETRY_MAX + 1):
        print(f"  [attempt {attempt}/{RETRY_MAX}] calling API...", flush=True)
        try:
            # Each retry starts a fresh chat so the model has no stale context.
            chat      = create_chat(client, system)
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
            # ask it to fix any layout/content issues (up to SELF_CORRECT_MAX rounds)
            # ------------------------------------------------------------------
            sc_rounds_done = 0
            if SELF_CORRECT_MAX > 0 and expected > 1:
                for sc_round in range(1, SELF_CORRECT_MAX + 1):
                    print(f"  [self-correct {sc_round}/{SELF_CORRECT_MAX}] reviewing PDF...",
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
                "sample_id":       sample_id,
                "doc_type":        doc_type,
                "status":          "ok",
                "html_pages":      html_pages,
                "pdf_pages":       pdf_pages,
                "self_corrections": sc_rounds_done,
                "params":          {k: v for k, v in job.items() if k != "doc_type"},
            }

        except Exception as e:
            exc = e
            print(f"  ✗ attempt {attempt} failed: {e}")
            if attempt < RETRY_MAX:
                print(f"  waiting {RETRY_WAIT}s before retry...")
                time.sleep(RETRY_WAIT)

    return {
        "sample_id": sample_id,
        "doc_type":  doc_type,
        "status":    "failed",
        "error":     str(exc),
        "params":    {k: v for k, v in job.items() if k != "doc_type"},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_types = list(DOC_TYPES.keys())

    parser = argparse.ArgumentParser(
        description="Generate PHI benchmark clinical documents (4 seed types)"
    )
    parser.add_argument(
        "--types", nargs="+", choices=all_types, default=all_types,
        metavar="TYPE",
        help=f"Document types to generate. Choices: {all_types}. Default: all."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generation plan without API calls")
    parser.add_argument("--count", type=int, default=None,
                        help="Generate N randomly sampled variants (across selected types)")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory (default: benchmark_output/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing manifest to skip already-completed combos
    existing_results, done_keys = load_existing_manifest(out_dir)

    jobs = build_jobs(args.types)

    # Filter out combos that already succeeded
    original_count = len(jobs)
    jobs = [j for j in jobs if job_key(j) not in done_keys]
    skipped = original_count - len(jobs)

    if args.count:
        jobs = random.sample(jobs, min(args.count, len(jobs)))

    # Count per type for display
    type_counts = {}
    for j in jobs:
        type_counts[j["doc_type"]] = type_counts.get(j["doc_type"], 0) + 1

    print(f"{'DRY RUN — ' if args.dry_run else ''}Generating {len(jobs)} variant(s)"
          + (f"  ({skipped} skipped — already in manifest)" if skipped else ""))
    print(f"Output dir : {out_dir.resolve()}")
    print(f"Model      : {MODEL}")
    if type_counts:
        print(f"Breakdown  :", "  |  ".join(f"{t}: {n}" for t, n in type_counts.items()))
    print()

    if not jobs:
        print("Nothing to do — all requested variants are already completed.")
        return

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            print(f"  [{i:04d}] {job['doc_type']:<12} country={job.get('country','?')}  "
                  f"pages={job.get('pages','?')}")
        return

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Error: GOOGLE_API_KEY environment variable not set.")

    client     = genai.Client(api_key=api_key)
    new_results = []
    start      = datetime.now()
    id_offset  = len(existing_results)  # new samples numbered after existing ones

    for i, job in enumerate(jobs, 1):
        doc_type  = job["doc_type"]
        country   = job.get("country", "XX")
        sex       = job.get("sex", job.get("age_range", "X")).lower().replace(" ", "")
        sample_id = f"sample_{id_offset + i:04d}_{doc_type}_{country.lower()}_{sex}"

        print(f"\n[{i}/{len(jobs)}] {sample_id}")

        # Sub-type detail line
        detail = job.get("panels") or job.get("exam_type") or job.get("scenario") or ""
        if detail:
            print(f"  detail : {detail[:80]}")

        result = generate_variant(client, job, sample_id, out_dir)
        new_results.append(result)

        if i < len(jobs):
            time.sleep(2)

    # Merge with existing results and save manifest
    all_results   = existing_results + new_results
    all_types     = sorted({r["doc_type"] for r in all_results})
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "generated_at": start.isoformat(),
        "model":        MODEL,
        "total":        len(all_results),
        "ok":           sum(1 for r in all_results if r["status"] == "ok"),
        "failed":       sum(1 for r in all_results if r["status"] == "failed"),
        "by_type": {
            t: {
                "ok":     sum(1 for r in all_results if r["doc_type"] == t and r["status"] == "ok"),
                "failed": sum(1 for r in all_results if r["doc_type"] == t and r["status"] == "failed"),
            }
            for t in all_types
        },
        "samples": all_results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = (datetime.now() - start).seconds
    new_ok     = sum(1 for r in new_results if r["status"] == "ok")
    new_failed = sum(1 for r in new_results if r["status"] == "failed")
    print(f"\n{'='*55}")
    print(f"Done in {elapsed}s  ({len(new_results)} new,  {skipped} skipped)")
    print(f"  This run — OK: {new_ok}  Failed: {new_failed}")
    print(f"  Manifest total — OK: {manifest['ok']}/{manifest['total']}")
    for t, counts in manifest["by_type"].items():
        print(f"  {t:<12}: {counts['ok']} ok / {counts['failed']} failed")
    print(f"  Manifest: {manifest_path.resolve()}")

    if new_failed:
        print("\nFailed samples (this run):")
        for r in new_results:
            if r["status"] == "failed":
                print(f"  - {r['sample_id']}: {r.get('error', '?')}")


if __name__ == "__main__":
    main()