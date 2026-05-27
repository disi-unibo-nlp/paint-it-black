"""
Unit tests for the benchmark generation pipeline.

Run via:
    ./scripts/run_cont.sh python3 -m pytest src/tests/test_generate_benchmark.py -v

No LLM API calls are made — all tests use local files only.
"""

import sys
import json
import tempfile
from pathlib import Path

import pytest

# Allow imports from src/ when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.template_handler import _safe_format, TemplateHandler
from dataprep.generate_benchmark import (
    _build_parser,
    build_jobs,
    job_key,
    extract_html,
    count_html_pages,
    DOC_TYPES,
    _PAGE_CONTEXT_LAB,
    _PAGE_CONTEXT_CT,
    _PAGE_CONTEXT_MRI,
    _PAGE_CONTEXT_GYN,
)


# ---------------------------------------------------------------------------
# A. _safe_format — TemplateHandler helper
# ---------------------------------------------------------------------------

class TestSafeFormat:
    def test_known_placeholder_substituted(self):
        result = _safe_format("Hello {name}!", name="World")
        assert result == "Hello World!"

    def test_css_braces_untouched(self):
        result = _safe_format("body { margin: 0; } {country}", country="UK")
        assert result == "body { margin: 0; } UK"

    def test_unknown_placeholder_preserved(self):
        result = _safe_format("value: {unknown}", known="x")
        assert result == "value: {unknown}"

    def test_multiple_placeholders(self):
        result = _safe_format("{a} and {b}", a="foo", b="bar")
        assert result == "foo and bar"

    def test_extra_kwargs_silently_ignored(self):
        result = _safe_format("only {a}", a="yes", unused_kwarg="ignored")
        assert result == "only yes"

    def test_empty_braces_untouched(self):
        # {} is not a valid identifier placeholder
        result = _safe_format("empty {} braces", foo="bar")
        assert result == "empty {} braces"

    def test_css_colon_format_spec_untouched(self):
        # {color: red} — colon after identifier; should not be treated as {color}
        result = _safe_format(".cls {color: red;}", color="blue")
        # _safe_format only matches {identifier} (no colon); the CSS block must survive
        assert "{color: red;}" in result

    def test_integer_value_converted_to_string(self):
        result = _safe_format("pages: {pages}", pages=3)
        assert result == "pages: 3"


# ---------------------------------------------------------------------------
# B. Template YAML loading and format()
# ---------------------------------------------------------------------------

TEMPLATE_DIR = Path("templates/benchmark")

BENCHMARK_TEMPLATES = [
    "lab_report.yaml",
    "ct_report.yaml",
    "mri_report.yaml",
    "gynecological_report.yaml",
]


class TestTemplateLoading:
    @pytest.mark.parametrize("filename", BENCHMARK_TEMPLATES)
    def test_template_loads_without_error(self, filename):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / filename)
        assert handler is not None

    @pytest.mark.parametrize("filename", BENCHMARK_TEMPLATES)
    def test_template_has_two_messages(self, filename):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / filename)
        assert len(handler.messages) == 2

    @pytest.mark.parametrize("filename", BENCHMARK_TEMPLATES)
    def test_template_roles(self, filename):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / filename)
        assert handler.messages[0]["role"] == "system"
        assert handler.messages[1]["role"] == "user"

    @pytest.mark.parametrize("filename", BENCHMARK_TEMPLATES)
    def test_system_prompt_non_trivial(self, filename):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / filename)
        system_content = handler.messages[0]["content"]
        assert len(system_content) > 500, "System prompt seems too short"

    @pytest.mark.parametrize("filename", BENCHMARK_TEMPLATES)
    def test_output_fields_defined(self, filename):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / filename)
        assert len(handler.output_fields) == 1
        assert handler.output_fields[0].name == "html"
        assert handler.output_fields[0].pattern is None
        assert handler.output_fields[0].required is True


class TestTemplateFormat:
    def _lab_kwargs(self):
        return dict(
            pages=2,
            page_context="standard outpatient workup",
            country="UK",
            sex="Female",
            age_range="40s",
            clinical_context="routine annual health check",
            panels="CBC, Lipid Profile",
            style="plain institutional",
        )

    def _ct_kwargs(self):
        return dict(
            pages=1,
            page_context="single-region routine exam",
            country="US",
            sex="Male",
            age_range="60s",
            exam_type="CT Head without contrast",
            indication="severe headache",
            style="plain institutional",
        )

    def _mri_kwargs(self):
        return dict(
            pages=2,
            page_context="complex multi-sequence exam",
            country="Australia",
            sex="Female",
            age_range="30s",
            exam_type="MRI Lumbar Spine without contrast",
            indication="lower back pain",
            style="plain institutional",
        )

    def _gyn_kwargs(self):
        return dict(
            pages=1,
            page_context="brief follow-up visit",
            country="Canada",
            age_range="40s",
            scenario="endometriosis stage III",
            style="plain institutional",
        )

    def test_lab_format_fills_country(self):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "lab_report.yaml")
        msgs = handler.format(**self._lab_kwargs())
        assert "UK" in msgs[1]["content"]

    def test_lab_format_fills_page_context(self):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "lab_report.yaml")
        msgs = handler.format(**self._lab_kwargs())
        assert "standard outpatient workup" in msgs[1]["content"]

    def test_lab_system_prompt_has_no_kwarg_placeholders_remaining(self):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "lab_report.yaml")
        msgs = handler.format(**self._lab_kwargs())
        # After format(), no {identifier} slots should remain in the user message
        import re
        remaining = re.findall(r'\{([A-Za-z_]\w*)\}', msgs[1]["content"])
        assert remaining == [], f"Unfilled placeholders: {remaining}"

    def test_ct_format_fills_exam_type(self):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "ct_report.yaml")
        msgs = handler.format(**self._ct_kwargs())
        assert "CT Head without contrast" in msgs[1]["content"]

    def test_mri_format_fills_indication(self):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "mri_report.yaml")
        msgs = handler.format(**self._mri_kwargs())
        assert "lower back pain" in msgs[1]["content"]

    def test_gyn_format_fills_scenario(self):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "gynecological_report.yaml")
        msgs = handler.format(**self._gyn_kwargs())
        assert "endometriosis stage III" in msgs[1]["content"]

    def test_gyn_reminder_present(self):
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "gynecological_report.yaml")
        msgs = handler.format(**self._gyn_kwargs())
        assert "REMINDER" in msgs[1]["content"]

    def test_gyn_extra_kwarg_sex_silently_ignored(self):
        """Gynecology template has no {sex} slot; passing it must not raise."""
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "gynecological_report.yaml")
        kwargs = self._gyn_kwargs()
        kwargs["sex"] = "Female"  # not a grid axis for gynecology
        msgs = handler.format(**kwargs)  # must not raise
        assert "REMINDER" in msgs[1]["content"]

    def test_system_prompt_css_braces_survive_format(self):
        """CSS blocks in system prompts must not be mangled by _safe_format."""
        handler = TemplateHandler.from_yaml(TEMPLATE_DIR / "mri_report.yaml")
        msgs = handler.format(**self._mri_kwargs())
        # The system prompt contains CSS with curly braces; they must survive
        assert ".page {" in msgs[0]["content"] or "body {" in msgs[0]["content"]


# ---------------------------------------------------------------------------
# C. ConfigArgumentParser + config file loading
# ---------------------------------------------------------------------------

class TestConfigParser:
    def test_defaults_loaded_from_config(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["generate_benchmark.py", "--config", "dataprep/generate_benchmark"],
        )
        args = _build_parser().parse_args()
        assert args.model == "gemini-3-flash-preview"
        assert args.retry_max == 3
        assert args.retry_wait == 10
        assert args.self_correct_max == 1
        assert args.output_dir == "output/benchmark"

    def test_dry_run_default_false(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["generate_benchmark.py", "--config", "dataprep/generate_benchmark"],
        )
        args = _build_parser().parse_args()
        assert args.dry_run is False

    def test_dry_run_flag_sets_true(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            [
                "generate_benchmark.py",
                "--config", "dataprep/generate_benchmark",
                "--dry_run",
            ],
        )
        args = _build_parser().parse_args()
        assert args.dry_run is True

    def test_cli_overrides_config(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            [
                "generate_benchmark.py",
                "--config", "dataprep/generate_benchmark",
                "--model", "gemini-2.5-pro",
                "--retry_max", "5",
            ],
        )
        args = _build_parser().parse_args()
        assert args.model == "gemini-2.5-pro"
        assert args.retry_max == 5

    def test_types_argument_parsed(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            [
                "generate_benchmark.py",
                "--config", "dataprep/generate_benchmark",
                "--types", "mri", "lab",
            ],
        )
        args = _build_parser().parse_args()
        assert set(args.types) == {"mri", "lab"}


# ---------------------------------------------------------------------------
# D. build_jobs and job_key
# ---------------------------------------------------------------------------

class TestBuildJobs:
    def test_build_jobs_all_same_type(self):
        jobs = build_jobs(["gynecology"])
        assert all(j["doc_type"] == "gynecology" for j in jobs)

    def test_build_jobs_non_empty(self):
        for doc_type in DOC_TYPES:
            jobs = build_jobs([doc_type])
            assert len(jobs) > 0, f"build_jobs returned empty list for {doc_type}"

    def test_build_jobs_multiple_types(self):
        jobs = build_jobs(["lab", "ct"])
        types_seen = {j["doc_type"] for j in jobs}
        assert types_seen == {"lab", "ct"}

    def test_job_key_deterministic(self):
        jobs = build_jobs(["mri"])
        j = jobs[0]
        assert job_key(j) == job_key(dict(j))

    def test_job_key_differs_for_different_jobs(self):
        jobs = build_jobs(["lab"])
        assert job_key(jobs[0]) != job_key(jobs[-1])

    def test_each_job_has_doc_type(self):
        jobs = build_jobs(["ct"])
        assert all("doc_type" in j for j in jobs)

    def test_page_context_keys_cover_all_grid_pages(self):
        """Every pages value in each doc type's param_grid has a page_context entry."""
        for doc_type, cfg in DOC_TYPES.items():
            pages_in_grid = cfg["param_grid"]["pages"]
            ctx = cfg["page_context"]
            for p in pages_in_grid:
                assert p in ctx, (
                    f"DOC_TYPES['{doc_type}']['page_context'] is missing key {p}"
                )


# ---------------------------------------------------------------------------
# E. extract_html
# ---------------------------------------------------------------------------

class TestExtractHtml:
    def test_strips_markdown_html_fence(self):
        raw = "```html\n<!DOCTYPE html>\n<body/>\n```"
        result = extract_html(raw)
        assert result.startswith("<!DOCTYPE html>")
        assert "```" not in result

    def test_strips_plain_backtick_fence(self):
        raw = "```\n<!DOCTYPE html>\n<body/>\n```"
        result = extract_html(raw)
        assert result.startswith("<!DOCTYPE html>")

    def test_no_fence_passthrough(self):
        raw = "<!DOCTYPE html>\n<body/>"
        result = extract_html(raw)
        assert result == raw.strip()

    def test_strips_leading_trailing_whitespace(self):
        raw = "  \n<!DOCTYPE html><body/>\n  "
        result = extract_html(raw)
        assert result == "<!DOCTYPE html><body/>"


# ---------------------------------------------------------------------------
# F. count_html_pages
# ---------------------------------------------------------------------------

class TestCountHtmlPages:
    def test_two_pages(self):
        html = '<div class="page">p1</div><div class="page">p2</div>'
        assert count_html_pages(html) == 2

    def test_single_page(self):
        html = '<html><body><div class="page">content</div></body></html>'
        assert count_html_pages(html) == 1

    def test_zero_pages(self):
        html = "<html><body><p>no page divs</p></body></html>"
        assert count_html_pages(html) == 0

    def test_nested_page_divs_counted(self):
        # Each div.page is counted individually regardless of nesting
        html = (
            '<div class="page">'
            '  <div class="page">inner</div>'
            '</div>'
        )
        assert count_html_pages(html) == 2
