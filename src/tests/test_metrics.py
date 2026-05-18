import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from inference.metrics import (
    enclosing_box, bbox_iou, text_char_f1, normalized_edit_distance,
    match_entities_by_bbox, match_entities_by_text, match_entities_by_exact_text,
    compute_metrics,
)


# ── bbox_iou ─────────────────────────────────────────────────────────────────

def test_bbox_iou_perfect():
    box = [[0.1, 0.1, 0.5, 0.5]]
    assert bbox_iou(box, box) == pytest.approx(1.0)

def test_bbox_iou_no_overlap():
    p = [[0.0, 0.0, 0.2, 0.2]]
    g = [[0.5, 0.5, 0.9, 0.9]]
    assert bbox_iou(p, g) == pytest.approx(0.0)

def test_bbox_iou_partial():
    # pred: y[0,0.5] x[0,0.5]  area=0.25
    # gt:   y[0.25,0.75] x[0.25,0.75]  area=0.25
    # inter: y[0.25,0.5] x[0.25,0.5]  area=0.0625
    # union: 0.25+0.25-0.0625=0.4375
    p = [[0.0, 0.0, 0.5, 0.5]]
    g = [[0.25, 0.25, 0.75, 0.75]]
    expected = 0.0625 / 0.4375
    assert bbox_iou(p, g) == pytest.approx(expected, rel=1e-5)

def test_bbox_iou_multi_box_enclosing():
    # Two-box entity: total enclosing rect is [0, 0, 0.6, 0.6]
    pred = [[0.0, 0.0, 0.3, 0.3], [0.3, 0.3, 0.6, 0.6]]
    gt = [[0.0, 0.0, 0.6, 0.6]]
    # enclosing of pred == gt → IoU = 1.0
    assert bbox_iou(pred, gt) == pytest.approx(1.0)

def test_bbox_iou_zero_area():
    p = [[0.5, 0.5, 0.5, 0.5]]  # zero area
    g = [[0.5, 0.5, 0.5, 0.5]]
    assert bbox_iou(p, g) == pytest.approx(0.0)


# ── text_char_f1 ──────────────────────────────────────────────────────────────

def test_char_f1_exact():
    assert text_char_f1("Mario Rossi", "Mario Rossi") == pytest.approx(1.0)

def test_char_f1_case_insensitive():
    assert text_char_f1("mario rossi", "Mario Rossi") == pytest.approx(1.0)

def test_char_f1_partial():
    # pred="abc" gt="abcd"  common=3 chars
    # precision=3/3=1.0  recall=3/4=0.75  F1=2*0.75/1.75=0.857
    f1 = text_char_f1("abc", "abcd")
    assert f1 == pytest.approx(2 * 1.0 * 0.75 / 1.75, rel=1e-5)

def test_char_f1_no_overlap():
    assert text_char_f1("xyz", "abc") == pytest.approx(0.0)

def test_char_f1_both_empty():
    assert text_char_f1("", "") == pytest.approx(1.0)

def test_char_f1_one_empty():
    assert text_char_f1("", "abc") == pytest.approx(0.0)
    assert text_char_f1("abc", "") == pytest.approx(0.0)


# ── normalized_edit_distance ──────────────────────────────────────────────────

def test_ned_identical():
    assert normalized_edit_distance("hello", "hello") == pytest.approx(0.0)

def test_ned_completely_different():
    assert normalized_edit_distance("abc", "xyz") == pytest.approx(1.0)

def test_ned_one_edit():
    # "cat" → "bat": 1 substitution; max len = 3; NED = 1/3
    assert normalized_edit_distance("cat", "bat") == pytest.approx(1 / 3, rel=1e-5)

def test_ned_empty():
    assert normalized_edit_distance("", "") == pytest.approx(0.0)
    assert normalized_edit_distance("", "abc") == pytest.approx(1.0)


# ── match_entities_by_bbox ────────────────────────────────────────────────────

def _ent(label, text, y1, x1, y2, x2):
    return {"label": label, "text": text, "bboxes": [[y1, x1, y2, x2]]}

def test_match_bbox_perfect():
    pred = [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]
    gt   = [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]
    matched, fp, fn = match_entities_by_bbox(pred, gt, iou_threshold=0.5)
    assert len(matched) == 1 and len(fp) == 0 and len(fn) == 0

def test_match_bbox_no_overlap():
    pred = [_ent("NAME:PATIENT", "Rossi", 0.0, 0.0, 0.1, 0.1)]
    gt   = [_ent("NAME:PATIENT", "Bianchi", 0.8, 0.8, 0.9, 0.9)]
    matched, fp, fn = match_entities_by_bbox(pred, gt, iou_threshold=0.5)
    assert len(matched) == 0 and len(fp) == 1 and len(fn) == 1

def test_match_bbox_label_mismatch():
    pred = [_ent("DATETIME", "01/01/2000", 0.1, 0.1, 0.2, 0.3)]
    gt   = [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]
    matched, fp, fn = match_entities_by_bbox(pred, gt)
    assert len(matched) == 0 and len(fp) == 1 and len(fn) == 1

def test_match_bbox_empty_pred():
    gt = [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]
    matched, fp, fn = match_entities_by_bbox([], gt)
    assert len(matched) == 0 and len(fp) == 0 and len(fn) == 1

def test_match_bbox_empty_gt():
    pred = [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]
    matched, fp, fn = match_entities_by_bbox(pred, [])
    assert len(matched) == 0 and len(fp) == 1 and len(fn) == 0


# ── match_entities_by_text ────────────────────────────────────────────────────

def test_match_text_perfect():
    pred = [_ent("NAME:PATIENT", "Mario Rossi", 0.0, 0.0, 0.1, 0.1)]
    gt   = [_ent("NAME:PATIENT", "Mario Rossi", 0.5, 0.5, 0.9, 0.9)]  # different bbox
    matched, fp, fn = match_entities_by_text(pred, gt)
    assert len(matched) == 1 and len(fp) == 0 and len(fn) == 0

def test_match_text_partial_text():
    # char F1 of "Mario" vs "Mario Rossi" > 0.5
    pred = [_ent("NAME:PATIENT", "Mario", 0.0, 0.0, 0.1, 0.1)]
    gt   = [_ent("NAME:PATIENT", "Mario Rossi", 0.0, 0.0, 0.1, 0.1)]
    matched, _, _ = match_entities_by_text(pred, gt, text_threshold=0.5)
    assert len(matched) == 1


# ── compute_metrics — structure ───────────────────────────────────────────────

def _sample_preds():
    return [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]

def _sample_gt():
    return [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]

def test_compute_metrics_output_keys():
    m = compute_metrics([[]], [[]], iou_thresholds=[0.5])
    for key in ["summary", "text_extraction", "bbox_localization",
                "hallucination_rate", "miss_rate", "coarse_f1",
                "label_confusion", "format_compliance"]:
        assert key in m, f"Missing top-level key: {key}"

def test_summary_block_keys():
    m = compute_metrics([[]], [[]], iou_thresholds=[0.5])
    for key in ["detection_micro_f1", "detection_macro_f1", "avg_e2e_f1",
                "unconditional_mean_iou", "mean_iou", "char_f1",
                "exact_match_rate", "spatial_char_f1", "spatial_exact_match_rate",
                "hallucination_rate", "miss_rate", "format_compliance"]:
        assert key in m["summary"], f"Missing key in summary: {key}"

def test_text_extraction_keys():
    m = compute_metrics([[]], [[]], iou_thresholds=[0.5])
    te = m["text_extraction"]
    for key in ["detection", "span_exact", "char_f1", "edit_distance", "exact_match_rate"]:
        assert key in te, f"Missing key in text_extraction: {key}"

def test_bbox_localization_keys():
    m = compute_metrics([[]], [[]], iou_thresholds=[0.5])
    bl = m["bbox_localization"]
    for key in ["avg_e2e_f1", "mean_iou", "unconditional_mean_iou", "end_to_end",
                "spatial_char_f1", "spatial_exact_match_rate"]:
        assert key in bl, f"Missing key in bbox_localization: {key}"

def test_e2e_threshold_keys_are_strings():
    m = compute_metrics([[]], [[]], iou_thresholds=[0.25, 0.5, 0.75])
    e2e = m["bbox_localization"]["end_to_end"]
    assert "@0.25" in e2e and "@0.5" in e2e and "@0.75" in e2e
    assert 0.5 not in e2e  # float keys must be gone

def test_per_label_breakdown_removed():
    m = compute_metrics([[]], [[]], iou_thresholds=[0.5])
    assert "per_label_breakdown" not in m


# ── compute_metrics — correctness ─────────────────────────────────────────────

def test_compute_metrics_perfect():
    preds = [_sample_preds()]
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    assert m["text_extraction"]["detection"]["micro"]["f1"] == pytest.approx(1.0)
    assert m["bbox_localization"]["end_to_end"]["@0.5"]["micro"]["f1"] == pytest.approx(1.0)
    assert m["text_extraction"]["exact_match_rate"] == pytest.approx(1.0)
    assert m["bbox_localization"]["mean_iou"] == pytest.approx(1.0)
    assert m["summary"]["hallucination_rate"] == pytest.approx(0.0)
    assert m["summary"]["miss_rate"] == pytest.approx(0.0)

def test_compute_metrics_all_miss():
    preds = [[]]  # model predicts nothing
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    assert m["text_extraction"]["detection"]["micro"]["recall"] == pytest.approx(0.0)
    assert m["summary"]["miss_rate"] == pytest.approx(1.0)
    assert m["summary"]["hallucination_rate"] == pytest.approx(0.0)

def test_compute_metrics_all_hallucination():
    preds = [_sample_preds()]
    gts   = [[]]  # no GT entities
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    assert m["text_extraction"]["detection"]["micro"]["precision"] == pytest.approx(0.0)
    assert m["summary"]["hallucination_rate"] == pytest.approx(1.0)
    assert m["summary"]["miss_rate"] == pytest.approx(0.0)

def test_compute_metrics_format_compliance():
    m = compute_metrics([[]], [[]], format_compliance=0.75)
    assert m["format_compliance"] == pytest.approx(0.75)
    assert m["summary"]["format_compliance"] == pytest.approx(0.75)

def test_compute_metrics_multiple_samples():
    preds = [_sample_preds(), []]
    gts   = [_sample_gt(), _sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    det = m["text_extraction"]["detection"]["micro"]
    # 1 TP, 0 FP, 1 FN → P=1.0, R=0.5, F1=0.667
    assert det["tp"] == 1 and det["fn"] == 1
    assert det["f1"] == pytest.approx(2/3, rel=1e-4)

def test_compute_metrics_coarse_grouping():
    preds = [[_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]]
    gts   = [[_ent("NAME:STAFF", "Bianchi", 0.5, 0.5, 0.6, 0.7)]]
    m = compute_metrics(preds, gts)
    assert "NAME" in m["coarse_f1"]


# ── macro P/R ─────────────────────────────────────────────────────────────────

def test_entity_detection_macro_prf():
    preds = [_sample_preds()]
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    det = m["text_extraction"]["detection"]
    assert det["macro_precision"] == pytest.approx(1.0)
    assert det["macro_recall"]    == pytest.approx(1.0)
    assert det["macro_f1"]        == pytest.approx(1.0)

def test_macro_all_vs_supported():
    preds = [_sample_preds()]
    gts   = [_sample_gt()]
    label_set = ["NAME:PATIENT", "DATETIME"]  # DATETIME has no GT → drags macro_all down
    m = compute_metrics(preds, gts, iou_thresholds=[0.5], label_set=label_set)
    det = m["text_extraction"]["detection"]
    assert det["macro_f1"]     == pytest.approx(1.0)
    assert det["macro_f1_all"] == pytest.approx(0.5)
    assert det["macro_f1_all"] <= det["macro_f1"]

def test_e2e_macro_prf_present():
    preds = [_sample_preds()]
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    thr_result = m["bbox_localization"]["end_to_end"]["@0.5"]
    for key in ("macro_f1", "macro_precision", "macro_recall", "macro_f1_all"):
        assert key in thr_result, f"Missing key in end_to_end[@0.5]: {key}"


# ── match_entities_by_exact_text & span_exact ─────────────────────────────────

def test_match_exact_text_perfect():
    pred = [_ent("NAME:PATIENT", "Mario Rossi", 0.0, 0.0, 0.1, 0.1)]
    gt   = [_ent("NAME:PATIENT", "Mario Rossi", 0.5, 0.5, 0.9, 0.9)]
    matched, fp, fn = match_entities_by_exact_text(pred, gt)
    assert len(matched) == 1 and len(fp) == 0 and len(fn) == 0

def test_match_exact_text_case_insensitive():
    pred = [_ent("NAME:PATIENT", "mario rossi", 0.0, 0.0, 0.1, 0.1)]
    gt   = [_ent("NAME:PATIENT", "Mario Rossi", 0.0, 0.0, 0.1, 0.1)]
    matched, fp, fn = match_entities_by_exact_text(pred, gt)
    assert len(matched) == 1 and len(fp) == 0 and len(fn) == 0

def test_match_exact_text_partial_no_match():
    pred = [_ent("NAME:PATIENT", "Mario", 0.0, 0.0, 0.1, 0.1)]
    gt   = [_ent("NAME:PATIENT", "Mario Rossi", 0.0, 0.0, 0.1, 0.1)]
    matched, fp, fn = match_entities_by_exact_text(pred, gt)
    assert len(matched) == 0 and len(fp) == 1 and len(fn) == 1

def test_span_exact_f1_perfect():
    preds = [_sample_preds()]
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    assert m["text_extraction"]["span_exact"]["micro"]["f1"] == pytest.approx(1.0)

def test_span_exact_f1_partial_text_no_match():
    pred = [_ent("NAME:PATIENT", "Mario", 0.0, 0.0, 0.1, 0.1)]
    gt   = [_ent("NAME:PATIENT", "Mario Rossi", 0.0, 0.0, 0.1, 0.1)]
    m = compute_metrics([pred], [gt], iou_thresholds=[0.5])
    assert m["text_extraction"]["span_exact"]["micro"]["f1"] == pytest.approx(0.0)
    assert m["text_extraction"]["detection"]["micro"]["f1"] > 0.0

def test_span_exact_output_structure():
    m = compute_metrics([[]], [[]], iou_thresholds=[0.5])
    ex = m["text_extraction"]["span_exact"]
    for key in ("micro", "per_label", "macro_f1", "macro_precision", "macro_recall",
                "macro_f1_all", "macro_precision_all", "macro_recall_all"):
        assert key in ex, f"Missing key in span_exact: {key}"


# ── new metrics ───────────────────────────────────────────────────────────────

def test_avg_e2e_f1_present_and_bounded():
    preds = [_sample_preds()]
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.25, 0.5, 0.75])
    v = m["summary"]["avg_e2e_f1"]
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0

def test_avg_e2e_f1_perfect():
    preds = [_sample_preds()]
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.25, 0.5, 0.75])
    assert m["summary"]["avg_e2e_f1"] == pytest.approx(1.0)

def test_avg_e2e_f1_is_mean_of_thresholds():
    # one pred overlaps at IoU > 0.25 but not > 0.75
    # pred y[0,0.5] x[0,0.5], gt y[0.25,0.75] x[0.25,0.75]
    pred = [_ent("NAME:PATIENT", "Rossi", 0.0, 0.0, 0.5, 0.5)]
    gt   = [_ent("NAME:PATIENT", "Rossi", 0.25, 0.25, 0.75, 0.75)]
    m = compute_metrics([pred], [gt], iou_thresholds=[0.25, 0.5, 0.75])
    e2e = m["bbox_localization"]["end_to_end"]
    f1s = [e2e[k]["micro"]["f1"] for k in ("@0.25", "@0.5", "@0.75")]
    assert m["summary"]["avg_e2e_f1"] == pytest.approx(sum(f1s) / 3, rel=1e-5)

def test_unconditional_mean_iou_penalises_misses():
    # 1 matching pred (IoU=1.0) + 3 GT entities with no pred → avg = 1/4
    pred = [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]
    gts  = [
        _ent("NAME:PATIENT", "Rossi",   0.1, 0.1, 0.2, 0.3),  # matched, IoU=1.0
        _ent("DATETIME",     "01/2024", 0.5, 0.5, 0.6, 0.6),  # no pred
        _ent("DATETIME",     "02/2024", 0.7, 0.7, 0.8, 0.8),  # no pred
        _ent("DATETIME",     "03/2024", 0.3, 0.3, 0.4, 0.4),  # no pred
    ]
    m = compute_metrics([pred], [gts], iou_thresholds=[0.5])
    assert m["summary"]["unconditional_mean_iou"] == pytest.approx(0.25, rel=1e-4)
    # mean_iou should be 1.0 (only the matched pair)
    assert m["summary"]["mean_iou"] == pytest.approx(1.0)

def test_unconditional_mean_iou_zero_when_all_miss():
    preds = [[]]
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    assert m["summary"]["unconditional_mean_iou"] == pytest.approx(0.0)

def test_text_quality_decoupled_from_iou():
    # pred has correct text but bbox in completely different location
    pred = [_ent("NAME:PATIENT", "Rossi", 0.8, 0.8, 0.9, 0.9)]
    gt   = [_ent("NAME:PATIENT", "Rossi", 0.0, 0.0, 0.1, 0.1)]
    m = compute_metrics([pred], [gt], iou_thresholds=[0.5])
    # text-based detection matches (same text) → char_f1 should be 1.0
    assert m["text_extraction"]["char_f1"] == pytest.approx(1.0)
    assert m["text_extraction"]["exact_match_rate"] == pytest.approx(1.0)
    # but bbox-based e2e fails (no overlap)
    assert m["bbox_localization"]["end_to_end"]["@0.5"]["micro"]["f1"] == pytest.approx(0.0)
    # mean_iou is 0 (no IoU-matched pairs)
    assert m["bbox_localization"]["mean_iou"] == pytest.approx(0.0)
    # spatial_char_f1 is 0 — no spatially matched pairs exist
    assert m["bbox_localization"]["spatial_char_f1"] == pytest.approx(0.0)


def test_spatial_char_f1_perfect():
    """Correct box AND correct text → spatial metrics = 1.0."""
    pred = [{"label": "NAME:PATIENT", "text": "Mario Rossi", "bboxes": [[0.1, 0.1, 0.3, 0.4]]}]
    gt   = [{"label": "NAME:PATIENT", "text": "Mario Rossi", "bboxes": [[0.1, 0.1, 0.3, 0.4]]}]
    m = compute_metrics([pred], [gt], iou_thresholds=[0.5])
    assert m["bbox_localization"]["spatial_char_f1"] == pytest.approx(1.0)
    assert m["bbox_localization"]["spatial_exact_match_rate"] == pytest.approx(1.0)
    assert m["summary"]["spatial_char_f1"] == pytest.approx(1.0)


def test_spatial_char_f1_box_ok_text_wrong():
    """Correct box but partial text → spatial pair exists, char_f1 partial, exact_match = 0."""
    pred = [{"label": "NAME:PATIENT", "text": "M. Rossi",    "bboxes": [[0.1, 0.1, 0.3, 0.4]]}]
    gt   = [{"label": "NAME:PATIENT", "text": "Mario Rossi", "bboxes": [[0.1, 0.1, 0.3, 0.4]]}]
    m = compute_metrics([pred], [gt], iou_thresholds=[0.5])
    # IoU = 1.0 → bbox match exists → spatial pair computed
    assert 0.0 < m["bbox_localization"]["spatial_char_f1"] < 1.0
    assert m["bbox_localization"]["spatial_exact_match_rate"] == pytest.approx(0.0)
