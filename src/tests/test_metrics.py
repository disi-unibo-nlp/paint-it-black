import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from inference.metrics import (
    enclosing_box, bbox_iou, text_char_f1, normalized_edit_distance,
    match_entities_by_bbox, match_entities_by_text, compute_metrics,
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
    pred = [_ent("DATE", "01/01/2000", 0.1, 0.1, 0.2, 0.3)]
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


# ── compute_metrics ──────────────────────────────────────────────────────────

def _sample_preds():
    return [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]

def _sample_gt():
    return [_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]

def test_compute_metrics_perfect():
    preds = [_sample_preds()]
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    assert m["entity_detection_f1"]["micro"]["f1"] == pytest.approx(1.0)
    assert m["end_to_end_f1"][0.5]["micro"]["f1"] == pytest.approx(1.0)
    assert m["exact_match_rate"] == pytest.approx(1.0)
    assert m["mean_iou"] == pytest.approx(1.0)
    assert m["hallucination_rate"]["overall"] == pytest.approx(0.0)
    assert m["miss_rate"]["overall"] == pytest.approx(0.0)

def test_compute_metrics_all_miss():
    preds = [[]]  # model predicts nothing
    gts   = [_sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    assert m["entity_detection_f1"]["micro"]["recall"] == pytest.approx(0.0)
    assert m["miss_rate"]["overall"] == pytest.approx(1.0)
    assert m["hallucination_rate"]["overall"] == pytest.approx(0.0)

def test_compute_metrics_all_hallucination():
    preds = [_sample_preds()]
    gts   = [[]]  # no GT entities
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    assert m["entity_detection_f1"]["micro"]["precision"] == pytest.approx(0.0)
    assert m["hallucination_rate"]["overall"] == pytest.approx(1.0)
    assert m["miss_rate"]["overall"] == pytest.approx(0.0)

def test_compute_metrics_output_keys():
    m = compute_metrics([[]], [[]], iou_thresholds=[0.5])
    for key in ["entity_detection_f1", "end_to_end_f1", "char_f1", "edit_distance",
                "exact_match_rate", "mean_iou", "hallucination_rate", "miss_rate",
                "coarse_f1", "per_label_breakdown", "format_compliance", "label_confusion"]:
        assert key in m, f"Missing key: {key}"

def test_compute_metrics_format_compliance():
    m = compute_metrics([[]], [[]], format_compliance=0.75)
    assert m["format_compliance"] == pytest.approx(0.75)

def test_compute_metrics_multiple_samples():
    preds = [_sample_preds(), []]
    gts   = [_sample_gt(), _sample_gt()]
    m = compute_metrics(preds, gts, iou_thresholds=[0.5])
    det = m["entity_detection_f1"]["micro"]
    # 1 TP, 0 FP, 1 FN → P=1.0, R=0.5, F1=0.667
    assert det["tp"] == 1 and det["fn"] == 1
    assert det["f1"] == pytest.approx(2/3, rel=1e-4)

def test_compute_metrics_coarse_grouping():
    preds = [[_ent("NAME:PATIENT", "Rossi", 0.1, 0.1, 0.2, 0.3)]]
    gts   = [[_ent("NAME:DOCTOR", "Bianchi", 0.5, 0.5, 0.6, 0.7)]]
    m = compute_metrics(preds, gts)
    # Both collapse to "NAME" under coarse_f1; but label mismatch means no TP
    assert "NAME" in m["coarse_f1"]
