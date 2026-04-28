from collections import defaultdict

import numpy as np


# ── Geometry ──────────────────────────────────────────────────────────────────

def enclosing_box(bboxes: list) -> list:
    """Return [y_min, x_min, y_max, x_max] of the axis-aligned union rectangle."""
    arr = np.array(bboxes, dtype=float)
    return [arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()]


def bbox_iou(pred_bboxes: list, gt_bboxes: list) -> float:
    """
    IoU between the enclosing rectangles of two multi-box entities.
    Both sides are reduced to their axis-aligned bounding rectangle first.
    Known simplification: gaps between boxes are counted as overlap area.
    """
    pb = enclosing_box(pred_bboxes)
    gb = enclosing_box(gt_bboxes)
    inter_y1, inter_x1 = max(pb[0], gb[0]), max(pb[1], gb[1])
    inter_y2, inter_x2 = min(pb[2], gb[2]), min(pb[3], gb[3])
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_area = inter_h * inter_w
    pb_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
    gb_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
    union_area = pb_area + gb_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ── Text quality ──────────────────────────────────────────────────────────────

def text_char_f1(pred: str, gt: str) -> float:
    """SQuAD-style character-level F1. Case- and whitespace-insensitive."""
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    from collections import Counter
    pc, gc = Counter(pred), Counter(gt)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    p = common / len(pred)
    r = common / len(gt)
    return 2 * p * r / (p + r)


def normalized_edit_distance(pred: str, gt: str) -> float:
    """Levenshtein distance normalised by max(len(pred), len(gt))."""
    pred, gt = pred.strip().lower(), gt.strip().lower()
    m, n = len(pred), len(gt)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if pred[i - 1] == gt[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / max(m, n)


# ── Entity matching ───────────────────────────────────────────────────────────

def _greedy_match(pred_entities, gt_entities, score_fn, threshold):
    """
    Greedy bipartite matching. score_fn(pred, gt) -> float.
    Candidates must share the same label and exceed threshold.
    Returns (matched_pairs, unmatched_preds, unmatched_gts).
    """
    candidates = []
    for pi, p in enumerate(pred_entities):
        for gi, g in enumerate(gt_entities):
            if p["label"] == g["label"]:
                score = score_fn(p, g)
                if score > threshold:
                    candidates.append((score, pi, gi))
    candidates.sort(reverse=True)

    used_p, used_g = set(), set()
    matched_pairs = []
    for _, pi, gi in candidates:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matched_pairs.append((pred_entities[pi], gt_entities[gi]))

    unmatched_preds = [p for i, p in enumerate(pred_entities) if i not in used_p]
    unmatched_gts = [g for i, g in enumerate(gt_entities) if i not in used_g]
    return matched_pairs, unmatched_preds, unmatched_gts


def match_entities_by_bbox(pred_entities, gt_entities, iou_threshold=0.5):
    """Match entities by label + bbox IoU > iou_threshold."""
    return _greedy_match(pred_entities, gt_entities,
                         lambda p, g: bbox_iou(p["bboxes"], g["bboxes"]),
                         iou_threshold)


def match_entities_by_text(pred_entities, gt_entities, text_threshold=0.5):
    """Match entities by label + text char-F1 > text_threshold."""
    return _greedy_match(pred_entities, gt_entities,
                         lambda p, g: text_char_f1(p["text"], g["text"]),
                         text_threshold)


# ── PRF helper ────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ── Main metric computation ───────────────────────────────────────────────────

def compute_metrics(
    all_predictions: list,
    all_gt: list,
    iou_thresholds: list = None,
    label_set: list = None,
    format_compliance: float = None,
) -> dict:
    """
    Compute the full benchmark metric suite.

    Args:
        all_predictions: List (one per sample) of entity lists.
                         Each entity: {"label": str, "text": str, "bboxes": [[y,x,y,x], ...]}.
        all_gt:          Same structure, ground-truth annotations.
        iou_thresholds:  IoU thresholds for end-to-end F1 (default [0.25, 0.5, 0.75]).
        label_set:       All expected label names. Inferred from data if None.
        format_compliance: Pre-computed fraction of parseable outputs (0–1), or None.

    Returns:
        Nested dict with keys:
          entity_detection_f1, end_to_end_f1, char_f1, edit_distance,
          exact_match_rate, mean_iou, hallucination_rate, miss_rate,
          coarse_f1, per_label_breakdown, format_compliance, label_confusion.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.25, 0.5, 0.75]

    assert len(all_predictions) == len(all_gt), "predictions and GT must have the same length"

    all_labels = label_set or sorted(
        {e["label"] for sample in all_gt for e in sample}
        | {e["label"] for sample in all_predictions for e in sample}
    )

    # Accumulators
    det_tp = det_fp = det_fn = 0
    det_per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    e2e_totals = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in iou_thresholds}
    e2e_per_label = {thr: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) for thr in iou_thresholds}

    char_f1_scores, edit_dist_scores, iou_scores = [], [], []
    exact_matches = 0
    total_gt = 0
    label_confusion = defaultdict(lambda: defaultdict(int))

    for preds, gts in zip(all_predictions, all_gt):
        total_gt += len(gts)

        # ── Detection (text-based matching) ───────────────────────────────────
        matched_d, unmatched_pd, unmatched_gd = match_entities_by_text(preds, gts)
        det_tp += len(matched_d)
        det_fp += len(unmatched_pd)
        det_fn += len(unmatched_gd)
        for _, g in matched_d:
            det_per_label[g["label"]]["tp"] += 1
        for p in unmatched_pd:
            det_per_label[p["label"]]["fp"] += 1
        for g in unmatched_gd:
            det_per_label[g["label"]]["fn"] += 1

        # ── End-to-end (bbox-based matching at each threshold) ────────────────
        for thr in iou_thresholds:
            matched_e, unmatched_pe, unmatched_ge = match_entities_by_bbox(preds, gts, thr)
            e2e_totals[thr]["tp"] += len(matched_e)
            e2e_totals[thr]["fp"] += len(unmatched_pe)
            e2e_totals[thr]["fn"] += len(unmatched_ge)
            for _, g in matched_e:
                e2e_per_label[thr][g["label"]]["tp"] += 1
            for p in unmatched_pe:
                e2e_per_label[thr][p["label"]]["fp"] += 1
            for g in unmatched_ge:
                e2e_per_label[thr][g["label"]]["fn"] += 1

        # ── Text quality and IoU for pairs matched at IoU>0.5 ─────────────────
        matched_05, unmatched_pe_05, _ = match_entities_by_bbox(preds, gts, 0.5)
        for p, g in matched_05:
            char_f1_scores.append(text_char_f1(p["text"], g["text"]))
            edit_dist_scores.append(normalized_edit_distance(p["text"], g["text"]))
            iou_scores.append(bbox_iou(p["bboxes"], g["bboxes"]))
            if p["text"].strip().lower() == g["text"].strip().lower():
                exact_matches += 1

        # ── Label confusion: FP preds overlapping a differently-labelled GT ───
        for p in unmatched_pe_05:
            best_iou, best_label = 0.0, None
            for g in gts:
                iou = bbox_iou(p["bboxes"], g["bboxes"])
                if iou > best_iou:
                    best_iou, best_label = iou, g["label"]
            if best_label and best_iou > 0.3:
                label_confusion[best_label][p["label"]] += 1

    # ── Aggregate detection metrics ───────────────────────────────────────────
    det_micro = _prf(det_tp, det_fp, det_fn)
    det_per_label_prf = {}
    macro_sum, macro_n = 0.0, 0
    for label in all_labels:
        cnt = det_per_label[label]
        prf = _prf(cnt["tp"], cnt["fp"], cnt["fn"])
        det_per_label_prf[label] = prf
        if cnt["tp"] + cnt["fn"] > 0:
            macro_sum += prf["f1"]
            macro_n += 1
    det_macro_f1 = macro_sum / macro_n if macro_n > 0 else 0.0

    # ── Aggregate end-to-end metrics ──────────────────────────────────────────
    e2e_results = {}
    for thr in iou_thresholds:
        tot = e2e_totals[thr]
        micro = _prf(tot["tp"], tot["fp"], tot["fn"])
        per_label_prf = {}
        ms, mn = 0.0, 0
        for label in all_labels:
            cnt = e2e_per_label[thr][label]
            prf = _prf(cnt["tp"], cnt["fp"], cnt["fn"])
            per_label_prf[label] = prf
            if cnt["tp"] + cnt["fn"] > 0:
                ms += prf["f1"]
                mn += 1
        e2e_results[thr] = {"micro": micro, "macro_f1": ms / mn if mn > 0 else 0.0, "per_label": per_label_prf}

    # ── Coarse F1 (collapse label to parent before ":") ──────────────────────
    coarse = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for label, cnt in det_per_label.items():
        parent = label.split(":")[0] if ":" in label else label
        for k in ("tp", "fp", "fn"):
            coarse[parent][k] += cnt[k]
    coarse_prf = {c: _prf(v["tp"], v["fp"], v["fn"]) for c, v in coarse.items()}

    # ── Hallucination and miss rates ──────────────────────────────────────────
    hall_overall = det_fp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0
    miss_overall = det_fn / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0.0
    hall_per_label, miss_per_label = {}, {}
    for label in all_labels:
        cnt = det_per_label[label]
        tp, fp, fn = cnt["tp"], cnt["fp"], cnt["fn"]
        hall_per_label[label] = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        miss_per_label[label] = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "entity_detection_f1": {
            "micro": det_micro,
            "macro_f1": det_macro_f1,
            "per_label": det_per_label_prf,
        },
        "end_to_end_f1": e2e_results,
        "char_f1": float(np.mean(char_f1_scores)) if char_f1_scores else 0.0,
        "edit_distance": float(np.mean(edit_dist_scores)) if edit_dist_scores else 0.0,
        "exact_match_rate": exact_matches / total_gt if total_gt > 0 else 0.0,
        "mean_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "hallucination_rate": {"overall": hall_overall, "per_label": hall_per_label},
        "miss_rate": {"overall": miss_overall, "per_label": miss_per_label},
        "coarse_f1": coarse_prf,
        "per_label_breakdown": det_per_label_prf,
        "format_compliance": format_compliance,
        "label_confusion": {k: dict(v) for k, v in label_confusion.items()},
    }
