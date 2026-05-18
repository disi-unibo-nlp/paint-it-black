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


def match_entities_by_exact_text(pred_entities, gt_entities):
    """Match entities by label + exact text (case-insensitive strip)."""
    return _greedy_match(
        pred_entities, gt_entities,
        lambda p, g: 1.0 if p["text"].strip().lower() == g["text"].strip().lower() else 0.0,
        threshold=0.5,
    )


# ── PRF helpers ───────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _macro_prf(per_label_counts: dict, all_labels: list) -> dict:
    """Macro-averaged P/R/F1 over two scopes: supported (≥1 GT) and all labels."""
    per_label_prf = {l: _prf(per_label_counts[l]["tp"],
                              per_label_counts[l]["fp"],
                              per_label_counts[l]["fn"]) for l in all_labels}
    supported = [prf for l, prf in per_label_prf.items()
                 if per_label_counts[l]["tp"] + per_label_counts[l]["fn"] > 0]
    all_prf = list(per_label_prf.values())

    def avg(lst, key):
        return sum(x[key] for x in lst) / len(lst) if lst else 0.0

    return {
        "macro_f1":            avg(supported, "f1"),
        "macro_precision":     avg(supported, "precision"),
        "macro_recall":        avg(supported, "recall"),
        "macro_f1_all":        avg(all_prf,   "f1"),
        "macro_precision_all": avg(all_prf,   "precision"),
        "macro_recall_all":    avg(all_prf,   "recall"),
    }


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
        Nested dict with top-level keys:
          summary, text_extraction, bbox_localization,
          hallucination_rate, miss_rate, coarse_f1, label_confusion, format_compliance.
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

    ex_tp = ex_fp = ex_fn = 0
    ex_per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # text_extraction accumulators — filled from text-matched pairs (bbox-independent)
    char_f1_scores, edit_dist_scores = [], []
    exact_matches = 0

    # bbox_localization accumulators
    iou_scores = []              # IoU of pairs matched at IoU > 0.5 (mean_iou)
    unconditional_iou_scores = []  # best same-label IoU per GT entity, 0 if unmatched
    spatial_char_f1_scores = []  # char_f1 on pairs matched at IoU > 0.5
    spatial_exact_matches = 0

    label_confusion = defaultdict(lambda: defaultdict(int))

    for preds, gts in zip(all_predictions, all_gt):

        # ── Detection (partial text-based matching) ───────────────────────────
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

        # ── Text quality — from text-matched pairs (decoupled from bbox) ──────
        for p, g in matched_d:
            char_f1_scores.append(text_char_f1(p["text"], g["text"]))
            edit_dist_scores.append(normalized_edit_distance(p["text"], g["text"]))
            if p["text"].strip().lower() == g["text"].strip().lower():
                exact_matches += 1

        # ── Exact span (exact text + label match) ─────────────────────────────
        matched_ex, unmatched_pex, unmatched_gex = match_entities_by_exact_text(preds, gts)
        ex_tp += len(matched_ex)
        ex_fp += len(unmatched_pex)
        ex_fn += len(unmatched_gex)
        for _, g in matched_ex:
            ex_per_label[g["label"]]["tp"] += 1
        for p in unmatched_pex:
            ex_per_label[p["label"]]["fp"] += 1
        for g in unmatched_gex:
            ex_per_label[g["label"]]["fn"] += 1

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

        # ── IoU scores at 0.5 (mean_iou) + label confusion ───────────────────
        matched_05, unmatched_pe_05, _ = match_entities_by_bbox(preds, gts, 0.5)
        for p, g in matched_05:
            iou_scores.append(bbox_iou(p["bboxes"], g["bboxes"]))
            spatial_char_f1_scores.append(text_char_f1(p["text"], g["text"]))
            if p["text"].strip().lower() == g["text"].strip().lower():
                spatial_exact_matches += 1

        for p in unmatched_pe_05:
            best_iou, best_label = 0.0, None
            for g in gts:
                iou = bbox_iou(p["bboxes"], g["bboxes"])
                if iou > best_iou:
                    best_iou, best_label = iou, g["label"]
            if best_label and best_iou > 0.3:
                label_confusion[best_label][p["label"]] += 1

        # ── Unconditional IoU: best same-label pred IoU per GT entity ─────────
        for g in gts:
            best = max(
                (bbox_iou(p["bboxes"], g["bboxes"]) for p in preds if p["label"] == g["label"]),
                default=0.0,
            )
            unconditional_iou_scores.append(best)

    # ── Aggregate detection metrics ───────────────────────────────────────────
    det_micro = _prf(det_tp, det_fp, det_fn)
    det_per_label_prf = {l: _prf(det_per_label[l]["tp"], det_per_label[l]["fp"], det_per_label[l]["fn"])
                         for l in all_labels}
    det_macro = _macro_prf(det_per_label, all_labels)

    # ── Aggregate exact-span metrics ──────────────────────────────────────────
    ex_micro = _prf(ex_tp, ex_fp, ex_fn)
    ex_per_label_prf = {l: _prf(ex_per_label[l]["tp"], ex_per_label[l]["fp"], ex_per_label[l]["fn"])
                        for l in all_labels}
    ex_macro = _macro_prf(ex_per_label, all_labels)

    # ── Aggregate end-to-end metrics ──────────────────────────────────────────
    e2e_results = {}
    for thr in iou_thresholds:
        tot = e2e_totals[thr]
        micro = _prf(tot["tp"], tot["fp"], tot["fn"])
        per_label_prf = {l: _prf(e2e_per_label[thr][l]["tp"], e2e_per_label[thr][l]["fp"], e2e_per_label[thr][l]["fn"])
                         for l in all_labels}
        e2e_macro = _macro_prf(e2e_per_label[thr], all_labels)
        e2e_results[thr] = {"micro": micro, "per_label": per_label_prf, **e2e_macro}

    avg_e2e_f1 = float(np.mean([
        _prf(e2e_totals[thr]["tp"], e2e_totals[thr]["fp"], e2e_totals[thr]["fn"])["f1"]
        for thr in iou_thresholds
    ])) if iou_thresholds else 0.0

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

    # ── Derived scalars ───────────────────────────────────────────────────────
    _char_f1           = float(np.mean(char_f1_scores))          if char_f1_scores           else 0.0
    _edit_dist         = float(np.mean(edit_dist_scores))         if edit_dist_scores         else 0.0
    _exact_rate        = exact_matches / len(char_f1_scores)      if char_f1_scores           else 0.0
    _mean_iou          = float(np.mean(iou_scores))               if iou_scores               else 0.0
    _uncond_iou        = float(np.mean(unconditional_iou_scores)) if unconditional_iou_scores else 0.0
    _spatial_char_f1   = float(np.mean(spatial_char_f1_scores))   if spatial_char_f1_scores   else 0.0
    _spatial_exact     = spatial_exact_matches / len(spatial_char_f1_scores) if spatial_char_f1_scores else 0.0

    return {
        # ── Summary (flat, first thing you see) ───────────────────────────────
        "summary": {
            "detection_micro_f1":     det_micro["f1"],
            "detection_macro_f1":     det_macro["macro_f1"],
            "avg_e2e_f1":             avg_e2e_f1,
            "unconditional_mean_iou": _uncond_iou,
            "mean_iou":               _mean_iou,
            "char_f1":                _char_f1,
            "exact_match_rate":       _exact_rate,
            "spatial_char_f1":        _spatial_char_f1,
            "spatial_exact_match_rate": _spatial_exact,
            "hallucination_rate":     hall_overall,
            "miss_rate":              miss_overall,
            "format_compliance":      format_compliance,
        },

        # ── Text extraction ───────────────────────────────────────────────────
        "text_extraction": {
            "detection": {
                "micro":    det_micro,
                "per_label": det_per_label_prf,
                **det_macro,
            },
            "span_exact": {
                "micro":    ex_micro,
                "per_label": ex_per_label_prf,
                **ex_macro,
            },
            "char_f1":         _char_f1,
            "edit_distance":   _edit_dist,
            "exact_match_rate": _exact_rate,
        },

        # ── Bbox localisation ─────────────────────────────────────────────────
        "bbox_localization": {
            "avg_e2e_f1":             avg_e2e_f1,
            "mean_iou":               _mean_iou,
            "unconditional_mean_iou": _uncond_iou,
            "spatial_char_f1":        _spatial_char_f1,
            "spatial_exact_match_rate": _spatial_exact,
            "end_to_end": {
                f"@{thr}": {
                    "micro":    e2e_results[thr]["micro"],
                    "per_label": e2e_results[thr]["per_label"],
                    **{k: v for k, v in e2e_results[thr].items() if k not in ("micro", "per_label")},
                }
                for thr in iou_thresholds
            },
        },

        # ── Diagnostics ───────────────────────────────────────────────────────
        "hallucination_rate": {"overall": hall_overall, "per_label": hall_per_label},
        "miss_rate":          {"overall": miss_overall, "per_label": miss_per_label},
        "coarse_f1":          coarse_prf,
        "label_confusion":    {k: dict(v) for k, v in label_confusion.items()},
        "format_compliance":  format_compliance,
    }
