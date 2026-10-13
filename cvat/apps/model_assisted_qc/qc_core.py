"""Model-QC v2 scoring core, synchronized with P-004's PD pilot."""
import math

DEFAULT_CFG = {
    "overlap_iou": 0.20, "match_iou": 0.30, "flip_iou": 0.30,
    "flip_min_conf": 0.25, "drop_max_iou": 0.30, "drift_iou": 0.65,
    "small_px": 32, "loc_min_conf": 0.30, "drop_nms_iou": 0.70,
    "ignore_ioa": 0.50,
}


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ioa(a, region):
    iw = min(a[2], region[2]) - max(a[0], region[0])
    ih = min(a[3], region[3]) - max(a[1], region[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return iw * ih / area if iw > 0 and ih > 0 and area > 0 else 0.0


def _nms(predictions, threshold):
    kept = []
    for prediction in sorted(predictions, key=lambda item: -item["confidence"]):
        if all(iou(prediction["bbox"], other["bbox"]) < threshold for other in kept):
            kept.append(prediction)
    return kept


def score_frame(humans, predictions, *, condition_metadata=None, config=None):
    """Return human/model suspicion units using P: PD + shift, without GT."""
    cfg = {**DEFAULT_CFG, **(config or {})}
    condition_metadata = condition_metadata or {}
    low_visibility = condition_metadata.get("lighting") == "night" or condition_metadata.get("visibility") == "low"
    spawn_scale = 0.5 if low_visibility else 1.0

    normalized_humans = [{
        **item, "class": item.get("class", item.get("label")),
    } for item in humans]
    normalized_predictions = [{
        **item, "class": item.get("class", item.get("label")),
        "conf": item.get("conf", item.get("confidence", 0.0)),
    } for item in predictions]
    ignore = [item for item in normalized_humans if item.get("ignore")]
    humans = [item for item in normalized_humans if not item.get("ignore")]
    predictions = [item for item in normalized_predictions if all(
        ioa(item["bbox"], region["bbox"]) < cfg["ignore_ioa"] for region in ignore
    )]

    units = []
    for human in humans:
        overlaps = [(prediction, iou(human["bbox"], prediction["bbox"])) for prediction in predictions]
        same_candidates = [(prediction, overlap) for prediction, overlap in overlaps
                           if prediction["class"] == human["class"] and overlap >= cfg["match_iou"]]
        other_candidates = [(prediction, overlap) for prediction, overlap in overlaps
                            if prediction["class"] != human["class"] and overlap >= cfg["flip_iou"]]
        same, same_iou = max(same_candidates, key=lambda item: item[0]["conf"], default=(None, None))
        other, other_iou = max(other_candidates, key=lambda item: item[0]["conf"], default=(None, None))
        s_same = same["conf"] if same else 0.0
        s_other = other["conf"] if other else 0.0
        pd_score = (1.0 + s_other - s_same) / 2.0
        if s_other > s_same and s_other >= cfg["flip_min_conf"]:
            risk_type = "flip"
        elif s_same == 0.0:
            risk_type = "spawn"
            pd_score *= spawn_scale
        elif same_iou is not None and same_iou < cfg["drift_iou"]:
            risk_type = "shift"
        else:
            risk_type = "ok"
        width = human["bbox"][2] - human["bbox"][0]
        height = human["bbox"][3] - human["bbox"][1]
        shift_score = 1.0 - same_iou if same and same["conf"] >= cfg["loc_min_conf"] and min(width, height) >= cfg["small_px"] else 0.0
        evidence = other if risk_type == "flip" else same
        evidence_iou = other_iou if risk_type == "flip" else same_iou
        units.append({
            "type": risk_type, "score": max(pd_score, shift_score), "iou": evidence_iou,
            "human": human, "prediction": evidence, "s_same": s_same, "s_other": s_other,
            "score_pd": pd_score, "score_shift": shift_score,
        })

    drop_candidates = []
    for prediction in predictions:
        max_overlap = max((iou(prediction["bbox"], human["bbox"]) for human in humans), default=0.0)
        if max_overlap < cfg["drop_max_iou"]:
            drop_candidates.append(prediction)
    for prediction in _nms(drop_candidates, cfg["drop_nms_iou"]):
        units.append({"type": "drop", "score": prediction["conf"], "iou": None,
                      "human": None, "prediction": prediction})
    return units


def frame_score(scores, min_score=0.5):
    return float(sum(score for score in scores if score >= min_score))


def rank_frames(frame_scores, budget=0.15):
    order = sorted(frame_scores, key=lambda frame: (-frame_scores[frame], frame))
    count = max(1, math.ceil(budget * len(order))) if order else 0
    return order, set(order[:count])


def top_risk_count(total, share=0.15):
    """Compatibility helper; UI selection now uses rank_frames instead."""
    return max(1, math.ceil(total * share)) if total else 0
