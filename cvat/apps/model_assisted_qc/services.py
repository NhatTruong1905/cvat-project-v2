import hashlib
import json
import os
from pathlib import Path

from django.db import transaction

from .models import QCRisk, RefereePrediction
from .qc_core import score_frame

MIN_RISK_SCORE = 0.5
DEFAULT_PREDICTIONS_PATH = "/opt/cvat/model_qc/referee_predictions.json"


@transaction.atomic
def ensure_referee_predictions(job):
    """Lazily attach the bundled referee output to a job by image filename."""
    existing = RefereePrediction.objects.filter(job=job).count()
    if existing:
        return existing

    source = Path(os.getenv("MODEL_ASSISTED_QC_PREDICTIONS", DEFAULT_PREDICTIONS_PATH))
    if not source.is_file():
        return 0

    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_predictions = payload if isinstance(payload, list) else payload.get("predictions", [])
    if not isinstance(raw_predictions, list):
        return 0

    frame_set = set(job.segment.frame_set)
    image_frames = {
        Path(image.path).name: image.frame
        for image in job.segment.task.data.images.filter(frame__in=frame_set)
    }
    predictions = []
    for index, item in enumerate(raw_predictions):
        image_name = Path(str(item.get("image", ""))).name
        frame = image_frames.get(image_name)
        if frame is None:
            continue
        predictions.append(RefereePrediction(
            job=job,
            frame=frame,
            external_id=str(item.get("id") or f"bundled-{frame}-{index}"),
            label=item["label"],
            confidence=float(item["confidence"]),
            bbox=item["bbox"],
        ))

    if predictions:
        RefereePrediction.objects.bulk_create(predictions, ignore_conflicts=True)
    return RefereePrediction.objects.filter(job=job).count()


def _fingerprint(frame, risk):
    human = risk["human"]
    prediction = risk["prediction"]
    identity = human["id"] if human else prediction["external_id"]
    raw = f"{frame}:{risk['type']}:{identity}"
    return hashlib.sha256(raw.encode()).hexdigest()


@transaction.atomic
def evaluate_frame(job, frame, annotations, *, include_spawn=False, condition_metadata=None):
    prediction_models = list(RefereePrediction.objects.filter(job=job, frame=frame))
    predictions = [
        {
            "external_id": prediction.external_id,
            "label": prediction.label,
            "confidence": prediction.confidence,
            "bbox": prediction.bbox,
            "model": prediction,
        }
        for prediction in prediction_models
    ]
    humans = [{"id": item["id"], "label": item["label"], "bbox": item["bbox"]} for item in annotations]
    units = score_frame(humans, predictions, condition_metadata=condition_metadata)

    QCRisk.objects.filter(
        job=job, frame=frame, status__in=(QCRisk.Status.PENDING, QCRisk.Status.IN_REVIEW)
    ).delete()
    created = []
    for unit in units:
        if unit["score"] < MIN_RISK_SCORE:
            continue
        if unit["type"] == "ok":
            continue
        human = unit["human"]
        prediction = unit["prediction"]
        risk, was_created = QCRisk.objects.get_or_create(
            job=job,
            fingerprint=_fingerprint(frame, unit),
            defaults={
                "frame": frame,
                "prediction": prediction["model"] if prediction else None,
                "risk_type": unit["type"],
                "score": unit["score"],
                "iou": unit["iou"],
                "human_object_id": human["id"] if human else None,
                "human_label": human["label"] if human else None,
                "referee_label": prediction["label"] if prediction else None,
                "human_bbox": human["bbox"] if human else None,
                "referee_bbox": prediction["bbox"] if prediction else None,
            },
        )
        if was_created:
            created.append(risk)
    return created, len(predictions)
