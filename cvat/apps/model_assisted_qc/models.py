from django.db import models

from cvat.apps.engine.models import Job


class RefereePrediction(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="qc_referee_predictions")
    frame = models.PositiveIntegerField(db_index=True)
    external_id = models.CharField(max_length=128)
    label = models.CharField(max_length=128)
    confidence = models.FloatField()
    bbox = models.JSONField()
    created_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(
            fields=("job", "frame", "external_id"), name="unique_qc_referee_prediction",
        )]


class QCRisk(models.Model):
    class Type(models.TextChoices):
        DROP = "drop", "Drop"
        FLIP = "flip", "Flip"
        SHIFT = "shift", "Shift"
        SPAWN = "spawn", "Spawn"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        IN_REVIEW = "in_review", "In review"
        CONFIRMED = "confirmed", "Confirmed"
        REJECTED = "rejected", "Rejected"
        RESOLVED = "resolved", "Resolved"

    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="qc_risks")
    prediction = models.ForeignKey(
        RefereePrediction, null=True, blank=True, on_delete=models.SET_NULL, related_name="risks",
    )
    frame = models.PositiveIntegerField(db_index=True)
    fingerprint = models.CharField(max_length=255)
    risk_type = models.CharField(max_length=16, choices=Type.choices)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    score = models.FloatField(db_index=True)
    iou = models.FloatField(null=True, blank=True)
    human_object_id = models.CharField(max_length=64, null=True, blank=True)
    human_label = models.CharField(max_length=128, null=True, blank=True)
    referee_label = models.CharField(max_length=128, null=True, blank=True)
    human_bbox = models.JSONField(null=True, blank=True)
    referee_bbox = models.JSONField(null=True, blank=True)
    updated_date = models.DateTimeField(auto_now=True)
    created_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-score", "frame", "id")
        constraints = [models.UniqueConstraint(
            fields=("job", "fingerprint"), name="unique_qc_risk",
        )]
