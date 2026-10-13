from django.db import transaction
from django.db.models import Count
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from cvat.apps.engine.models import Job
from cvat.apps.engine.permissions import JobPermission

from .models import QCRisk, RefereePrediction
from .qc_core import rank_frames
from .serializers import (
    FrameEvaluationSerializer, PredictionImportSerializer, RiskSerializer, RiskStatusSerializer,
)
from .services import ensure_referee_predictions, evaluate_frame


def _get_job(request, job_id):
    job = get_object_or_404(Job.objects.select_related("segment__task"), pk=job_id)
    if not JobPermission.create_scope_view(request, job).check_access().allow:
        raise NotFound()
    return job


class PredictionImportView(APIView):
    permission_classes = [IsAuthenticated]

    @transaction.atomic
    def post(self, request, job_id):
        job = _get_job(request, job_id)
        serializer = PredictionImportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        if data["replace"]:
            RefereePrediction.objects.filter(job=job).delete()
            QCRisk.objects.filter(job=job).delete()

        predictions = []
        for index, item in enumerate(data["predictions"]):
            predictions.append(RefereePrediction(
                job=job,
                frame=item["frame"],
                external_id=item.get("id") or f"{item['frame']}-{index}",
                label=item["label"],
                confidence=item["confidence"],
                bbox=item["bbox"],
            ))
        RefereePrediction.objects.bulk_create(predictions)
        return Response({"imported": len(predictions)}, status=status.HTTP_201_CREATED)


class FrameEvaluationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, job_id, frame):
        job = _get_job(request, job_id)
        serializer = FrameEvaluationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        prediction_count = ensure_referee_predictions(job)
        if not prediction_count:
            return Response(
                {"detail": "No matching referee predictions are available for this job."},
                status=status.HTTP_409_CONFLICT,
            )
        created, frame_prediction_count = evaluate_frame(job, frame, **serializer.validated_data)
        return Response({
            "created": RiskSerializer(created, many=True).data,
            "prediction_count": frame_prediction_count,
            "referee_ready": prediction_count,
        })


class RiskQueueView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        job = _get_job(request, job_id)
        try:
            share = float(request.query_params.get("share", 0.15))
        except ValueError:
            share = 0.15
        share = min(max(share, 0.01), 1.0)
        pending = QCRisk.objects.filter(job=job, status=QCRisk.Status.PENDING)
        total = pending.count()
        frame_scores = {frame: 0.0 for frame in job.segment.frame_set}
        for risk in pending.only("frame", "score"):
            if risk.score >= 0.5:
                frame_scores[risk.frame] = frame_scores.get(risk.frame, 0.0) + risk.score
        frame_order, top_frames = rank_frames(frame_scores, budget=share)
        frame_rank = {frame: index for index, frame in enumerate(frame_order)}
        visible = list(pending.filter(frame__in=top_frames))
        visible.sort(key=lambda risk: (frame_rank[risk.frame], -risk.score, risk.id))
        visible_count = len(visible)
        reviewed = QCRisk.objects.filter(job=job).exclude(status=QCRisk.Status.PENDING).order_by("-updated_date")[:50]
        return Response({
            "share": share,
            "pending_total": total,
            "visible_count": visible_count,
            "selected_frame_count": len(top_frames),
            "total_frame_count": len(frame_scores),
            "risks": RiskSerializer(visible, many=True).data,
            "reviewed": RiskSerializer(reviewed, many=True).data,
        })


class RiskStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, job_id, risk_id):
        job = _get_job(request, job_id)
        risk = get_object_or_404(QCRisk, job=job, pk=risk_id)
        serializer = RiskStatusSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        risk.status = serializer.validated_data["status"]
        risk.save(update_fields=("status", "updated_date"))
        return Response(RiskSerializer(risk).data)


class DashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        visible_jobs = JobPermission.create_scope_list(request).filter(Job.objects.all())
        risks = QCRisk.objects.filter(job__in=visible_jobs)
        status_counts = {
            item["status"]: item["count"]
            for item in risks.values("status").annotate(count=Count("id"))
        }
        type_counts = {
            item["risk_type"]: item["count"]
            for item in risks.values("risk_type").annotate(count=Count("id"))
        }
        rows = risks.select_related("job__segment__task").order_by("-updated_date")[:1000]
        return Response({
            "total": risks.count(),
            "status_counts": status_counts,
            "type_counts": type_counts,
            "risks": [
                {
                    **RiskSerializer(risk).data,
                    "job_id": risk.job_id,
                    "task_id": risk.job.segment.task_id,
                    "task_name": risk.job.segment.task.name,
                }
                for risk in rows
            ],
        })
