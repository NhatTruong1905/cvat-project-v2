from django.urls import path

from .phase1_dataset_ground_truth.views import DatasetView, GroundTruthValidationView
from .phase2_yolo_evaluation.views import EvaluationView, PredictionView
from .phase3_condition_threshold.views import ConditionSummaryView, ThresholdAnalysisView
from .phase4_risk_review.views import AlertDetailView, AlertQueueView
from .phase5_human_review.views import AlertReviewView
from .phase6_benchmark.views import BenchmarkView
from .views import DashboardView, FrameEvaluationView, PredictionImportView, RiskQueueView, RiskStatusView

urlpatterns = [
    path("dashboard", DashboardView.as_view()),
    path("jobs/<int:job_id>/predictions", PredictionImportView.as_view()),
    path("jobs/<int:job_id>/frames/<int:frame>/evaluate", FrameEvaluationView.as_view()),
    path("jobs/<int:job_id>/risks", RiskQueueView.as_view()),
    path("jobs/<int:job_id>/risks/<int:risk_id>", RiskStatusView.as_view()),
    path("condition-summary", ConditionSummaryView.as_view()),
    path("tasks/<int:task_id>/dataset", DatasetView.as_view()),
    path("tasks/<int:task_id>/validate-ground-truth", GroundTruthValidationView.as_view()),
    path("tasks/<int:task_id>/runs", EvaluationView.as_view()),
    path("tasks/<int:task_id>/threshold-analyses", ThresholdAnalysisView.as_view()),
    path("runs/<int:run_id>/predictions", PredictionView.as_view()),
    path("tasks/<int:task_id>/alerts", AlertQueueView.as_view()),
    path("alerts/<int:alert_id>", AlertDetailView.as_view()),
    path("alerts/<int:alert_id>/review", AlertReviewView.as_view()),
    path("tasks/<int:task_id>/benchmarks", BenchmarkView.as_view()),
]
