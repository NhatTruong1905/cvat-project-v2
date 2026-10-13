from django.urls import path

from .views import DashboardView, FrameEvaluationView, PredictionImportView, RiskQueueView, RiskStatusView


urlpatterns = [
    path("dashboard", DashboardView.as_view()),
    path("jobs/<int:job_id>/predictions", PredictionImportView.as_view()),
    path("jobs/<int:job_id>/frames/<int:frame>/evaluate", FrameEvaluationView.as_view()),
    path("jobs/<int:job_id>/risks", RiskQueueView.as_view()),
    path("jobs/<int:job_id>/risks/<int:risk_id>", RiskStatusView.as_view()),
]
