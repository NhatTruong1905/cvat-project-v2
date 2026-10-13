from django.apps import AppConfig


class ModelAssistedQCConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "cvat.apps.model_assisted_qc"
    verbose_name = "Model-assisted QA/QC"
