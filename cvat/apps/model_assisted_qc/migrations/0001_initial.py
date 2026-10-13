from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True
    dependencies = [("engine", "0108_profile_registration_survey_fields")]
    operations = [
        migrations.CreateModel(
            name="RefereePrediction",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("frame", models.PositiveIntegerField(db_index=True)),
                ("external_id", models.CharField(max_length=128)),
                ("label", models.CharField(max_length=128)),
                ("confidence", models.FloatField()),
                ("bbox", models.JSONField()),
                ("created_date", models.DateTimeField(auto_now_add=True)),
                ("job", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="qc_referee_predictions", to="engine.job")),
            ],
        ),
        migrations.CreateModel(
            name="QCRisk",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("frame", models.PositiveIntegerField(db_index=True)),
                ("fingerprint", models.CharField(max_length=255)),
                ("risk_type", models.CharField(choices=[("drop", "Drop"), ("flip", "Flip"), ("shift", "Shift"), ("spawn", "Spawn")], max_length=16)),
                ("status", models.CharField(choices=[("pending", "Pending"), ("in_review", "In review"), ("confirmed", "Confirmed"), ("rejected", "Rejected"), ("resolved", "Resolved")], default="pending", max_length=16)),
                ("score", models.FloatField(db_index=True)),
                ("iou", models.FloatField(blank=True, null=True)),
                ("human_object_id", models.CharField(blank=True, max_length=64, null=True)),
                ("human_label", models.CharField(blank=True, max_length=128, null=True)),
                ("referee_label", models.CharField(blank=True, max_length=128, null=True)),
                ("human_bbox", models.JSONField(blank=True, null=True)),
                ("referee_bbox", models.JSONField(blank=True, null=True)),
                ("updated_date", models.DateTimeField(auto_now=True)),
                ("created_date", models.DateTimeField(auto_now_add=True)),
                ("job", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="qc_risks", to="engine.job")),
                ("prediction", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="risks", to="model_assisted_qc.refereeprediction")),
            ],
            options={"ordering": ("-score", "frame", "id")},
        ),
        migrations.AddConstraint(model_name="refereeprediction", constraint=models.UniqueConstraint(fields=("job", "frame", "external_id"), name="unique_qc_referee_prediction")),
        migrations.AddConstraint(model_name="qcrisk", constraint=models.UniqueConstraint(fields=("job", "fingerprint"), name="unique_qc_risk")),
    ]
