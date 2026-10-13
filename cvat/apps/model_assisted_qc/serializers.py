from rest_framework import serializers

from .models import QCRisk


class BBoxField(serializers.ListField):
    child = serializers.FloatField()
    min_length = 4
    max_length = 4

    def to_internal_value(self, data):
        value = super().to_internal_value(data)
        if value[2] <= value[0] or value[3] <= value[1]:
            raise serializers.ValidationError("bbox must be [x1, y1, x2, y2] with positive area")
        return value


class PredictionSerializer(serializers.Serializer):
    frame = serializers.IntegerField(min_value=0)
    id = serializers.CharField(required=False)
    label = serializers.CharField(max_length=128)
    confidence = serializers.FloatField(min_value=0, max_value=1)
    bbox = BBoxField()


class PredictionImportSerializer(serializers.Serializer):
    replace = serializers.BooleanField(default=True)
    predictions = PredictionSerializer(many=True)


class HumanAnnotationSerializer(serializers.Serializer):
    id = serializers.CharField(max_length=64)
    label = serializers.CharField(max_length=128)
    bbox = BBoxField()


class FrameEvaluationSerializer(serializers.Serializer):
    annotations = HumanAnnotationSerializer(many=True)
    include_spawn = serializers.BooleanField(default=False)
    condition_metadata = serializers.DictField(required=False, default=dict)


class RiskStatusSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=QCRisk.Status.choices)


class RiskSerializer(serializers.ModelSerializer):
    type = serializers.CharField(source="risk_type")

    class Meta:
        model = QCRisk
        fields = (
            "id", "frame", "type", "status", "score", "iou", "human_object_id",
            "human_label", "referee_label", "human_bbox", "referee_bbox", "updated_date",
        )

