# Model-assisted QC MVP

This app implements an independent model-referee workflow at **risk level**:

1. YOLOv8n creates prelabels outside this app.
2. A separately trained YOLO-BDD100K referee runs on images only and exports predictions.
3. The annotator edits and saves a frame in CVAT.
4. The UI compares that saved frame snapshot with cached referee predictions.
5. CVAT keeps risks with score >= 0.5 and shows the top 15% of pending object risks,
   not the top 15% of frames.

The MVP enables `drop`, `flip`, and `shift`. `spawn` is disabled in the UI because referee
false negatives make it unreliable in low visibility and for small objects.

## Referee JSON

Use the **Model QC** button in the annotation top bar and import either an array or an object
with a `predictions` array:

```json
{
  "predictions": [
    {
      "frame": 0,
      "id": "yolo-bdd100k-0-0",
      "label": "car",
      "confidence": 0.91,
      "bbox": [100, 120, 260, 310]
    }
  ]
}
```

Coordinates must be absolute image coordinates in `[x1, y1, x2, y2]` format. The referee
export must be generated from images only. Do not pass CVAT annotations to the inference step.

An image-only exporter is included:

```bash
pip install ultralytics
python utils/model_qc/export_referee_predictions.py \
  --weights /path/to/yolo-bdd100k.pt \
  --images /path/to/the/task/images \
  --output referee_predictions.json
```

The exporter maps lexicographically sorted image files to consecutive CVAT frame numbers. Use
`--start-frame` if the job starts after frame zero, and verify that file order matches CVAT.

## Start

After rebuilding the server and UI, apply the migration:

```bash
python manage.py migrate model_assisted_qc
```

Open an annotation job, import referee JSON once, then save or manually evaluate a frame.
The queue refreshes to the top 15% of currently pending risks.
