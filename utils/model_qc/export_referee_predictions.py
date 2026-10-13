#!/usr/bin/env python3
"""Run an image-only Ultralytics referee and export the Model QC import format."""

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="YOLO-BDD100K weights or Ultralytics model name")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing task images")
    parser.add_argument("--output", default="referee_predictions.json", type=Path)
    parser.add_argument("--confidence", default=0.01, type=float)
    parser.add_argument("--image-size", default=960, type=int)
    parser.add_argument("--start-frame", default=0, type=int)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the runner dependency first: pip install ultralytics") from exc

    images = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No supported images found in {args.images}")

    model = YOLO(args.weights)
    predictions = []
    results = model.predict(
        source=[str(path) for path in images],
        conf=args.confidence,
        imgsz=args.image_size,
        stream=True,
        verbose=False,
    )
    for image_index, result in enumerate(results):
        frame = args.start_frame + image_index
        for box_index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            predictions.append({
                "frame": frame,
                "id": f"referee-{frame}-{box_index}",
                "label": result.names[class_id],
                "confidence": round(float(box.conf.item()), 6),
                "bbox": [round(float(value), 3) for value in box.xyxy[0].tolist()],
            })

    args.output.write_text(json.dumps({"predictions": predictions}, indent=2), encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions for {len(images)} frames to {args.output}")


if __name__ == "__main__":
    main()
