#!/usr/bin/env python3
"""Run independent prelabel and referee models and create CVAT/QC artifacts."""

import argparse
import json
import time
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BDD_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "traffic light", "traffic sign",
]
COCO_TO_BDD = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
}
BDD_ALIASES = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}


def normalized_box(xyxy, width, height):
    x1, y1, x2, y2 = xyxy
    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def write_prelabel_archive(output, images, results):
    root = output / "prelabels_yolov8n"
    # Datumaro's Ultralytics importer used by CVAT requires the canonical
    # images/<subset> <-> labels/<subset> layout, even for annotation-only ZIPs.
    labels = root / "labels" / "train"
    labels.mkdir(parents=True, exist_ok=True)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for image, result in zip(images, results):
        height, width = result.orig_shape
        lines = []
        for box in result.boxes:
            raw_name = result.names[int(box.cls.item())]
            label = COCO_TO_BDD.get(raw_name)
            if label is None:
                continue
            class_id = BDD_CLASSES.index(label)
            values = normalized_box(box.xyxy[0].tolist(), width, height)
            lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
            counts[label] += 1
        (labels / f"{image.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(BDD_CLASSES))
    (root / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nnames:\n{names}\n", encoding="utf-8"
    )
    archive = output / "prelabels_yolov8n_cvat.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        # Keep the empty image subset directory in an annotation-only archive.
        # The CVAT/Datumaro importer checks that this directory exists.
        bundle.writestr("images/train/", "")
        for path in root.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(root))
    return archive, counts


def write_referee_json(output, images, results):
    predictions = []
    counts = Counter()
    for frame, (image, result) in enumerate(zip(images, results)):
        for box_index, box in enumerate(result.boxes):
            raw_name = result.names[int(box.cls.item())]
            label = BDD_ALIASES.get(raw_name, raw_name)
            if label not in BDD_CLASSES:
                continue
            predictions.append({
                "frame": frame,
                "id": f"bdd-referee-{frame}-{box_index}",
                "label": label,
                "confidence": round(float(box.conf.item()), 6),
                "bbox": [round(float(value), 3) for value in box.xyxy[0].tolist()],
                "image": image.name,
            })
            counts[label] += 1
    path = output / "referee_predictions.json"
    path.write_text(json.dumps({"predictions": predictions}, indent=2), encoding="utf-8")
    return path, counts


def run_model(model, images, confidence, image_size):
    started = time.perf_counter()
    results = list(model.predict(
        source=[str(image) for image in images], conf=confidence, imgsz=image_size,
        max_det=300, stream=True, verbose=False,
    ))
    return results, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prelabel-weights", default="yolov8n.pt")
    parser.add_argument("--referee-repo", default="dronefreak/bdd100k-yolov8s")
    parser.add_argument("--referee-file", default="best.pt")
    parser.add_argument("--prelabel-confidence", type=float, default=0.25)
    parser.add_argument("--referee-confidence", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (args.output / "referee_predictions.json").is_file():
        print("Referee predictions already exist; skipping model inference.", flush=True)
        return

    temporary_directory = None
    image_root = args.images
    if args.images.is_file() and args.images.suffix.lower() == ".zip":
        temporary_directory = tempfile.TemporaryDirectory(prefix="model-qc-images-")
        with zipfile.ZipFile(args.images) as archive:
            archive.extractall(temporary_directory.name)
        image_root = Path(temporary_directory.name)

    images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    print(f"[1/4] Loading YOLOv8n prelabel model: {args.prelabel_weights}", flush=True)
    prelabel_model = YOLO(args.prelabel_weights)
    print(f"[2/4] Running prelabel model on {len(images)} images", flush=True)
    prelabel_results, prelabel_seconds = run_model(
        prelabel_model, images, args.prelabel_confidence, args.image_size
    )
    prelabel_archive, prelabel_counts = write_prelabel_archive(
        args.output, images, prelabel_results
    )

    print(f"[3/4] Downloading independent referee: {args.referee_repo}/{args.referee_file}", flush=True)
    referee_weights = hf_hub_download(
        repo_id=args.referee_repo, filename=args.referee_file, cache_dir="/cache/huggingface"
    )
    referee_model = YOLO(referee_weights)
    print(f"[4/4] Running referee on images only", flush=True)
    referee_results, referee_seconds = run_model(
        referee_model, images, args.referee_confidence, args.image_size
    )
    referee_path, referee_counts = write_referee_json(args.output, images, referee_results)

    report = {
        "images": len(images),
        "image_order": [image.name for image in images],
        "prelabel": {
            "model": args.prelabel_weights,
            "seconds": round(prelabel_seconds, 3),
            "seconds_per_image": round(prelabel_seconds / len(images), 4),
            "boxes": sum(prelabel_counts.values()),
            "classes": dict(prelabel_counts),
            "artifact": prelabel_archive.name,
        },
        "referee": {
            "model": f"{args.referee_repo}/{args.referee_file}",
            "ground_truth_input": False,
            "seconds": round(referee_seconds, 3),
            "seconds_per_image": round(referee_seconds / len(images), 4),
            "predictions": sum(referee_counts.values()),
            "classes": dict(referee_counts),
            "artifact": referee_path.name,
        },
        "configuration": {
            "prelabel_confidence": args.prelabel_confidence,
            "referee_confidence": args.referee_confidence,
            "image_size": args.image_size,
        },
    }
    (args.output / "pipeline_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
