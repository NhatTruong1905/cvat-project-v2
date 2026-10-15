#!/usr/bin/env python3
"""Evaluate common-format predictions against BDD100K ground truth with COCO metrics."""

import argparse
import json
from pathlib import Path


def to_coco_ground_truth(frames, classes):
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(classes)]
    category_ids = {item["name"]: item["id"] for item in categories}
    images, annotations = [], []
    annotation_id = 1
    for image_id, frame in enumerate(frames, 1):
        images.append({
            "id": image_id, "file_name": frame["filename"],
            "width": frame["width"], "height": frame["height"],
        })
        for box in frame["boxes"]:
            x1, y1, x2, y2 = box["bbox"]
            width, height = max(0, x2 - x1), max(0, y2 - y1)
            annotations.append({
                "id": annotation_id, "image_id": image_id,
                "category_id": category_ids[box["class"]],
                "bbox": [x1, y1, width, height], "area": width * height, "iscrowd": 0,
            })
            annotation_id += 1
    return {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": categories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path, default=here.parents[3] / "data/bdd100k_mvp_100_seed42/bdd100k_ground_truth.json")
    parser.add_argument("--config", type=Path, default=here / "config.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    classes = json.loads(args.config.read_text(encoding="utf-8"))["classes"]
    frames = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    gt = to_coco_ground_truth(frames, classes)
    image_ids = {item["file_name"]: item["id"] for item in gt["images"]}
    category_ids = {item["name"]: item["id"] for item in gt["categories"]}
    detections = []
    for item in predictions:
        if item.get("image") not in image_ids or item.get("label") not in category_ids:
            continue
        x1, y1, x2, y2 = item["bbox"]
        detections.append({
            "image_id": image_ids[item["image"]], "category_id": category_ids[item["label"]],
            "bbox": [x1, y1, max(0, x2 - x1), max(0, y2 - y1)],
            "score": float(item["confidence"]),
        })

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": gt["images"], "categories": gt["categories"], "annotations": []
        }
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["mAP_50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    metrics = {name: round(float(value), 6) for name, value in zip(names, evaluator.stats)}
    metrics.update({"ground_truth_boxes": len(gt["annotations"]), "predicted_boxes": len(detections)})
    output = args.output or args.predictions.with_name(args.predictions.stem.replace("_predictions", "") + "_metrics.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

