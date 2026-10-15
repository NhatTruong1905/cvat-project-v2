#!/usr/bin/env python3
"""Run one referee model on BDD100K and emit a common prediction format."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALIASES = {
    "person": "pedestrian",
    "people": "pedestrian",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "motor": "motorcycle",
    "stop sign": "traffic sign",
    "street sign": "traffic sign",
}


def normalize_label(label, classes):
    label = str(label).lower().strip().strip(". ")
    label = ALIASES.get(label, label)
    return label if label in classes else None


def synchronize(torch, device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class YoloRunner:
    def __init__(self, cfg, device, classes):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        weights = hf_hub_download(cfg["model_id"], cfg.get("weights_file", "best.pt"))
        self.model = YOLO(weights)
        self.cfg, self.device, self.classes = cfg, device, classes

    def predict(self, image):
        result = self.model.predict(
            str(image), conf=self.cfg["confidence"], imgsz=self.cfg["image_size"],
            device=self.device, max_det=300, verbose=False,
        )[0]
        output = []
        for box in result.boxes:
            label = normalize_label(result.names[int(box.cls.item())], self.classes)
            if label:
                output.append((label, float(box.conf.item()), box.xyxy[0].tolist()))
        return output


class GroundingDinoRunner:
    def __init__(self, cfg, device, classes):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.device, self.cfg, self.classes = torch, device, cfg, classes
        self.processor = AutoProcessor.from_pretrained(cfg["model_id"])
        # Grounding DINO's encoder creates some intermediate tensors in FP32.
        # Loading all weights as FP16 can therefore fail in Linear layers on
        # CUDA (mat1 Float vs mat2 Half). Keep this model in FP32 for stability.
        dtype = torch.float32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg["model_id"], torch_dtype=dtype
        ).to(device).eval()
        self.prompt = ". ".join(classes) + "."

    def predict(self, image):
        from PIL import Image

        rgb = Image.open(image).convert("RGB")
        inputs = self.processor(images=rgb, text=self.prompt, return_tensors="pt").to(self.device)
        # BatchFeature.to(device) does not cast floating image tensors to the
        # model dtype. A FP16 model with FP32 pixels fails in the backbone conv.
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        postprocess_args = {
            "text_threshold": self.cfg["text_threshold"],
            "target_sizes": [rgb.size[::-1]],
        }
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=self.cfg["box_threshold"], **postprocess_args,
            )
        except TypeError:
            # Transformers 5 renamed box_threshold to threshold.
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=self.cfg["box_threshold"], **postprocess_args,
            )
        result = results[0]
        labels = result.get("text_labels", result.get("labels", []))
        found = []
        for label, score, box in zip(labels, result["scores"], result["boxes"]):
            label = normalize_label(label, self.classes)
            if label:
                found.append((label, float(score.item()), box.tolist()))
        return found


class FlorenceRunner:
    def __init__(self, cfg, device, classes):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.torch, self.device, self.cfg, self.classes = torch, device, cfg, classes
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg["model_id"])
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=dtype, attn_implementation="eager"
        ).to(device).eval()

    def predict(self, image):
        from PIL import Image

        rgb = Image.open(image).convert("RGB")
        task = self.cfg.get("task", "<OD>")
        inputs = self.processor(text=task, images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(
                **inputs, max_new_tokens=self.cfg.get("max_new_tokens", 1024),
                num_beams=3, do_sample=False,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=rgb.size)[task]
        score = float(self.cfg.get("default_score", 1.0))
        found = []
        for label, box in zip(parsed.get("labels", []), parsed.get("bboxes", [])):
            label = normalize_label(label, self.classes)
            if label:
                found.append((label, score, list(map(float, box))))
        return found


RUNNERS = {"yolo": YoloRunner, "grounding_dino": GroundingDinoRunner, "florence2": FlorenceRunner}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", required=True, choices=["yolo_bdd100k", "grounding_dino", "florence2"])
    parser.add_argument("--images", type=Path, default=here.parents[3] / "data/bdd100k_mvp_100_seed42/images")
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--config", type=Path, default=here / "config.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; pass --device cpu to run on CPU")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg, classes = config["models"][args.model], config["classes"]
    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    images = images[:args.limit] if args.limit else images
    if not images:
        raise SystemExit(f"No supported images found under {args.images}")

    runner = RUNNERS[cfg["kind"]](cfg, args.device, classes)
    for image in images[:min(args.warmup, len(images))]:
        runner.predict(image)
    synchronize(torch, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    predictions, latencies = [], []
    for frame, image in enumerate(images):
        synchronize(torch, args.device)
        started = time.perf_counter()
        detections = runner.predict(image)
        synchronize(torch, args.device)
        latencies.append(time.perf_counter() - started)
        for index, (label, confidence, bbox) in enumerate(detections):
            predictions.append({
                "frame": frame, "id": f"{args.model}-{frame}-{index}", "image": image.name,
                "label": label, "confidence": round(confidence, 6),
                "bbox": [round(float(value), 3) for value in bbox],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / f"{args.model}_predictions.json"
    prediction_path.write_text(json.dumps({"predictions": predictions}, indent=2), encoding="utf-8")
    timing = {
        "model": args.model, "role": cfg["role"], "model_id": cfg["model_id"],
        "device": args.device, "python": platform.python_version(), "torch": torch.__version__,
        "images": len(images), "detections": len(predictions),
        "total_seconds": round(sum(latencies), 4),
        "mean_ms": round(statistics.mean(latencies) * 1000, 2),
        "median_ms": round(statistics.median(latencies) * 1000, 2),
        "p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * .95) - 1)] * 1000, 2),
        "fps": round(len(images) / sum(latencies), 3),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(args.device) / 2**20, 1)
        if args.device.startswith("cuda") else None,
        "warmup_images": min(args.warmup, len(images)),
    }
    (args.output_dir / f"{args.model}_runtime.json").write_text(
        json.dumps(timing, indent=2), encoding="utf-8"
    )
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()

