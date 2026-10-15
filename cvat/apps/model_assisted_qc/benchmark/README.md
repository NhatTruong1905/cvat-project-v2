# Model-assisted QC detector benchmark

This benchmark compares three image-only referee detectors on the same BDD100K subset:

- **Baseline:** YOLO-BDD100K (`dronefreak/bdd100k-yolov8s`).
- **Candidate:** Grounding DINO tiny, prompted with the ten BDD100K classes.
- **Candidate:** Florence-2 base fine-tuned, using its `<OD>` task and the
  Transformers-native `florence-community/Florence-2-base-ft` conversion.

The default dataset is `data/bdd100k_mvp_100_seed42`: 100 images and the matching
`bdd100k_ground_truth.json`. Model loading is excluded from timing. Each model receives the
same ordered images and produces the existing referee JSON schema (`frame`, `image`, `label`,
`confidence`, `bbox`).

## Metrics

Accuracy uses standard COCO bounding-box metrics: mAP@[.50:.95], AP50, AP75, AP by object size,
and average recall. Runtime includes preprocessing, inference and postprocessing and reports mean,
median, p95, FPS, and peak allocated CUDA memory. Run models one at a time on an otherwise idle GPU.

Florence-2 `<OD>` does not expose calibrated per-box confidence. Its detections therefore receive
score `1.0`; its AP ranking and suitability for the current Probability Differential QC algorithm
must be interpreted with this limitation. Grounding DINO and YOLO provide native confidence scores.

## Setup and run

From the repository root, create a dedicated environment and install dependencies:

```powershell
python -m venv .venv-benchmark
.\.venv-benchmark\Scripts\Activate.ps1
python -m pip install -r cvat/apps/model_assisted_qc/benchmark/requirements.txt
```

Run each model in a fresh process so peak memory measurements are isolated:

```powershell
$bench = "cvat/apps/model_assisted_qc/benchmark"
python "$bench/run_benchmark.py" --model yolo_bdd100k
python "$bench/run_benchmark.py" --model grounding_dino
python "$bench/run_benchmark.py" --model florence2

python "$bench/evaluate.py" --predictions "$bench/results/yolo_bdd100k_predictions.json"
python "$bench/evaluate.py" --predictions "$bench/results/grounding_dino_predictions.json"
python "$bench/evaluate.py" --predictions "$bench/results/florence2_predictions.json"
python "$bench/summarize.py"
```

Use `--device cpu` when CUDA is unavailable, and `--limit 5` for a smoke test. Do not compare a
limited run with the full ground truth; use it only to verify model execution. Final outputs are
`results/leaderboard.{json,md}` plus predictions, metrics, and runtime metadata per model.

Model IDs, thresholds and the fixed class vocabulary are version-controlled in `config.json`.
For a selection decision, prioritize mAP and AR first, then require acceptable latency/VRAM and
confidence output compatible with PD scoring. Threshold tuning, if performed, must use a separate
validation split rather than these 100 benchmark images.

## Google Colab

Open `benchmark_colab.ipynb` in Google Colab, select a T4 GPU, and run every cell in order. The
notebook downloads the image ZIP and all three model weights at runtime. It asks for one local file,
`data/bdd100k_mvp_100_seed42/bdd100k_ground_truth.json`, then downloads
`model_qc_benchmark_results.zip` automatically after evaluation.
