# Model-assisted QA/QC

The active MVP contains one workflow only:

1. YOLOv8n supplies visible prelabels.
2. An independent YOLO-BDD100K referee runs on images only.
3. Saved human boxes are scored with Probability Differential (PD).
4. DROP, FLIP, SHIFT and SPAWN risks contribute to frame scores when score >= 0.5.
5. The UI displays risks belonging to the top 15% ranked frames.

The referee does not receive ground truth, human annotations, or YOLOv8n outputs.

Active implementation:

- `qc_core.py`: PD scoring and frame ranking.
- `services.py`: prediction attachment and risk persistence.
- `views.py`: job evaluation, risk queue, decisions, and dashboard.
- `models.py`: `RefereePrediction` and `QCRisk` only.
- `benchmark/`: isolated YOLO-BDD100K baseline comparison against Grounding DINO and Florence-2.

Legacy phase-based Ground Truth/evaluation/benchmark code has been removed.
