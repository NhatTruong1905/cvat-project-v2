#!/usr/bin/env python3
"""Merge model metrics and runtime files into JSON and Markdown leaderboards."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--results", type=Path, default=here / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in sorted(args.results.glob("*_metrics.json")):
        name = metrics_path.stem.removesuffix("_metrics")
        runtime_path = args.results / f"{name}_runtime.json"
        if not runtime_path.exists():
            continue
        rows.append({"model": name, **json.loads(metrics_path.read_text()), **json.loads(runtime_path.read_text())})
    rows.sort(key=lambda row: row["mAP_50_95"], reverse=True)
    (args.results / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    header = "| Model | Role | mAP50-95 | AP50 | AP75 | Recall@100 | Mean ms | FPS | Peak VRAM MB |\n|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['role']} | {row['mAP_50_95']:.4f} | {row['AP50']:.4f} | "
            f"{row['AP75']:.4f} | {row['AR_100']:.4f} | {row['mean_ms']:.2f} | {row['fps']:.2f} | "
            f"{row['peak_gpu_memory_mb'] if row['peak_gpu_memory_mb'] is not None else 'n/a'} |"
        )
    (args.results / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

