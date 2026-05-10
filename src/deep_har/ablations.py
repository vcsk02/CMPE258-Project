from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model and sensor ablation experiments.")
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn", "cnn_lstm", "tcn", "attention"])
    parser.add_argument("--sensor-subsets", nargs="+", default=["all", "body_acc", "body_gyro", "total_acc"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--quick-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in args.models:
        for subset in args.sensor_subsets:
            run_dir = output_dir / f"{model}_{subset}"
            cmd = [
                sys.executable,
                "-m",
                "deep_har.train",
                "--model",
                model,
                "--sensor-subset",
                subset,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--output-dir",
                str(run_dir),
            ]
            if args.quick_run:
                cmd.append("--quick-run")
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            metrics_path = run_dir / "results" / f"{model}_test_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(metrics)

    csv_path = output_dir / "ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
