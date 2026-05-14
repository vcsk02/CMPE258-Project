from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small reproducible hyperparameter sweeps.")
    parser.add_argument("--models", nargs="+", default=["cnn", "cnn_lstm", "sensor_transformer"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3, 5e-4])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/sweeps")
    parser.add_argument("--quick-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for model, lr, batch_size in itertools.product(args.models, args.learning_rates, args.batch_sizes):
        run_name = f"{model}_lr{lr:g}_bs{batch_size}".replace(".", "p")
        run_dir = output_dir / run_name
        cmd = [
            sys.executable,
            "-m",
            "deep_har.train",
            "--model",
            model,
            "--learning-rate",
            str(lr),
            "--batch-size",
            str(batch_size),
            "--epochs",
            str(args.epochs),
            "--output-dir",
            str(run_dir),
        ]
        if args.quick_run:
            cmd.append("--quick-run")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        metrics_path = run_dir / "results" / f"{model}_test_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update({"learning_rate": lr, "batch_size": batch_size, "run_dir": str(run_dir)})
        rows.append(metrics)

    if not rows:
        raise RuntimeError("No sweep runs were executed.")

    csv_path = output_dir / "sweep_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
