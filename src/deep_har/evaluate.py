from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .constants import ACTIVITY_LABELS
from .data import load_dataset, select_sensor_subset
from .preprocess import ChannelStandardizer
from .visualization import plot_confusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepHAR checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--normalizer-path", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sensor-subset", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    model = tf.keras.models.load_model(args.model_path)
    normalizer = ChannelStandardizer.load(args.normalizer_path)
    data = load_dataset(args.data_dir)
    X_test = select_sensor_subset(data["X_test"], args.sensor_subset)
    X_test = normalizer.transform(X_test)
    y_test = data["y_test"]

    y_proba = model.predict(X_test, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    output_dir = Path(args.output_dir)
    result_dir = output_dir / "results"
    plot_dir = output_dir / "plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (result_dir / "evaluation_classification_report.json").write_text(
        json.dumps(classification_report(y_test, y_pred, target_names=ACTIVITY_LABELS, output_dict=True), indent=2),
        encoding="utf-8",
    )
    np.savetxt(result_dir / "evaluation_confusion_matrix.csv", confusion_matrix(y_test, y_pred), delimiter=",", fmt="%d")
    plot_confusion(y_test, y_pred, plot_dir / "evaluation_confusion_matrix.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
