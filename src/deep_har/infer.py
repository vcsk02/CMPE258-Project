from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constants import ACTIVITY_LABELS
from .data import load_example_window
from .preprocess import ChannelStandardizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single HAR sensor window.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--normalizer-path", required=True)
    parser.add_argument("--input", required=True, help="CSV or NPY file with shape 128 x channels")
    return parser.parse_args()


def predict(model_path: str | Path, normalizer_path: str | Path, input_path: str | Path) -> dict:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    normalizer = ChannelStandardizer.load(normalizer_path)
    X = load_example_window(input_path)
    X = normalizer.transform(X)
    proba = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(proba))
    return {
        "predicted_class": ACTIVITY_LABELS[idx],
        "predicted_index": idx,
        "confidence": float(proba[idx]),
        "probabilities": {label: float(p) for label, p in zip(ACTIVITY_LABELS, proba)},
    }


def main() -> None:
    args = parse_args()
    result = predict(args.model_path, args.normalizer_path, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
