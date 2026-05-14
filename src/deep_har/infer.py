from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constants import ACTIVITY_LABELS
from .data import load_example_window
from .prompting import build_explanation_prompt
from .preprocess import ChannelStandardizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single HAR sensor window.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--normalizer-path", required=True)
    parser.add_argument("--input", required=True, help="CSV or NPY file with shape 128 x channels")
    parser.add_argument("--explain", action="store_true", help="Include deterministic LLM-prompt explanation artifact.")
    return parser.parse_args()


def predict(model_path: str | Path, normalizer_path: str | Path, input_path: str | Path, explain: bool = False) -> dict:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    normalizer = ChannelStandardizer.load(normalizer_path)
    X = load_example_window(input_path)
    X = normalizer.transform(X)
    proba = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(proba))
    result = {
        "predicted_class": ACTIVITY_LABELS[idx],
        "predicted_index": idx,
        "confidence": float(proba[idx]),
        "probabilities": {label: float(p) for label, p in zip(ACTIVITY_LABELS, proba)},
    }
    if explain:
        raw_window = load_example_window(input_path)[0]
        prompt = build_explanation_prompt(
            result["predicted_class"],
            result["confidence"],
            result["probabilities"],
            raw_window,
        )
        result["explanation"] = prompt.deterministic_explanation
        result["llm_system_prompt"] = prompt.system_prompt
        result["llm_user_prompt"] = prompt.user_prompt
    return result


def main() -> None:
    args = parse_args()
    result = predict(args.model_path, args.normalizer_path, args.input, explain=args.explain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
