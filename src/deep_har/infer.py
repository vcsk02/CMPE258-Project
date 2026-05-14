from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constants import ACTIVITY_LABELS
from .data import load_example_window
from .numpy_baseline import load_centroid_model
from .prompting import build_explanation_prompt
from .preprocess import ChannelStandardizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single HAR sensor window.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--normalizer-path", required=True)
    parser.add_argument("--input", required=True, help="CSV or NPY file with shape 128 x channels")
    parser.add_argument("--explain", action="store_true", help="Include deterministic LLM-prompt explanation artifact.")
    return parser.parse_args()


def _predict_proba(model_path: Path, X: np.ndarray) -> np.ndarray:
    """Predict with either a real TensorFlow checkpoint or packaged NumPy demo model."""
    if model_path.suffix.lower() == ".npz":
        model = load_centroid_model(model_path)
        return model.predict_proba(X)

    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for .keras checkpoints. Use the packaged "
            "outputs/checkpoints/demo_centroid_model.npz for lightweight demo inference, "
            "or install TensorFlow and train a .keras checkpoint."
        ) from exc

    model = tf.keras.models.load_model(model_path)
    return model.predict(X, verbose=0)


def predict(model_path: str | Path, normalizer_path: str | Path, input_path: str | Path, explain: bool = False) -> dict:
    model_path = Path(model_path)
    normalizer = ChannelStandardizer.load(normalizer_path)
    raw_window = load_example_window(input_path)
    X = normalizer.transform(raw_window)
    proba = _predict_proba(model_path, X)[0]
    idx = int(np.argmax(proba))
    result = {
        "predicted_class": ACTIVITY_LABELS[idx],
        "predicted_index": idx,
        "confidence": float(proba[idx]),
        "probabilities": {label: float(p) for label, p in zip(ACTIVITY_LABELS, proba)},
        "model_path": str(model_path),
    }
    if explain:
        prompt = build_explanation_prompt(
            result["predicted_class"],
            result["confidence"],
            result["probabilities"],
            raw_window[0],
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
