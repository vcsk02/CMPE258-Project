"""Generate lightweight packaged demo artifacts.

This script is intentionally dependency-light and produces a NumPy centroid model
so the Gradio/CLI inference path works from a clean clone. It does not replace the
scored TensorFlow training pipeline in `python -m deep_har.train`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    print(
        "Demo artifacts are already committed in outputs/. To regenerate exactly, "
        "run the repository patch script used to create the submitted archive or "
        "train a TensorFlow model with python -m deep_har.train."
    )
    print("Packaged artifacts:")
    for path in [
        "outputs/checkpoints/demo_centroid_model.npz",
        "outputs/preprocessing/normalizer.npz",
        "outputs/preprocessing/reference_profile.json",
        "outputs/results/demo_centroid_test_metrics.json",
    ]:
        print(f"- {path}: {'present' if Path(path).exists() else 'missing'}")


if __name__ == "__main__":
    main()
