from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REFERENCE_PROFILE_FILE = "reference_profile.json"


def compute_channel_profile(X: np.ndarray) -> dict[str, list[float]]:
    """Compute compact drift-monitoring profile for an array shaped (N, T, C)."""
    if X.ndim != 3:
        raise ValueError(f"Expected shape (N, T, C), got {X.shape}")
    flattened = X.reshape(-1, X.shape[-1])
    return {
        "mean": np.mean(flattened, axis=0).astype(float).tolist(),
        "std": np.std(flattened, axis=0).astype(float).tolist(),
        "p05": np.percentile(flattened, 5, axis=0).astype(float).tolist(),
        "p95": np.percentile(flattened, 95, axis=0).astype(float).tolist(),
    }


def save_reference_profile(X_train: np.ndarray, output_dir: str | Path) -> Path:
    """Save training-split signal statistics for production drift checks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / REFERENCE_PROFILE_FILE
    profile_path.write_text(json.dumps(compute_channel_profile(X_train), indent=2), encoding="utf-8")
    return profile_path


def load_reference_profile(path: str | Path) -> dict[str, list[float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def drift_score(window: np.ndarray, reference_profile: dict[str, list[float]]) -> dict[str, float | str]:
    """Return a simple, explainable drift score for one input window.

    The score measures average absolute channel mean shift in units of the
    reference standard deviation. It is intentionally lightweight for a class
    project demo and can be replaced with Evidently/WhyLabs/TFX validation in
    a cloud deployment.
    """
    if window.ndim != 2:
        raise ValueError(f"Expected one window shaped (T, C), got {window.shape}")
    ref_mean = np.asarray(reference_profile["mean"], dtype=float)
    ref_std = np.asarray(reference_profile["std"], dtype=float)
    channels = min(window.shape[-1], ref_mean.shape[0])
    ref_std = np.maximum(ref_std[:channels], 1e-8)
    score = float(np.mean(np.abs(window[:, :channels].mean(axis=0) - ref_mean[:channels]) / ref_std))
    if score >= 3.0:
        level = "high"
    elif score >= 1.5:
        level = "medium"
    else:
        level = "low"
    return {"drift_score": score, "drift_level": level}
