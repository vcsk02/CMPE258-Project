from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .constants import ACTIVITY_LABELS


def window_features(X: np.ndarray) -> np.ndarray:
    """Convert windows shaped (N, T, C) into compact statistical features.

    This feature extractor is intentionally simple and dependency-free. It is used
    only for the packaged centroid demo checkpoint so a clean clone can run
    inference before a TensorFlow `.keras` checkpoint is trained locally.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected shape (N, T, C), got {X.shape}")
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    rms = np.sqrt(np.mean(np.square(X), axis=1))
    abs_mean = np.mean(np.abs(X), axis=1)
    span = X.max(axis=1) - X.min(axis=1)
    slope = X[:, -1, :] - X[:, 0, :]
    # Correlation-style features between accelerometer/gyro magnitude signals.
    mag_body_acc = np.linalg.norm(X[:, :, 0:3], axis=2)
    mag_body_gyro = np.linalg.norm(X[:, :, 3:6], axis=2)
    mag_total_acc = np.linalg.norm(X[:, :, 6:9], axis=2)
    mag_stats = np.stack([
        mag_body_acc.mean(axis=1), mag_body_acc.std(axis=1),
        mag_body_gyro.mean(axis=1), mag_body_gyro.std(axis=1),
        mag_total_acc.mean(axis=1), mag_total_acc.std(axis=1),
    ], axis=1)
    return np.concatenate([mean, std, rms, abs_mean, span, slope, mag_stats], axis=1).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)


@dataclass
class CentroidModel:
    centroids: np.ndarray
    labels: list[str]
    temperature: float = 1.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = window_features(X)
        # Negative squared Euclidean distance to each class centroid.
        dists = ((features[:, None, :] - self.centroids[None, :, :]) ** 2).mean(axis=2)
        logits = -dists / max(self.temperature, 1e-6)
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return idx, proba


def save_centroid_model(path: str | Path, centroids: np.ndarray, temperature: float = 1.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=centroids.astype(np.float32), labels=np.array(ACTIVITY_LABELS), temperature=np.array(float(temperature)))


def load_centroid_model(path: str | Path) -> CentroidModel:
    data = np.load(path, allow_pickle=True)
    labels = [str(x) for x in data['labels'].tolist()]
    return CentroidModel(centroids=data['centroids'], labels=labels, temperature=float(data['temperature']))
