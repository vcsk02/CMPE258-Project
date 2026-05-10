from __future__ import annotations

import numpy as np


def jitter(X: np.ndarray, sigma: float = 0.03, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0.0, sigma, size=X.shape)).astype(np.float32)


def scaling(X: np.ndarray, sigma: float = 0.10, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = rng.normal(1.0, sigma, size=(X.shape[0], 1, X.shape[2]))
    return (X * factors).astype(np.float32)


def time_mask(X: np.ndarray, max_width: int = 12, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_aug = X.copy()
    n, t, _ = X_aug.shape
    width = int(rng.integers(1, max_width + 1))
    starts = rng.integers(0, max(1, t - width), size=n)
    for i, start in enumerate(starts):
        X_aug[i, start : start + width, :] = 0.0
    return X_aug.astype(np.float32)


def augment_training_data(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple augmented copy of the training data."""
    X_jitter = jitter(X, seed=seed)
    X_scaled = scaling(X, seed=seed + 1)
    X_masked = time_mask(X, seed=seed + 2)
    X_out = np.concatenate([X, X_jitter, X_scaled, X_masked], axis=0)
    y_out = np.concatenate([y, y, y, y], axis=0)
    return X_out.astype(np.float32), y_out.astype(np.int64)
