from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChannelStandardizer:
    """Per-channel z-score normalization fitted only on training data."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        if X.ndim != 3:
            raise ValueError(f"Expected shape (N, T, C), got {X.shape}")
        self.mean_ = X.mean(axis=(0, 1), keepdims=True)
        self.std_ = X.std(axis=(0, 1), keepdims=True) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ChannelStandardizer must be fitted before transform.")
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot save an unfitted normalizer.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, std=self.std_, eps=np.array(self.eps))

    @classmethod
    def load(cls, path: str | Path) -> "ChannelStandardizer":
        data = np.load(path)
        return cls(mean_=data["mean"], std_=data["std"], eps=float(data["eps"]))
