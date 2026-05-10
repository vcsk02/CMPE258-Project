from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

import numpy as np

from .constants import DATASET_URL, SENSOR_GROUPS, SIGNAL_NAMES

Split = Literal["train", "test"]


def download_uci_har(data_dir: str | Path = "data", force: bool = False) -> Path:
    """Download and extract the UCI HAR dataset if it is not already present."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    extract_dir = raw_dir / "UCI HAR Dataset"
    zip_path = raw_dir / "UCI_HAR_Dataset.zip"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if extract_dir.exists() and not force:
        return extract_dir

    if force and extract_dir.exists():
        import shutil
        shutil.rmtree(extract_dir)

    print(f"Downloading UCI HAR dataset to {zip_path}...")
    urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    return extract_dir


def _load_signal_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing signal file: {path}")
    return np.loadtxt(path, dtype=np.float32)


def load_split(split: Split, data_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one UCI HAR split.

    Returns:
        X: shape (n_samples, 128, 9)
        y: integer labels in [0, 5]
        subject: subject IDs for each window
    """
    extract_dir = download_uci_har(data_dir)
    split_dir = extract_dir / split
    signal_dir = split_dir / "Inertial Signals"

    signals = []
    for name in SIGNAL_NAMES:
        file_name = f"{name}_{split}.txt"
        signals.append(_load_signal_file(signal_dir / file_name))

    X = np.stack(signals, axis=-1).astype(np.float32)
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    subject = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)
    return X, y, subject


def load_dataset(data_dir: str | Path = "data") -> dict[str, np.ndarray]:
    X_train, y_train, subject_train = load_split("train", data_dir)
    X_test, y_test, subject_test = load_split("test", data_dir)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "subject_train": subject_train,
        "X_test": X_test,
        "y_test": y_test,
        "subject_test": subject_test,
    }


def select_sensor_subset(X: np.ndarray, sensor_subset: str = "all") -> np.ndarray:
    if sensor_subset not in SENSOR_GROUPS:
        valid = ", ".join(SENSOR_GROUPS)
        raise ValueError(f"Unknown sensor subset '{sensor_subset}'. Valid values: {valid}")
    return X[:, :, SENSOR_GROUPS[sensor_subset]]


def load_example_window(path: str | Path) -> np.ndarray:
    """Load a single 128 x channels CSV or NPY window."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.loadtxt(path, delimiter=",", skiprows=1 if _has_header(path) else 0)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected array shape (128, C) or (N, 128, C), got {arr.shape}")
    return arr


def _has_header(path: Path) -> bool:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    try:
        [float(x) for x in first_line.split(",")]
        return False
    except ValueError:
        return True
