from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .constants import ACTIVITY_LABELS


def plot_training_curves(history: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("loss", [])) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history.get("loss", []), label="train_loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTIVITY_LABELS)
    disp.plot(ax=ax, values_format=".2f", xticks_rotation=45, colorbar=False)
    ax.set_title("Normalized confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_sensor_window(window: np.ndarray, title: str = "Sensor window"):
    fig, ax = plt.subplots(figsize=(10, 4))
    for channel in range(window.shape[-1]):
        ax.plot(window[:, channel], linewidth=1.0, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized value")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig
