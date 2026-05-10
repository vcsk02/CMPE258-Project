from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deep_har.constants import ACTIVITY_LABELS, SIGNAL_NAMES
from deep_har.data import load_example_window
from deep_har.preprocess import ChannelStandardizer
from deep_har.visualization import plot_sensor_window

DEFAULT_MODEL = Path("outputs/checkpoints/cnn_lstm_best.keras")
DEFAULT_NORMALIZER = Path("outputs/preprocessing/normalizer.npz")
DEFAULT_SAMPLE = Path("examples/sample_window.csv")


def _load_runtime(model_path: str, normalizer_path: str):
    import tensorflow as tf

    model_path = Path(model_path)
    normalizer_path = Path(normalizer_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Train a model first.")
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Normalizer not found: {normalizer_path}. Train a model first.")
    model = tf.keras.models.load_model(model_path)
    normalizer = ChannelStandardizer.load(normalizer_path)
    return model, normalizer


def _predict(window: np.ndarray, model_path: str, normalizer_path: str):
    model, normalizer = _load_runtime(model_path, normalizer_path)
    X = normalizer.transform(window)
    proba = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(proba))
    probs = pd.DataFrame({"activity": ACTIVITY_LABELS, "probability": proba}).sort_values(
        "probability", ascending=False
    )
    return ACTIVITY_LABELS[idx], float(proba[idx]), probs


def predict_uploaded(file_obj, model_path: str, normalizer_path: str):
    if file_obj is None:
        path = DEFAULT_SAMPLE
    else:
        path = file_obj.name
    window = load_example_window(path)
    pred, confidence, probs = _predict(window, model_path, normalizer_path)
    fig = plot_sensor_window(window[0], title=f"Sensor window: predicted {pred}")
    return pred, f"{confidence:.4f}", probs, fig


def build_demo():
    import gradio as gr

    with gr.Blocks(title="DeepHAR-MLOps") as demo:
        gr.Markdown(
            """
            # DeepHAR-MLOps: Smartphone Human Activity Recognition

            Upload a single 128 x 9 sensor window as CSV/NPY or use the included sample.
            Train first with `python -m deep_har.train --model cnn_lstm --epochs 40`.
            """
        )
        with gr.Row():
            model_path = gr.Textbox(value=str(DEFAULT_MODEL), label="Model checkpoint path")
            normalizer_path = gr.Textbox(value=str(DEFAULT_NORMALIZER), label="Normalizer path")
        file_in = gr.File(label="Optional CSV/NPY sensor window")
        run_btn = gr.Button("Predict activity")
        with gr.Row():
            pred = gr.Textbox(label="Predicted activity")
            conf = gr.Textbox(label="Confidence")
        probs = gr.Dataframe(label="Class probabilities")
        plot = gr.Plot(label="Sensor plot")
        run_btn.click(
            predict_uploaded,
            inputs=[file_in, model_path, normalizer_path],
            outputs=[pred, conf, probs, plot],
        )
        gr.Markdown(
            """
            ## Expected input format
            CSV with 128 rows and 9 columns in this order:
            body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z,
            total_acc_x, total_acc_y, total_acc_z.
            """
        )
    return demo
