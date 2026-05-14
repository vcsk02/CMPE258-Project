from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deep_har.constants import ACTIVITY_LABELS, SIGNAL_NAMES
from deep_har.data import load_example_window
from deep_har.monitoring import drift_score, load_reference_profile
from deep_har.numpy_baseline import load_centroid_model
from deep_har.preprocess import ChannelStandardizer
from deep_har.prompting import build_explanation_prompt
from deep_har.visualization import plot_sensor_window

# Default to the packaged dependency-light demo artifact so a clean clone can
# perform inference immediately. After training a deep model, replace this with
# outputs/checkpoints/cnn_lstm_best.keras or sensor_transformer_best.keras.
DEFAULT_MODEL = Path("outputs/checkpoints/demo_centroid_model.npz")
DEFAULT_NORMALIZER = Path("outputs/preprocessing/normalizer.npz")
DEFAULT_REFERENCE_PROFILE = Path("outputs/preprocessing/reference_profile.json")
DEFAULT_SAMPLE = Path("examples/sample_window.csv")


def _predict_with_model(model_path: Path, X: np.ndarray) -> np.ndarray:
    if model_path.suffix.lower() == ".npz":
        return load_centroid_model(model_path).predict_proba(X)
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for .keras checkpoints. The packaged demo uses "
            "outputs/checkpoints/demo_centroid_model.npz and does not require TensorFlow."
        ) from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Train a model first.")
    model = tf.keras.models.load_model(model_path)
    return model.predict(X, verbose=0)


def _predict(window: np.ndarray, model_path: str, normalizer_path: str):
    model_path = Path(model_path)
    normalizer_path = Path(normalizer_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}.")
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Normalizer not found: {normalizer_path}.")
    normalizer = ChannelStandardizer.load(normalizer_path)
    X = normalizer.transform(window)
    proba = _predict_with_model(model_path, X)[0]
    idx = int(np.argmax(proba))
    probs = pd.DataFrame({"activity": ACTIVITY_LABELS, "probability": proba}).sort_values(
        "probability", ascending=False
    )
    probability_map = {label: float(score) for label, score in zip(ACTIVITY_LABELS, proba)}
    return ACTIVITY_LABELS[idx], float(proba[idx]), probs, probability_map


def _monitoring_markdown(window: np.ndarray, reference_profile_path: str) -> str:
    path = Path(reference_profile_path)
    if not path.exists():
        return (
            "### Monitoring status\n"
            "Reference profile not found. Train or generate artifacts first to create "
            "`outputs/preprocessing/reference_profile.json`."
        )
    result = drift_score(window[0], load_reference_profile(path))
    return (
        "### Monitoring status\n"
        f"- Drift level: **{result['drift_level']}**\n"
        f"- Drift score: `{result['drift_score']:.3f}`\n"
        "- Production action: log prediction, confidence, latency, and drift score. "
        "Trigger review/retraining when confidence drops or drift stays elevated."
    )


def predict_uploaded(file_obj, model_path: str, normalizer_path: str, reference_profile_path: str):
    if file_obj is None:
        path = DEFAULT_SAMPLE
    else:
        path = file_obj.name
    window = load_example_window(path)
    pred, confidence, probs, probability_map = _predict(window, model_path, normalizer_path)
    fig = plot_sensor_window(window[0], title=f"Sensor window: predicted {pred}")
    prompt = build_explanation_prompt(pred, confidence, probability_map, window[0])
    prompt_markdown = (
        "### Prompt-engineered explanation\n"
        f"{prompt.deterministic_explanation}\n\n"
        "### LLM prompt artifact\n"
        "**System prompt**\n\n"
        f"```text\n{prompt.system_prompt}\n```\n\n"
        "**User prompt**\n\n"
        f"```text\n{prompt.user_prompt}\n```"
    )
    monitoring = _monitoring_markdown(window, reference_profile_path)
    return pred, f"{confidence:.4f}", probs, fig, prompt_markdown, monitoring


def build_demo():
    import gradio as gr

    with gr.Blocks(title="DeepHAR-MLOps") as demo:
        gr.Markdown(
            """
            # DeepHAR-MLOps: Smartphone Human Activity Recognition

            Upload a single 128 x 9 sensor window as CSV/NPY or use the included sample.
            The default packaged `.npz` checkpoint supports immediate inference from a clean clone.
            For the deep-learning submission model, train with
            `python -m deep_har.train --model sensor_transformer --epochs 40` and switch the
            model path to the generated `.keras` checkpoint.
            """
        )
        with gr.Row():
            model_path = gr.Textbox(value=str(DEFAULT_MODEL), label="Model checkpoint path")
            normalizer_path = gr.Textbox(value=str(DEFAULT_NORMALIZER), label="Normalizer path")
        reference_profile_path = gr.Textbox(value=str(DEFAULT_REFERENCE_PROFILE), label="Reference drift profile path")
        file_in = gr.File(label="Optional CSV/NPY sensor window")
        run_btn = gr.Button("Predict activity")
        with gr.Row():
            pred = gr.Textbox(label="Predicted activity")
            conf = gr.Textbox(label="Confidence")
        probs = gr.Dataframe(label="Class probabilities")
        plot = gr.Plot(label="Sensor plot")
        prompt_output = gr.Markdown(label="Prompt explanation")
        monitoring_output = gr.Markdown(label="Monitoring")
        run_btn.click(
            predict_uploaded,
            inputs=[file_in, model_path, normalizer_path, reference_profile_path],
            outputs=[pred, conf, probs, plot, prompt_output, monitoring_output],
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


if __name__ == "__main__":
    build_demo().launch()
