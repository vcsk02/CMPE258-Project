from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .constants import ACTIVITY_LABELS, SIGNAL_NAMES


@dataclass(frozen=True)
class PromptExplanation:
    """Prompt-engineering artifact for the LLM/foundation-model component.

    The trained HAR model produces probabilities. This module converts that
    structured prediction plus lightweight sensor statistics into a controlled
    natural-language prompt that can be pasted into an LLM or sent to an LLM
    endpoint in a deployed version. A deterministic fallback explanation is
    included so the app remains runnable without API keys.
    """

    system_prompt: str
    user_prompt: str
    deterministic_explanation: str


def summarize_window(window: np.ndarray) -> dict[str, float]:
    """Return compact numeric features used by the explanation prompt.

    Args:
        window: Single sensor window with shape ``(128, channels)``.
    """
    if window.ndim != 2:
        raise ValueError(f"Expected a single 2D window, got shape {window.shape}")

    stats: dict[str, float] = {}
    channels = min(window.shape[1], len(SIGNAL_NAMES))
    for idx in range(channels):
        name = SIGNAL_NAMES[idx]
        values = window[:, idx]
        stats[f"{name}_mean"] = float(np.mean(values))
        stats[f"{name}_std"] = float(np.std(values))
        stats[f"{name}_abs_energy"] = float(np.mean(values**2))

    if channels >= 3:
        acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
        stats["body_acc_magnitude_mean"] = float(np.mean(acc_mag))
        stats["body_acc_magnitude_std"] = float(np.std(acc_mag))
    if channels >= 6:
        gyro_mag = np.sqrt(np.sum(window[:, 3:6] ** 2, axis=1))
        stats["body_gyro_magnitude_mean"] = float(np.mean(gyro_mag))
        stats["body_gyro_magnitude_std"] = float(np.std(gyro_mag))
    return stats


def _top_k(probabilities: Mapping[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]


def build_explanation_prompt(
    predicted_label: str,
    confidence: float,
    probabilities: Mapping[str, float],
    window: np.ndarray,
    include_stats: bool = True,
) -> PromptExplanation:
    """Create a controlled LLM prompt and deterministic fallback explanation."""
    if predicted_label not in ACTIVITY_LABELS:
        raise ValueError(f"Unknown activity label: {predicted_label}")

    top = _top_k(probabilities)
    top_text = ", ".join(f"{label}={score:.3f}" for label, score in top)
    stats = summarize_window(window) if include_stats else {}
    selected_stats = {
        key: value
        for key, value in stats.items()
        if key.endswith("magnitude_mean") or key.endswith("magnitude_std")
    }
    stat_text = "\n".join(f"- {key}: {value:.4f}" for key, value in selected_stats.items())
    if not stat_text:
        stat_text = "- Sensor summary unavailable for this channel subset."

    system_prompt = (
        "You are an ML model explanation assistant for a human activity recognition demo. "
        "Explain predictions cautiously. Do not claim medical certainty. Mention uncertainty, "
        "likely failure modes, and what data was used. Keep the answer under 120 words."
    )
    user_prompt = f"""Prediction context:
- Model output: {predicted_label}
- Confidence: {confidence:.3f}
- Top probabilities: {top_text}
- Input: 2.56-second smartphone IMU window with 128 timesteps
- Sensor statistics:
{stat_text}

Write a concise explanation for a project demo audience. Include one caveat and one possible next step for improving reliability."""

    if confidence < 0.55:
        uncertainty = "The model is uncertain, so this prediction should be treated as a low-confidence demo result."
    elif top[0][1] - top[1][1] < 0.10 if len(top) > 1 else False:
        uncertainty = "The top classes are close, so the app should surface ambiguity instead of hiding it."
    else:
        uncertainty = "The prediction is reasonably confident for this demo input."

    deterministic_explanation = (
        f"Predicted activity: {predicted_label} with confidence {confidence:.3f}. "
        f"The strongest competing classes were {top_text}. {uncertainty} "
        "A known failure mode is confusion between activities with similar motion signatures, "
        "especially static postures such as SITTING and STANDING. A next step is to monitor "
        "confidence and input drift before trusting the model in a new phone placement."
    )

    return PromptExplanation(system_prompt, user_prompt, deterministic_explanation)
