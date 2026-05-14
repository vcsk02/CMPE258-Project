import numpy as np

from deep_har.prompting import build_explanation_prompt, summarize_window


def test_summarize_window_has_expected_stats():
    window = np.ones((128, 9), dtype="float32")
    stats = summarize_window(window)
    assert "body_acc_x_mean" in stats
    assert "body_acc_magnitude_mean" in stats


def test_build_explanation_prompt_contains_prediction_context():
    window = np.zeros((128, 9), dtype="float32")
    probabilities = {
        "WALKING": 0.7,
        "WALKING_UPSTAIRS": 0.1,
        "WALKING_DOWNSTAIRS": 0.05,
        "SITTING": 0.05,
        "STANDING": 0.05,
        "LAYING": 0.05,
    }
    prompt = build_explanation_prompt("WALKING", 0.7, probabilities, window)
    assert "Prediction context" in prompt.user_prompt
    assert "Predicted activity: WALKING" in prompt.deterministic_explanation
