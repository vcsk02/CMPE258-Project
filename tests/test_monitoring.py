import numpy as np

from deep_har.monitoring import compute_channel_profile, drift_score


def test_drift_score_low_for_reference_like_window():
    X = np.zeros((4, 128, 9), dtype="float32")
    profile = compute_channel_profile(X)
    result = drift_score(np.zeros((128, 9), dtype="float32"), profile)
    assert result["drift_level"] == "low"
