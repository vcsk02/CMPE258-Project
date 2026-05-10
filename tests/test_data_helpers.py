import numpy as np

from deep_har.data import select_sensor_subset


def test_select_sensor_subset():
    X = np.zeros((5, 128, 9), dtype="float32")
    assert select_sensor_subset(X, "all").shape == (5, 128, 9)
    assert select_sensor_subset(X, "body_acc").shape == (5, 128, 3)
    assert select_sensor_subset(X, "accel_all").shape == (5, 128, 6)
