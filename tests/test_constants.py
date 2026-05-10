from deep_har.constants import ACTIVITY_LABELS, SENSOR_GROUPS, SIGNAL_NAMES


def test_label_and_signal_counts():
    assert len(ACTIVITY_LABELS) == 6
    assert len(SIGNAL_NAMES) == 9
    assert SENSOR_GROUPS["all"] == list(range(9))
