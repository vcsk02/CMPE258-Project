ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

SENSOR_GROUPS = {
    "all": list(range(9)),
    "body_acc": [0, 1, 2],
    "body_gyro": [3, 4, 5],
    "total_acc": [6, 7, 8],
    "accel_all": [0, 1, 2, 6, 7, 8],
}

TIMESTEPS = 128
NUM_CLASSES = len(ACTIVITY_LABELS)
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)
