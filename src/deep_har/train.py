from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .augment import augment_training_data
from .constants import ACTIVITY_LABELS, NUM_CLASSES, SENSOR_GROUPS
from .data import load_dataset, select_sensor_subset
from .models import MODEL_REGISTRY, build_model, compile_model
from .preprocess import ChannelStandardizer
from .visualization import plot_confusion, plot_training_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepHAR models on UCI HAR data.")
    parser.add_argument("--model", default="cnn_lstm", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sensor-subset", default="all", choices=sorted(SENSOR_GROUPS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--quick-run", action="store_true", help="Use a small subset for smoke tests.")
    parser.add_argument("--use-mlflow", action="store_true")
    return parser.parse_args()


def _maybe_start_mlflow(args):
    if not args.use_mlflow:
        return None
    try:
        import mlflow
        mlflow.set_experiment("deep-har-mlops")
        run = mlflow.start_run(run_name=f"{args.model}_{args.sensor_subset}")
        mlflow.log_params(vars(args))
        return mlflow
    except Exception as exc:
        print(f"MLflow disabled because it failed to start: {exc}")
        return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    plot_dir = output_dir / "plots"
    prep_dir = output_dir / "preprocessing"
    tb_dir = output_dir / "tensorboard"
    for d in [ckpt_dir, result_dir, plot_dir, prep_dir, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data_dir)
    X_train_raw = select_sensor_subset(data["X_train"], args.sensor_subset)
    X_test_raw = select_sensor_subset(data["X_test"], args.sensor_subset)
    y_train = data["y_train"]
    y_test = data["y_test"]

    if args.quick_run:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(len(X_train_raw), size=min(1200, len(X_train_raw)), replace=False)
        test_idx = rng.choice(len(X_test_raw), size=min(500, len(X_test_raw)), replace=False)
        X_train_raw, y_train = X_train_raw[train_idx], y_train[train_idx]
        X_test_raw, y_test = X_test_raw[test_idx], y_test[test_idx]

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=y_train,
    )

    normalizer = ChannelStandardizer().fit(X_tr_raw)
    X_tr = normalizer.transform(X_tr_raw)
    X_val = normalizer.transform(X_val_raw)
    X_test = normalizer.transform(X_test_raw)
    normalizer_path = prep_dir / "normalizer.npz"
    normalizer.save(normalizer_path)

    if args.augment:
        X_tr, y_tr = augment_training_data(X_tr, y_tr, seed=args.seed)

    input_shape = X_tr.shape[1:]
    model = build_model(args.model, input_shape=input_shape, num_classes=NUM_CLASSES)
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    import tensorflow as tf

    checkpoint_path = ckpt_dir / f"{args.model}_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-6,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir / args.model)),
        tf.keras.callbacks.CSVLogger(str(result_dir / f"{args.model}_history.csv")),
    ]

    mlflow = _maybe_start_mlflow(args)
    start = time.time()
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - start

    y_proba = model.predict(X_test, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, target_names=ACTIVITY_LABELS, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model": args.model,
        "sensor_subset": args.sensor_subset,
        "input_shape": list(input_shape),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "train_time_seconds": train_time,
        "normalizer_path": str(normalizer_path),
        "checkpoint_path": str(checkpoint_path),
    }

    metrics_path = result_dir / f"{args.model}_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (result_dir / f"{args.model}_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savetxt(result_dir / f"{args.model}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    np.savetxt(result_dir / f"{args.model}_predictions.csv", y_pred, delimiter=",", fmt="%d")

    plot_training_curves(history.history, plot_dir / f"{args.model}_training_curves.png")
    plot_confusion(y_test, y_pred, plot_dir / f"{args.model}_confusion_matrix.png")

    if mlflow is not None:
        mlflow.log_metrics({"test_accuracy": acc, "test_f1_macro": f1_macro, "train_time_seconds": train_time})
        mlflow.log_artifact(str(metrics_path))
        mlflow.end_run()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
