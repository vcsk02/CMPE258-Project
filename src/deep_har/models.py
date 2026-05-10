from __future__ import annotations

from typing import Callable


def _tf():
    import tensorflow as tf
    return tf


def build_mlp(input_shape: tuple[int, int] = (128, 9), num_classes: int = 6):
    tf = _tf()
    layers = tf.keras.layers
    inp = tf.keras.Input(shape=input_shape, name="sensor_window")
    x = layers.Flatten()(inp)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="mlp")


def build_cnn(input_shape: tuple[int, int] = (128, 9), num_classes: int = 6):
    tf = _tf()
    layers = tf.keras.layers
    inp = tf.keras.Input(shape=input_shape, name="sensor_window")
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="cnn")


def build_cnn_lstm(input_shape: tuple[int, int] = (128, 9), num_classes: int = 6):
    tf = _tf()
    layers = tf.keras.layers
    inp = tf.keras.Input(shape=input_shape, name="sensor_window")
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(96, return_sequences=False)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="cnn_lstm")


def _tcn_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float):
    tf = _tf()
    layers = tf.keras.layers
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
    x = layers.Add()([shortcut, x])
    return layers.Activation("relu")(x)


def build_tcn(input_shape: tuple[int, int] = (128, 9), num_classes: int = 6):
    tf = _tf()
    layers = tf.keras.layers
    inp = tf.keras.Input(shape=input_shape, name="sensor_window")
    x = inp
    for dilation in [1, 2, 4, 8]:
        x = _tcn_block(x, filters=64, kernel_size=5, dilation_rate=dilation, dropout=0.20)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="tcn")


def build_attention(input_shape: tuple[int, int] = (128, 9), num_classes: int = 6):
    tf = _tf()
    layers = tf.keras.layers
    inp = tf.keras.Input(shape=input_shape, name="sensor_window")
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.10)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="cnn_bilstm_attention")


MODEL_REGISTRY: dict[str, Callable] = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "cnn_lstm": build_cnn_lstm,
    "tcn": build_tcn,
    "attention": build_attention,
}


def build_model(model_name: str, input_shape: tuple[int, int], num_classes: int = 6):
    if model_name not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{model_name}'. Valid values: {valid}")
    return MODEL_REGISTRY[model_name](input_shape=input_shape, num_classes=num_classes)


def compile_model(model, learning_rate: float = 1e-3):
    tf = _tf()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
