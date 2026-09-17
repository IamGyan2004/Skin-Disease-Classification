"""Train the HAM10000 EfficientNetB0 classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .data_pipeline import load_metadata, make_dataset, split_data
from .preprocessing import IMAGE_SIZE, create_augmentation


SEED = 42


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def build_model(number_of_classes: int, weights: str | None = "imagenet") -> tf.keras.Model:
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(*IMAGE_SIZE, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = create_augmentation()(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(data_dir: Path, output_dir: Path, epochs: int) -> None:
    set_seed()
    data = load_metadata(data_dir)
    class_names = sorted(data["dx"].unique())
    train_frame, validation_frame, test_frame = split_data(data)
    train_dataset = make_dataset(train_frame, class_names, training=True)
    validation_dataset = make_dataset(validation_frame, class_names, training=False)
    test_dataset = make_dataset(test_frame, class_names, training=False)
    label_map = {name: index for index, name in enumerate(class_names)}
    class_ids = np.arange(len(class_names))
    weights = compute_class_weight("balanced", classes=class_ids, y=train_frame["dx"].map(label_map))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(len(class_names))
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=dict(enumerate(weights.tolist())),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(output_dir / "skin_disease_model.keras", monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.CSVLogger(output_dir / "training_history.csv"),
        ],
    )
    predictions = model.predict(test_dataset, verbose=1).argmax(axis=1)
    actual = test_frame["dx"].map(label_map).to_numpy()
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(actual, predictions, target_names=class_names, output_dict=True, zero_division=0)
    (output_dir / "metrics.json").write_text(json.dumps({
        "classes": class_names,
        "split_sizes": {"train": len(train_frame), "validation": len(validation_frame), "test": len(test_frame)},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(actual, predictions).tolist(),
    }, indent=2))
    print(json.dumps({"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
