"""Train and evaluate a skin-condition classifier on HAM10000."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def find_image_paths(metadata: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    image_dirs = [data_dir, data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]
    lookup = {}
    for image_dir in image_dirs:
        if image_dir.exists():
            lookup.update({path.stem: str(path) for path in image_dir.glob("*.jpg")})
    result = metadata.copy()
    result["image_path"] = result["image_id"].map(lookup)
    missing = result["image_path"].isna().sum()
    if missing:
        raise FileNotFoundError(
            f"Could not find {missing} image(s). Expected JPG files in {data_dir} "
            "or its HAM10000_images_part_1/2 subdirectories."
        )
    return result


def load_metadata(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found in {data_dir}. Download HAM10000_metadata.csv "
            "from Kaggle and place it in the dataset directory."
        )
    metadata = pd.read_csv(metadata_path)
    required = {"image_id", "dx"}
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    return find_image_paths(metadata[["image_id", "dx"]].dropna(), data_dir)


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(
        data, test_size=0.30, stratify=data["dx"], random_state=SEED
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, stratify=remainder["dx"], random_state=SEED
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def make_dataset(frame: pd.DataFrame, class_names: list[str], training: bool) -> tf.data.Dataset:
    labels = frame["dx"].map({name: index for index, name in enumerate(class_names)}).to_numpy()
    paths = frame["image_path"].to_numpy()

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return dataset.map(load_image, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)


def create_demo_dataset(data_dir: Path) -> None:
    """Create a tiny synthetic dataset to validate the pipeline without HAM10000."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_name in ("nv", "mel", "bkl"):
        for index in range(6):
            image_id = f"demo_{class_name}_{index}"
            image = tf.random.uniform((64, 64, 3), 0, 256, dtype=tf.int32, seed=SEED + index)
            image = tf.cast(image, tf.uint8)
            (data_dir / f"{image_id}.jpg").write_bytes(tf.io.encode_jpeg(image).numpy())
            rows.append({"image_id": image_id, "dx": class_name})
    pd.DataFrame(rows).to_csv(data_dir / "metadata.csv", index=False)


def build_model(number_of_classes: int, weights: str | None = "imagenet") -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
        ],
        name="augmentation",
    )
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(*IMAGE_SIZE, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("HAM10000"), help="HAM10000 dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--demo", action="store_true", help="Run an offline synthetic smoke test")
    parser.add_argument("--complete-epochs", action="store_true", help="Run every requested epoch without early stopping")
    args = parser.parse_args()

    set_seed()
    if args.demo:
        with tempfile.TemporaryDirectory() as temporary_dir:
            demo_dir = Path(temporary_dir)
            create_demo_dataset(demo_dir)
            run_training(args, demo_dir, weights=None)
        return
    run_training(args, args.data_dir)


def run_training(args: argparse.Namespace, data_dir: Path, weights: str | None = "imagenet") -> None:
    data = load_metadata(data_dir)
    class_names = sorted(data["dx"].unique())
    train, validation, test = split_data(data)
    train_dataset = make_dataset(train, class_names, training=True)
    validation_dataset = make_dataset(validation, class_names, training=False)
    test_dataset = make_dataset(test, class_names, training=False)

    class_ids = np.arange(len(class_names))
    class_weight_values = compute_class_weight(
        "balanced",
        classes=class_ids,
        y=train["dx"].map({name: i for i, name in enumerate(class_names)}),
    )
    class_weights = dict(enumerate(class_weight_values.tolist()))

    model = build_model(len(class_names), weights=weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output_dir / "skin_classifier.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_history.csv"),
    ]
    if not args.complete_epochs:
        callbacks.insert(0, tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    probabilities = model.predict(test_dataset, verbose=1)
    predictions = probabilities.argmax(axis=1)
    actual = test["dx"].map({name: i for i, name in enumerate(class_names)}).to_numpy()
    report = classification_report(actual, predictions, target_names=class_names, output_dict=True, zero_division=0)
    results = {
        "classes": class_names,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(actual, predictions).tolist(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}, indent=2))


if __name__ == "__main__":
    main()
