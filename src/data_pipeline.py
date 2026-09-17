"""HAM10000 metadata loading, image lookup, splitting, and tf.data input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .preprocessing import load_image


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def load_metadata(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found in {data_dir}.")

    metadata = pd.read_csv(metadata_path)
    required = {"image_id", "dx"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    image_dirs = [
        data_dir,
        data_dir / "HAM10000_images_part_1",
        data_dir / "HAM10000_images_part_2",
        data_dir.parent / "images" / "HAM10000_images_part_1",
        data_dir.parent / "images" / "HAM10000_images_part_2",
    ]
    lookup: dict[str, str] = {}
    for image_dir in image_dirs:
        if image_dir.exists():
            lookup.update({path.stem: str(path) for path in image_dir.glob("*.jpg")})
    result = metadata[["image_id", "dx"]].dropna().copy()
    result["image_path"] = result["image_id"].map(lookup)
    missing_count = int(result["image_path"].isna().sum())
    if missing_count:
        raise FileNotFoundError(f"Could not find {missing_count} image(s) in {data_dir}.")
    return result


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(data, test_size=0.30, stratify=data["dx"], random_state=SEED)
    validation, test = train_test_split(
        remainder, test_size=0.50, stratify=remainder["dx"], random_state=SEED
    )
    return tuple(frame.reset_index(drop=True) for frame in (train, validation, test))


def make_dataset(frame: pd.DataFrame, class_names: list[str], training: bool) -> tf.data.Dataset:
    label_map = {name: index for index, name in enumerate(class_names)}
    labels = frame["dx"].map(label_map).to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((frame["image_path"].to_numpy(), labels))
    if training:
        dataset = dataset.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return dataset.map(load_image, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)
