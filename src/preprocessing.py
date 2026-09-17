"""Image preprocessing and augmentation utilities."""

from __future__ import annotations

import tensorflow as tf


IMAGE_SIZE = (224, 224)


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode a JPEG and resize it for EfficientNetB0."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32), label


def create_augmentation() -> tf.keras.Sequential:
    """Return training-only image augmentation layers."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
        ],
        name="augmentation",
    )
