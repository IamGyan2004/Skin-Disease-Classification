"""Evaluate a saved HAM10000 model and write classification metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .data_pipeline import load_metadata, make_dataset, split_data


def evaluate(data_dir: Path, model_path: Path, output_path: Path) -> dict:
    data = load_metadata(data_dir)
    class_names = sorted(data["dx"].unique())
    _, _, test_frame = split_data(data)
    dataset = make_dataset(test_frame, class_names, training=False)
    predictions = tf.keras.models.load_model(model_path).predict(dataset, verbose=1).argmax(axis=1)
    label_map = {name: index for index, name in enumerate(class_names)}
    actual = test_frame["dx"].map(label_map).to_numpy()
    report = classification_report(actual, predictions, target_names=class_names, output_dict=True, zero_division=0)
    results = {"classes": class_names, "classification_report": report, "confusion_matrix": confusion_matrix(actual, predictions).tolist()}
    output_path.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, default=Path("models/skin_disease_model.keras"))
    parser.add_argument("--output", type=Path, default=Path("models/metrics.json"))
    args = parser.parse_args()
    results = evaluate(args.data_dir, args.model, args.output)
    print(json.dumps({"accuracy": results["classification_report"]["accuracy"], "macro_f1": results["classification_report"]["macro avg"]["f1-score"]}, indent=2))


if __name__ == "__main__":
    main()
