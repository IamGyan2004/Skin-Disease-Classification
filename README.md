# Skin Disease Classification - HAM10000

CNN-based multi-class classification of dermoscopic images from the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset.

This is an educational/research classification model, not a medical diagnosis tool. Do not use its predictions for clinical decisions.

## Project structure

```text
dataset/HAM10000_metadata.csv
images/HAM10000_images_part_1/
images/HAM10000_images_part_2/
models/skin_disease_model.keras
notebooks/skin_disease_classification.ipynb
src/{preprocessing,data_pipeline,train,evaluate}.py
app.py
```

## Setup

Use Python 3.10-3.13. TensorFlow does not currently provide a compatible wheel for Python 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download HAM10000 from Kaggle and arrange the files like this:

```text
dataset/
	HAM10000_metadata.csv

images/
	HAM10000_images_part_1/
		ISIC_0024306.jpg
	HAM10000_images_part_2/
		ISIC_0034310.jpg
```

The canonical pipeline reads metadata from `dataset/` and images from the sibling `images/` directory.

## Train and evaluate

```bash
python -m src.train --data-dir ./dataset --output-dir ./models --epochs 15
```

To verify the complete pipeline without downloading HAM10000, run the small offline smoke test:

```bash
python train.py --demo --epochs 1
```

The demo uses generated images and random CNN weights only to test the software path; its metrics are not meaningful medical results.

The command performs a stratified 70/15/15 split, resizes images to 224x224, applies augmentation, trains an ImageNet-initialized EfficientNetB0 classifier, compensates for class imbalance, and reports accuracy, precision, recall, F1-score, and a confusion matrix.

Evaluate a saved model with `python -m src.evaluate --data-dir ./dataset --model ./models/skin_disease_model.keras`.

Launch the research demo with `streamlit run app.py`. The application accepts a JPG, JPEG, or PNG image and displays a predicted dataset class with a clear non-diagnostic warning.

Outputs are written to `artifacts/`:

- `skin_classifier.keras`: best saved model
- `training_history.csv`: epoch history
- `metrics.json`: per-class metrics and confusion matrix

This project is intended for research and education, not clinical diagnosis.