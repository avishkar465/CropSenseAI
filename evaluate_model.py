import argparse
import json
from pathlib import Path

import tensorflow as tf

from train_model import calculate_metrics, make_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved CropSense model on the held-out test split.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "PlantVillage")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "model" / "crop_disease_model.keras")
    parser.add_argument("--class-names", type=Path, default=Path(__file__).resolve().parent / "model" / "class_names.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not args.model.is_file() or not args.class_names.is_file():
        raise SystemExit("Model or class label file is missing. Run train_model.py first.")
    class_names, splits = split_dataset(args.data_dir)
    saved_classes = json.loads(args.class_names.read_text(encoding="utf-8"))
    if saved_classes != class_names:
        raise SystemExit("Dataset class folders differ from model/class_names.json; evaluation would be misleading.")
    test_paths, test_labels = splits["test"]
    dataset = make_dataset(test_paths, test_labels, (224, 224), args.batch_size)
    model = tf.keras.models.load_model(args.model)
    metrics = calculate_metrics(model, dataset, class_names)
    print(json.dumps({key: metrics[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}, indent=2))


if __name__ == "__main__":
    main()