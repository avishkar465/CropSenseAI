import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def split_dataset(data_dir: Path, seed: int = 42):
    classes = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two class-named subfolders.")
    randomizer = random.Random(seed)
    splits = {name: ([], []) for name in ("train", "validation", "test")}
    class_names = [directory.name for directory in classes]
    for class_index, directory in enumerate(classes):
        images = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) < 5:
            raise ValueError(f"Class '{directory.name}' needs at least 5 images for train/validation/test splits.")
        randomizer.shuffle(images)
        test_count = max(1, round(len(images) * 0.15))
        validation_count = max(1, round(len(images) * 0.15))
        test_paths = images[:test_count]
        validation_paths = images[test_count:test_count + validation_count]
        train_paths = images[test_count + validation_count:]
        for name, paths in (("train", train_paths), ("validation", validation_paths), ("test", test_paths)):
            splits[name][0].extend(str(path) for path in paths)
            splits[name][1].extend([class_index] * len(paths))
    return class_names, splits


def make_dataset(paths, labels, image_size, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        raw = tf.io.read_file(path)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(paths), 2048), seed=42, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(class_count: int, image_size: int = 224, weights: str | None = "imagenet"):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="leaf_image")
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights=weights
    )
    backbone.trainable = False
    features = backbone(augmentation(inputs), training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = tf.keras.layers.Dropout(0.25)(features)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="class_probabilities")(features)
    return tf.keras.Model(inputs, outputs, name="cropsense_mobilenetv2"), backbone


def calculate_metrics(model, dataset, class_names):
    actual_batches = []
    predicted_batches = []
    for images, labels in dataset:
        actual_batches.append(labels.numpy())
        probabilities = model(images, training=False)
        predicted_batches.append(np.asarray(probabilities).argmax(axis=1))
    actual = np.concatenate(actual_batches)
    predicted = np.concatenate(predicted_batches)
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            actual, predicted, labels=labels, target_names=class_names, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train CropSense AI using MobileNetV2 transfer learning.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "PlantVillage")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "model")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"Dataset folder not found: {args.data_dir}")

    class_names, splits = split_dataset(args.data_dir)
    datasets = {
        name: make_dataset(paths, labels, (args.image_size, args.image_size), args.batch_size, name == "train")
        for name, (paths, labels) in splits.items()
    }
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, backbone = build_model(
        len(class_names),
        args.image_size,
        None if args.weights == "none" else "imagenet"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best_checkpoint.keras",
            monitor="val_loss",
            save_best_only=True
        ),
    ]

    history = defaultdict(list)

    first = model.fit(datasets["train"], validation_data=datasets["validation"], epochs=args.epochs, callbacks=callbacks)
    for key, values in first.history.items():
        history[key].extend(float(value) for value in values)

    if args.fine_tune_epochs > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        second = model.fit(datasets["train"], validation_data=datasets["validation"], epochs=args.fine_tune_epochs, callbacks=callbacks)
        for key, values in second.history.items():
            history[key].extend(float(value) for value in values)

    model_path = args.output_dir / "crop_disease_model.keras"
    model.save(model_path)
    (args.output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(dict(history), indent=2), encoding="utf-8")
    metrics = calculate_metrics(model, datasets["test"], class_names)
    (args.output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(max(8, len(class_names) * 0.45), 7))
    axis.imshow(metrics["confusion_matrix"], cmap="Greens")
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names, yticklabels=class_names, xlabel="Predicted", ylabel="Actual")
    plt.setp(axis.get_xticklabels(), rotation=90, ha="right")
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    print(json.dumps({key: metrics[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}, indent=2))
    print(f"Saved evaluated model: {model_path}")


if __name__ == "__main__":
    main()
    