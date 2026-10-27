import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Verify saved model files and optionally run image inference.")
    parser.add_argument("--image", type=Path, help="Optional image file for a real inference check.")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "model" / "crop_disease_model.keras")
    parser.add_argument("--class-names", type=Path, default=BASE_DIR / "model" / "class_names.json")
    args = parser.parse_args()
    if not args.model.is_file() or not args.class_names.is_file():
        raise SystemExit("Model or class label file is missing. No inference was attempted.")
    model = tf.keras.models.load_model(args.model)
    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    if int(model.output_shape[-1]) != len(class_names):
        raise SystemExit("Model output count does not match class label count.")
    print(f"Loaded model with {len(class_names)} classes; output shape: {model.output_shape}.")
    if args.image:
        with Image.open(args.image) as image:
            image = image.convert("RGB").resize((224, 224))
            values = np.asarray(image, dtype=np.float32)
        values = tf.keras.applications.mobilenet_v2.preprocess_input(values)[None, ...]
        probabilities = model.predict(values, verbose=0)[0]
        best = int(probabilities.argmax())
        print(f"Inference completed: {class_names[best]} ({probabilities[best] * 100:.2f}%).")


if __name__ == "__main__":
    main()