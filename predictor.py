import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class ModelUnavailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    pass


class CropDiseasePredictor:
    def __init__(self, model_path: Path, class_names_path: Path, image_size=(224, 224)):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.image_size = image_size
        self._model = None
        self._class_names = None

    def _load(self):
        if self._model is not None and self._class_names is not None:
            return
        if not self.model_path.is_file() or not self.class_names_path.is_file():
            raise ModelUnavailable(
                "The trained crop model is not installed. Train it with the documented dataset first."
            )
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(self.model_path)
            class_names = json.loads(self.class_names_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelUnavailable("The crop model could not be loaded. Check the model files and TensorFlow version.") from exc
        if not isinstance(class_names, list) or not class_names:
            raise ModelUnavailable("The model class label file is empty or invalid.")
        if int(model.output_shape[-1]) != len(class_names):
            raise ModelUnavailable("The model output does not match the saved class labels.")
        self._model = model
        self._class_names = class_names

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def class_names(self):
        self._load()
        return self._class_names

    def preprocess(self, image_source):
        try:
            with Image.open(image_source) as image:
                image = image.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
                return np.asarray(image, dtype=np.float32)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImage("The uploaded file is not a readable image.") from exc

    @staticmethod
    def describe_label(label: str) -> tuple[str, str]:
        crop, separator, disease = label.partition("___")
        if not separator:
            crop, separator, disease = label.partition("__")
        crop = crop.replace("_", " ").strip().title()
        disease = disease.replace("_", " ").strip().replace("  ", " ") if separator else ""
        return crop or "Unknown crop", disease or crop or label

    def predict_array(self, image_array):
        self._load()
        import tensorflow as tf

        values = np.asarray(image_array, dtype=np.float32)
        if values.ndim == 3:
            values = values[None, ...]
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(values.copy())
        try:
            probabilities = np.asarray(self._model.predict(batch, verbose=0)[0], dtype=np.float64)
        except Exception as exc:
            raise ModelUnavailable("The model failed during inference. Check the saved model and TensorFlow version.") from exc
        if (
            probabilities.ndim != 1
            or probabilities.size != len(self._class_names)
            or not np.all(np.isfinite(probabilities))
        ):
            raise ModelUnavailable("The model returned an unexpected prediction shape.")
        indexes = np.argsort(probabilities)[::-1][:3]
        results = []
        for index in indexes:
            label = self._class_names[int(index)]
            crop, disease = self.describe_label(label)
            results.append({
                "label": label,
                "crop_name": crop,
                "disease_name": disease,
                "confidence": float(probabilities[index]),
                "class_index": int(index),
            })
        return results

    def predict(self, image_source):
        return self.predict_array(self.preprocess(image_source))


__all__ = ["CropDiseasePredictor", "InvalidImage", "ModelUnavailable"]