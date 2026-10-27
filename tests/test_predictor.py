import json
import tempfile
import unittest
from pathlib import Path

from gradcam import GradCAMError, generate_gradcam
from predictor import CropDiseasePredictor, InvalidImage, ModelUnavailable


class PredictorTests(unittest.TestCase):
    def test_missing_model_is_reported_without_fabricating_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            predictor = CropDiseasePredictor(Path(folder) / "missing.keras", Path(folder) / "classes.json")
            with self.assertRaises(ModelUnavailable):
                predictor.predict_array([[[0, 0, 0]]])

    def test_invalid_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            bad_image = Path(folder) / "bad.jpg"
            bad_image.write_text("not an image", encoding="utf-8")
            predictor = CropDiseasePredictor(Path(folder) / "model.keras", Path(folder) / "classes.json")
            with self.assertRaises(InvalidImage):
                predictor.preprocess(bad_image)

    def test_label_parser_handles_plantvillage_names(self):
        self.assertEqual(
            CropDiseasePredictor.describe_label("Tomato___Early_blight"),
            ("Tomato", "Early blight"),
        )

    def test_gradcam_reports_unsupported_model(self):
        with self.assertRaises(GradCAMError):
            generate_gradcam(object(), [[[0.0, 0.0, 0.0]]], 0)

    def test_gradcam_computes_for_nested_convolutional_backbone(self):
        import numpy as np
        import tensorflow as tf

        backbone_input = tf.keras.Input((16, 16, 3))
        activation = tf.keras.layers.Conv2D(4, 3, activation="relu", name="feature_conv")(backbone_input)
        backbone = tf.keras.Model(backbone_input, activation, name="tiny_backbone")
        model_input = tf.keras.Input((16, 16, 3))
        features = backbone(model_input)
        pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
        output = tf.keras.layers.Dense(2, activation="softmax")(pooled)
        model = tf.keras.Model(model_input, output)

        heatmap = generate_gradcam(model, np.ones((16, 16, 3), dtype=np.float32), 0)

        self.assertEqual(heatmap.ndim, 2)
        self.assertTrue(np.all(np.isfinite(heatmap)))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)


if __name__ == "__main__":
    unittest.main()