import unittest

import numpy as np

from train_model import calculate_metrics


class ArrayLabels:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


class CountingDataset:
    def __init__(self):
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield np.zeros((4, 1)), ArrayLabels([0, 1, 1, 0])


class FakeModel:
    def __call__(self, images, training=False):
        self.training = training
        return np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])


class TrainingMetricsTests(unittest.TestCase):
    def test_metrics_visit_dataset_once_and_preserve_results(self):
        dataset = CountingDataset()
        model = FakeModel()

        metrics = calculate_metrics(model, dataset, ["healthy", "diseased"])

        self.assertEqual(dataset.iterations, 1)
        self.assertFalse(model.training)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision_macro"], 1.0)
        self.assertEqual(metrics["recall_macro"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)


if __name__ == "__main__":
    unittest.main()