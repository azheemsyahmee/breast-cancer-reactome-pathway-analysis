"""
Unit tests for synthetic fidelity evaluation metrics.
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluate import evaluate_synthetic_fidelity, format_evaluation_table


class TestEvaluateModule(unittest.TestCase):

    def test_evaluate_synthetic_fidelity_identical(self):
        # When real and synthetic are identical, cosine similarity is 1.0
        np.random.seed(42)
        real = np.random.randn(100, 20)
        synthetic = real.copy()

        metrics = evaluate_synthetic_fidelity(real, synthetic)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=5)
        self.assertAlmostEqual(metrics["real_mean"], metrics["synthetic_mean"], places=5)
        self.assertAlmostEqual(metrics["real_std"], metrics["synthetic_std"], places=5)

    def test_evaluate_synthetic_fidelity_orthogonal(self):
        # Opposing profiles should yield distinct similarity
        real = np.ones((50, 10))
        synthetic = -1.0 * np.ones((50, 10))

        metrics = evaluate_synthetic_fidelity(real, synthetic)
        self.assertAlmostEqual(metrics["cosine_similarity"], -1.0, places=5)

    def test_format_evaluation_table(self):
        dummy_metrics = {
            "real_mean": 0.0,
            "synthetic_mean": 0.05,
            "real_std": 1.0,
            "synthetic_std": 0.5,
            "cosine_similarity": 0.8
        }
        df = format_evaluation_table(dummy_metrics)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertIn("Metric", df.columns)
        self.assertIn("Value", df.columns)


if __name__ == "__main__":
    unittest.main()
