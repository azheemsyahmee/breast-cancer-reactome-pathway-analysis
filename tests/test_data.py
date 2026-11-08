"""
Unit tests for data loading and pathway matrix generation.
"""

import os
import unittest
import numpy as np

from src.data import (
    load_clinical_data,
    load_reactome_pathways,
    generate_pathway_activity_matrix
)


class TestDataModule(unittest.TestCase):

    def setUp(self):
        # Resolve path to sample GMT or real GMT
        self.gmt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "ReactomePathways_2023-03-06.gmt"
        )

    def test_load_clinical_data(self):
        X, y, feature_names = load_clinical_data()
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(len(y), 569)
        self.assertEqual(len(feature_names), 30)
        # Check target labels are binary (0 and 1)
        self.assertEqual(set(np.unique(y)), {0, 1})

    def test_load_reactome_pathways(self):
        if not os.path.exists(self.gmt_path):
            self.skipTest("Reactome GMT file not found at default path.")
        pathways = load_reactome_pathways(self.gmt_path)
        self.assertGreater(len(pathways), 2000)
        # Check that each pathway maps to a non-empty list of gene symbols
        first_pw = list(pathways.keys())[0]
        self.assertIsInstance(pathways[first_pw], list)
        self.assertGreater(len(pathways[first_pw]), 0)

    def test_generate_pathway_activity_matrix(self):
        X, _, _ = load_clinical_data()
        dummy_pathways = {f"PW_{i}": ["GENE_A", "GENE_B"] for i in range(10)}
        pathway_df, X_scaled, scaler = generate_pathway_activity_matrix(
            X_clinical=X,
            pathways=dummy_pathways,
            n_pathways=10,
            random_state=42
        )
        self.assertEqual(pathway_df.shape, (569, 10))
        self.assertEqual(X_scaled.shape, (569, 10))
        # Standardized matrix should have mean close to 0 and std close to 1
        np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
