"""
Data loading and preprocessing utilities for Wisconsin Breast Cancer
and Reactome biological pathway activity matrices.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def load_clinical_data() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
    
    Returns:
        X: DataFrame of 569 samples with 30 morphological cell features.
        y: Target array (0 = Malignant, 1 = Benign).
        feature_names: List of feature names.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y, list(data.feature_names)


def load_reactome_pathways(gmt_path: str) -> Dict[str, List[str]]:
    """
    Parse a Reactome GMT pathway file.
    
    Args:
        gmt_path: Path to the .gmt file.
        
    Returns:
        Dictionary mapping pathway names to lists of member gene symbols.
    """
    pathways = {}
    if not os.path.exists(gmt_path):
        raise FileNotFoundError(f"Reactome GMT file not found at: {gmt_path}")
        
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathway_name = parts[0]
                genes = parts[2:]
                pathways[pathway_name] = genes
    return pathways


def generate_pathway_activity_matrix(
    X_clinical: pd.DataFrame,
    pathways: Dict[str, List[str]],
    n_pathways: int = 50,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Generate and standardize a simulated pathway-level activity matrix
    matching the biological feature statistics of the breast cancer cohort.
    
    Args:
        X_clinical: Clinical feature DataFrame.
        pathways: Reactome pathway dictionary.
        n_pathways: Number of top pathways to sample.
        random_state: Random seed for reproducibility.
        
    Returns:
        pathway_df: Raw pathway activity DataFrame.
        X_scaled: Standardized numpy array of shape (N, n_pathways).
        scaler: Fitted StandardScaler instance.
    """
    np.random.seed(random_state)
    
    pathway_names = list(pathways.keys())[:n_pathways] if len(pathways) > 0 else [
        f"Pathway_{i}" for i in range(n_pathways)
    ]
    
    loc = float(np.mean(X_clinical.values))
    scale = float(np.std(X_clinical.values))
    n_samples = len(X_clinical)
    
    pathway_data = {}
    for p in pathway_names:
        pathway_data[p] = np.random.normal(loc=loc, scale=scale, size=n_samples)
        
    pathway_df = pd.DataFrame(pathway_data)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pathway_df)
    
    return pathway_df, X_scaled, scaler
