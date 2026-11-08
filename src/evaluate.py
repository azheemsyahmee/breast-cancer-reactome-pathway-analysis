"""
Evaluation utilities to assess fidelity and distribution matching
between real and synthetic biomedical pathway profiles.
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_synthetic_fidelity(
    real_data: np.ndarray,
    synthetic_data: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistical similarity metrics comparing real vs. synthetic pathway matrices.

    Args:
        real_data: Numpy array of shape (N, P) representing physical patient profiles.
        synthetic_data: Numpy array of shape (M, P) representing generated profiles.

    Returns:
        Dictionary of statistical summary metrics.
    """
    real_mean = float(np.mean(real_data))
    synthetic_mean = float(np.mean(synthetic_data))

    real_std = float(np.std(real_data))
    synthetic_std = float(np.std(synthetic_data))

    # Mean profile across samples: shape (1, P)
    real_profile = np.mean(real_data, axis=0).reshape(1, -1)
    synthetic_profile = np.mean(synthetic_data, axis=0).reshape(1, -1)

    cos_sim = float(cosine_similarity(real_profile, synthetic_profile)[0][0])

    return {
        "real_mean": real_mean,
        "synthetic_mean": synthetic_mean,
        "real_std": real_std,
        "synthetic_std": synthetic_std,
        "cosine_similarity": cos_sim
    }


def format_evaluation_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """Format evaluation metrics dictionary into a display DataFrame."""
    display_names = {
        "real_mean": "Real Mean",
        "synthetic_mean": "Synthetic Mean",
        "real_std": "Real Std",
        "synthetic_std": "Synthetic Std",
        "cosine_similarity": "Cosine Similarity"
    }
    return pd.DataFrame([
        {"Metric": display_names.get(k, k), "Value": v}
        for k, v in metrics.items()
    ])
