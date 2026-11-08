"""
Hyperparameter optimization modules comparing Optuna Bayesian optimization
against exhaustive Grid Search for GAN training.
"""

import time
import itertools
from typing import Dict, Any, Tuple
import torch
import optuna

from src.train import train_gan


def run_optuna_tuning(
    tensor_data: torch.Tensor,
    n_trials: int = 10,
    epochs: int = 50,
    random_seed: int = 42
) -> Tuple[Dict[str, Any], float, float, optuna.Study]:
    """
    Execute Bayesian hyperparameter optimization using Optuna.

    Args:
        tensor_data: Standardized tensor of real pathway profiles.
        n_trials: Number of Bayesian trials to evaluate.
        epochs: Epoch count per trial.
        random_seed: Random seed for study sampler.

    Returns:
        best_params: Dictionary of best hyperparameter values.
        best_loss: Best generator loss achieved.
        wall_time: Total elapsed time in seconds.
        study: The completed Optuna Study object.
    """
    def objective(trial: optuna.Trial) -> float:
        latent_dim = trial.suggest_int("latent_dim", 10, 50)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        g_loss, _, _ = train_gan(
            tensor_data=tensor_data,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            random_seed=random_seed
        )
        return g_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    t_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    wall_time = time.time() - t_start

    return study.best_params, study.best_value, wall_time, study


def run_grid_search(
    tensor_data: torch.Tensor,
    param_grid: Dict[str, list] = None,
    epochs: int = 50,
    random_seed: int = 42
) -> Tuple[Dict[str, Any], float, float, list]:
    """
    Execute exhaustive Grid Search across the specified parameter combinations.

    Args:
        tensor_data: Standardized tensor of real pathway profiles.
        param_grid: Dictionary defining parameter search spaces.
        epochs: Epoch count per combination.
        random_seed: Random seed for initialization.

    Returns:
        best_params: Dictionary of best hyperparameter values.
        best_loss: Best generator loss achieved.
        wall_time: Total elapsed time in seconds.
        all_results: List of all evaluated trial result dictionaries.
    """
    if param_grid is None:
        param_grid = {
            "latent_dim": [10, 20, 30, 40, 50],
            "hidden_dim": [32, 64, 128],
            "lr": [0.0001, 0.001, 0.01]
        }

    t_start = time.time()
    best_loss = float("inf")
    best_params = None
    all_results = []

    for latent_dim, hidden_dim, lr in itertools.product(
        param_grid["latent_dim"],
        param_grid["hidden_dim"],
        param_grid["lr"]
    ):
        g_loss, d_loss, _ = train_gan(
            tensor_data=tensor_data,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            random_seed=random_seed
        )

        record = {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "g_loss": g_loss,
            "d_loss": d_loss
        }
        all_results.append(record)

        if g_loss < best_loss:
            best_loss = g_loss
            best_params = {"latent_dim": latent_dim, "hidden_dim": hidden_dim, "lr": lr}

    wall_time = time.time() - t_start
    return best_params, best_loss, wall_time, all_results
