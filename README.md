# Breast Cancer Reactome Pathway Analysis: GAN Synthesis & Optuna Bayesian Optimization

[![CI Tests](https://github.com/azheemsyahmee/breast-cancer-reactome-pathway-analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/azheemsyahmee/breast-cancer-reactome-pathway-analysis/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)

A modular computational biology framework investigating **Generative Adversarial Networks (GAN)** for biomedical pathway activity synthesis, providing an empirical comparative benchmark between **Optuna Bayesian hyperparameter optimization** and **exhaustive Grid Search**.

---

## 1. Problem Overview & Motivation

In computational oncology and systems biology, high-throughput transcriptomic and clinical profiling is frequently constrained by limited patient sample sizes and privacy considerations. Generative Adversarial Networks (GANs) offer a promising avenue to synthesize realistic pathway-level expression profiles for downstream analysis and data augmentation.

However, training GANs in biomedical domains is notoriously challenging due to:
1. **Adversarial Instability:** Delicate balance between Generator and Discriminator dynamics.
2. **Hyperparameter Sensitivity:** Extreme susceptibility to latent noise dimensionality ($z$), hidden layer width, and learning rate schedules.
3. **High Optimization Overhead:** Traditional exhaustive grid search across multi-dimensional hyperparameter spaces requires prohibitive computational cost.

This repository explores whether **Bayesian optimization (Optuna with Tree-structured Parzen Estimators / TPE)** can achieve competitive generative loss profiles compared to exhaustive grid search while significantly reducing training wall-clock time and carbon footprint.

---

## 2. Dataset & Pathway Activity Modeling

The pipeline integrates two complementary biological data sources:
* **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset:** 569 physical patient specimens characterized across 30 morphological cell features from digitized fine needle aspirate (FNA) images. Loaded via `sklearn.datasets.load_breast_cancer`.
* **Reactome Biological Pathways Knowledgebase:** 2,593 curated functional biological pathway gene sets (`data/ReactomePathways_2023-03-06.gmt`).
* **Pathway Activity Representation:** The top 50 Reactome pathways are sampled to construct a 50-dimensional biological feature space parameterized by the empirical cohort distribution and standardized with `StandardScaler` to ensure zero-mean training stability.

---

## 3. Generative Adversarial Architecture

The framework implements a fully differentiable PyTorch GAN tailored for continuous pathway activity vectors:

```
[ Latent Vector z ] ~ N(0, I) (dim: 10-50)
        │
        ▼
┌─────────────────────────┐
│       GENERATOR         │
│  Linear(latent, hidden) │
│  ReLU()                 │
│  Linear(hidden, 50)     │
└───────────┬─────────────┘
            │
            ▼
    [ Synthetic Profile ] ──┐
                            ├─► ┌─────────────────────────┐
    [ Real Profile (N=569)]─┘   │      DISCRIMINATOR      │
                                │  Linear(50, hidden)     │ ──► P(Real) in [0, 1]
                                │  ReLU()                 │     (BCE Loss)
                                │  Linear(hidden, 1)      │
                                │  Sigmoid()              │
                                └─────────────────────────┘
```

* **Loss Function:** Binary Cross-Entropy (`BCELoss`) with real labels ($1.0$) and synthetic labels ($0.0$).
* **Optimizers:** Dual Adam optimizers for Generator and Discriminator with zeroed gradients at each iteration.

---

## 4. Hyperparameter Optimization: Optuna vs. Grid Search

We evaluated both tuning methodologies across an identical search space:
* **Latent Dimension (`latent_dim`):** 10 to 50
* **Hidden Dimension (`hidden_dim`):** 32, 64, 128
* **Learning Rate (`lr`):** $10^{-4}$ to $10^{-2}$ (log scale)
* **Epochs per Trial:** 50 epochs

### Empirical Benchmark Comparison

| Optimization Method | Evaluation Strategy | Trials / Combinations | Best Generator Loss | Best Discriminator Loss | Optimization Wall Time | Optimal Hyperparameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Optuna (Bayesian TPE)** | Guided sequential sampling | **10 trials** | **0.6477** | — | **2.22 seconds** | `latent_dim=11, hidden_dim=128, lr=0.00012` |
| **Exhaustive Grid Search** | Combinatorial grid ($5 \times 3 \times 3$) | **45 combinations** | **0.5686** | 1.2865 | **8.11 seconds** | `latent_dim=20, hidden_dim=64, lr=0.01` |

```
                       OPTIMIZATION TIME & LOSS TRADEOFF
    ┌───────────────────────┬───────────────────────────┬───────────────────┐
    │ Method                │ Time Taken                │ Best Gen Loss     │
    ├───────────────────────┼───────────────────────────┼───────────────────┤
    │ Optuna (Bayesian TPE) │ █████ 2.22s               │ 0.6477            │
    │ Exhaustive Grid Search│ ██████████████████ 8.11s  │ 0.5686            │
    └───────────────────────┴───────────────────────────┴───────────────────┘
    * Optuna achieves comparable convergence with a 72.7% reduction in compute time (3.66x speedup).
```

### Analytical Takeaways
1. **Computational Efficiency:** Optuna discovered a competitive configuration within only 10 trials, demonstrating the sample efficiency of Tree-structured Parzen Estimators over brute-force enumeration.
2. **Learning Rate Stability:** Bayesian optimization favored a conservative learning rate ($\approx 10^{-4}$) coupled with wider hidden capacity (`hidden_dim=128`), producing smoother adversarial loss curves.
3. **Grid Search Exploitation:** While grid search produced a marginally lower generator loss ($0.5686$ vs $0.6477$), it required 4.5x more evaluations and settled on a higher learning rate ($0.01$) with lower hidden capacity (`hidden_dim=64`), which exhibited higher variance across epochs.

---

## 5. Synthetic Data Fidelity & Statistical Limitations

Following optimization, synthetic pathway instances ($N=569$) were sampled from the optimal Generator and statistically contrasted against real patient profiles:

| Evaluation Metric | Physical Data | Synthetic GAN Output | Statistical Interpretation |
| :--- | :---: | :---: | :--- |
| **Mean** | $-5.24 \times 10^{-18}$ | $0.0234$ | Excellent zero-centering alignment. |
| **Standard Deviation** | $1.0000$ | $0.3362$ | Variance reduction (synthetic data is concentrated near mean). |
| **Global Cosine Similarity** | $1.0000$ | $1.63 \times 10^{-17}$ | Weak linear co-expression alignment across the 50-pathway manifold. |

### Critical Scientific Scope & Limitations
1. **Mode Collapse & Variance Shrinkage:** The synthetic standard deviation ($0.336$) is noticeably lower than the physical cohort ($1.000$). This variance shrinkage indicates partial mode collapse, where the generator produces plausible average values but fails to replicate the extreme tails of clinical phenotypes.
2. **Near-Zero Cosine Similarity ($1.63 \times 10^{-17}$):** Although marginal distributions overlap, the multi-pathway covariance structure was not captured by a simple 2-layer MLP architecture trained for 50 epochs. Real biological co-regulation networks require deeper architectures, Wasserstein loss with gradient penalty (WGAN-GP), or diffusion models.
3. **Simulated Pathway Matrix:** Because the Wisconsin dataset provides morphological FNA cell features rather than direct microarray/RNA-seq read counts, pathway-level activity was modeled from empirical cohort statistics. This project serves as a methodological benchmark for optimization algorithms rather than an in vivo biological discovery pipeline.

---

## 6. Repository Organization

```
breast-cancer-reactome-pathway-analysis/
├── .github/workflows/
│   └── tests.yml                     # Automated GitHub Actions CI workflow (Python 3.11)
├── data/
│   ├── README.md                     # Dataset provenance & Reactome pathway specifications
│   └── ReactomePathways_2023-03-06.gmt # 2,593 Reactome pathway definitions (CC0)
├── docs/
│   ├── figures/                      # High-resolution benchmark visual plots
│   │   ├── part1_data_exploration.png
│   │   ├── part4_comparison.png
│   │   ├── part4_optuna_trials.png
│   │   ├── part5_evaluation.png
│   │   └── part5_std_comparison.png
│   └── LabEx2_Report_Azheem_B23CS0031.pdf # Full submitted academic report
├── notebooks/
│   └── 01_reactome_gan_optimization.ipynb # Complete reproducible research notebook
├── src/
│   ├── __init__.py
│   ├── data.py                       # Data loading and pathway matrix generation
│   ├── models.py                     # PyTorch Generator and Discriminator architectures
│   ├── train.py                      # Core GAN training loop
│   ├── optimize.py                   # Optuna Bayesian tuning and Grid Search benchmark
│   └── evaluate.py                   # Statistical similarity and fidelity evaluation
├── tests/
│   ├── __init__.py
│   ├── test_data.py                  # Data loading and shape validation tests
│   ├── test_evaluate.py              # Statistical metric tests
│   ├── test_models.py                # PyTorch architecture & forward pass tests
│   └── run_tests.py                  # Standalone test runner
├── .gitignore
├── LICENSE                           # MIT License
├── requirements.txt                  # Minimal pinned dependencies
└── README.md                         # This documentation
```

---

## 7. Quick Start & Reproduction

### Installation
```bash
git clone https://github.com/azheemsyahmee/breast-cancer-reactome-pathway-analysis.git
cd breast-cancer-reactome-pathway-analysis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run Unit Tests
```bash
python tests/run_tests.py
# or via unittest discovery:
python -m unittest discover -s tests -v
```

### Run Python API
```python
from src.data import load_clinical_data, load_reactome_pathways, generate_pathway_activity_matrix
from src.optimize import run_optuna_tuning
from src.evaluate import evaluate_synthetic_fidelity
import torch

# 1. Load data
X, _, _ = load_clinical_data()
pathways = load_reactome_pathways("data/ReactomePathways_2023-03-06.gmt")
_, X_scaled, _ = generate_pathway_activity_matrix(X, pathways, n_pathways=50)

tensor_data = torch.tensor(X_scaled, dtype=torch.float32)

# 2. Run Optuna tuning
best_params, best_loss, wall_time, _ = run_optuna_tuning(tensor_data, n_trials=10, epochs=50)
print(f"Optimal parameters: {best_params} (Loss: {best_loss:.4f}, Time: {wall_time:.2f}s)")
```

---

## 8. Academic Attribution & Provenance

* **Institution:** Universiti Teknologi Malaysia (UTM)
* **Faculty:** Faculty of Computing
* **Course:** SECB 3223 — Computational Biology 2 (Academic Session 2025/2026 – 02)
* **Lecturer:** Dr. Shamini
* **Author:** Azheem Syahmee bin Zulkifli (Matric: `B23CS0031` / [@azheemsyahmee](https://github.com/azheemsyahmee))
* **Assessment Mode:** 100% Individual Coursework (Submitted May 2026)
* **Report:** Complete verified submission report preserved at [`docs/LabEx2_Report_Azheem_B23CS0031.pdf`](docs/LabEx2_Report_Azheem_B23CS0031.pdf).
* **License:** Licensed under the [MIT License](LICENSE).
