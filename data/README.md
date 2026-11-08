# Dataset Provenance & Pathway Specifications

This repository utilizes two complementary biomedical data resources:

## 1. Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
* **Source:** UCI Machine Learning Repository / scikit-learn (`sklearn.datasets.load_breast_cancer`)
* **Cohort Size ($N$):** 569 patient samples
  - Malignant: 212 samples (37.3%)
  - Benign: 357 samples (62.7%)
* **Feature Count ($P$):** 30 real-valued morphological features computed from digitized images of fine needle aspirates (FNA) of breast masses (e.g. mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
* **Usage:** Provides the empirical biological mean and variance statistics used to parametrize pathway-level activity profiles.

## 2. Reactome Biological Pathway GMT
* **File:** `ReactomePathways_2023-03-06.gmt`
* **Source:** Reactome Knowledgebase (Release 84, March 2023)
* **License:** Creative Commons Public Domain (CC0)
* **Pathway Count:** 2,593 curated biological pathways
* **Format:** Gene Matrix Transposed (`.gmt`), where each row denotes:
  `[Pathway Name]\t[Pathway ID or Source]\t[Gene 1]\t[Gene 2]\t...`
* **Cohort Sampling:** The top 50 biological pathways are extracted to form the multi-dimensional activity space for generative modeling and optimization experiments.
