# SIAM 🚶‍♂️  
**System-Level Validation Framework for Inertial Gait Analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Research](https://img.shields.io/badge/Type-Research%20Software-purple)](#intended-use)

---

## Overview

**SIAM** (System-Level Inertial Analysis of Motion) is an **open-source, modular, and reproducible research framework** designed for the **methodological validation of inertial gait analysis pipelines** using wearable IMU sensors and machine learning models.

The framework focuses on **system-level validation**, addressing common pitfalls in gait analysis research such as subject leakage, improper cross-validation, demographic confounding, and over-optimistic performance reporting.

> ⚠️ **SIAM is not a clinical diagnostic tool.**  
> It is intended exclusively for **methodological research, benchmarking, and exploratory studies**.

---

## Rationale

Inertial gait analysis studies often suffer from:
- Subject-dependent data leakage
- Insufficient validation strategies
- Lack of probabilistic calibration
- Ignoring demographic confounding variables
- Overinterpretation of small cohorts

SIAM was developed to provide a **controlled, transparent, and statistically grounded framework** that enables:
- Reproducible experimentation
- Fair comparison between models
- Explicit acknowledgment of uncertainty and limitations

---

## Scope and Intended Contribution

SIAM is designed to:
- Validate **complete analysis pipelines**, not isolated models
- Support **exploratory and early-stage research**
- Encourage **best practices in validation and reporting**

Out of scope:
- Clinical diagnosis or decision-making
- Large-scale epidemiological inference
- Real-time embedded inference

---

## Architectural Principles

SIAM follows five core principles:

1. **Subject Independence** – strict separation between training and testing subjects  
2. **Reproducibility** – deterministic pipelines and explicit configuration  
3. **Transparency** – interpretable features and models  
4. **Statistical Prudence** – emphasis on uncertainty and effect sizes  
5. **Modularity** – interchangeable components across the pipeline  

---

## System Overview

The framework is organized into modular stages:

1. Data acquisition  
2. Signal preprocessing  
3. Feature extraction and aggregation  
4. Machine learning modeling  
5. Validation and calibration  
6. Statistical and demographic analysis  
7. Visualization and reporting  

---

## Data Acquisition

- Wearable IMU sensors
- Triaxial acceleration and angular velocity
- Fixed sampling frequency
- Structured acquisition protocols under controlled conditions

Data are organized per subject to enforce **subject-level validation**.

---

## Signal Preprocessing

Typical preprocessing steps include:
- Noise filtering
- Segmentation and windowing
- Signal normalization

All preprocessing steps are configurable and reproducible.

---

## Feature Extraction and Aggregation

SIAM extracts interpretable gait features from:
- Time domain
- Frequency domain
- Statistical descriptors

Features are aggregated at the **subject level**, avoiding trial-level leakage.

---

## Machine Learning Models

Baseline and classical models are supported, including:
- Logistic Regression
- Support Vector Machines (SVM)
- k-Nearest Neighbors
- Tree-based classifiers

The framework emphasizes **model comparison under identical conditions** rather than model complexity.

---

## Validation Strategy

SIAM employs **Leave-One-Subject-Out (LOSO)** cross-validation to ensure:
- True subject independence
- Realistic generalization estimates

Performance metrics are reported as **distributions**, not single-point estimates.

---

## Probabilistic Calibration

Beyond accuracy-based metrics, SIAM evaluates:
- Prediction confidence
- Calibration curves
- Reliability of probabilistic outputs

This enables analysis of **model trustworthiness**, not just correctness.

---

## Demographic Confounding Analysis

The framework explicitly analyzes potential confounders such as:
- Sex
- Age (when available)

Performance stratification is used to identify systematic biases.

---

## Statistical Perspective

SIAM adopts a **descriptive and exploratory statistical stance**, prioritizing:
- Effect sizes
- Variability
- Uncertainty awareness

Hypothesis testing is secondary to **robust methodological insight**.

---

## Software Implementation

- **Language:** Python  
- **Core libraries:** NumPy, SciPy, scikit-learn  
- **Visualization & UI:** Streamlit  

The framework is designed for **readability and extensibility**, not black-box optimization.

---

## Computational Performance

- Designed for offline analysis
- Suitable for small-to-medium cohorts
- Emphasis on methodological correctness over runtime optimization

---

## Limitations

Known limitations include:
- Small cohort sizes
- Single-sensor configurations
- Controlled acquisition environments

These limitations are **explicitly acknowledged and reported**, not hidden.

---

## Intended Use

SIAM is intended for:
- Academic research
- Methodological benchmarking
- Reproducibility studies
- Doctoral and master’s theses

**Not intended for:**
- Clinical diagnosis
- Medical decision support
- Commercial healthcare products

---

## Installation

```bash
git clone https://github.com/nietoisra/SIAM.git
cd SIAM
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
