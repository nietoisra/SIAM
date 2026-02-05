# SIAM 🚶‍♂️  
**System-Level Validation Architecture for Inertial Gait Analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Type-Methodological%20Framework-purple)](#intended-use)

---

## Conceptual Overview

**SIAM (System-Level Inertial Analysis of Motion)** is an open-source, modular, and reproducible **validation architecture** for inertial-sensor-based gait analysis systems.

SIAM is **not designed to optimize diagnostic accuracy** nor to propose novel machine-learning algorithms.  
Instead, it is explicitly designed to evaluate **how a complete gait-analysis pipeline behaves** under:

- strict subject-independent validation  
- demographic imbalance  
- small-sample uncertainty  
- deployment-relevant conditions  

> SIAM treats validation as an **architectural property**, not as a post-hoc performance estimate.

---

## Methodological Rationale

In inertial gait analysis research, performance metrics are frequently reported under subject-dependent or mixed validation schemes, leading to optimistic and non-deployable estimates.

SIAM addresses this by:
- enforcing **Leave-One-Subject-Out (LOSO)** evaluation
- treating LOSO as a **deliberate architectural stress test**
- characterizing **uncertainty and variability**, not just mean performance
- explicitly analyzing **demographic confounding**, rather than ignoring it

---

## Scope and Epistemological Position

### In Scope
- System-level validation of gait-analysis pipelines  
- Reproducible evaluation of preprocessing, feature extraction, and modeling interactions  
- Probabilistic calibration assessment  
- Demographic confounding characterization  

### Out of Scope
- Clinical diagnosis  
- Biomarker discovery  
- Performance optimization  
- Medical decision support  

SIAM is a **methodological framework**, not a clinical system.

---

## Architectural Design Principles

SIAM is built around the following principles:

1. **Subject Independence**  
   No data from a test subject is used during training or calibration.

2. **Validation as Stress Testing**  
   LOSO is used to amplify inter-subject variability and expose architectural fragility.

3. **Probabilistic Reliability**  
   Model confidence is evaluated through calibration metrics, not assumed.

4. **Demographic Transparency**  
   Potential confounders (e.g., biological sex) are explicitly quantified and reported.

5. **Reproducibility by Design**  
   Fixed preprocessing, deterministic pipelines, and version-controlled code.

---

## System-Level Pipeline

SIAM integrates the following stages into a single validation architecture:

1. Standardized inertial data acquisition  
2. Fixed and reproducible signal preprocessing  
3. Window-based feature extraction and aggregation  
4. Classical machine-learning models (baseline behavior)  
5. Subject-independent validation (LOSO)  
6. Probabilistic calibration assessment  
7. Demographic confounding analysis  
8. Visualization and reporting  

Each component is evaluated **as part of the system**, not in isolation.

---

## Validation Strategy

SIAM employs **Leave-One-Subject-Out (LOSO)** cross-validation.

In small-sample wearable studies, LOSO constitutes an **adverse validation regime** that:
- maximizes inter-subject heterogeneity
- amplifies uncertainty
- exposes overfitting and leakage
- reflects deployment-like conditions

Observed stability under LOSO is treated as a **methodological result**.

---

## Probabilistic Calibration

Beyond discrimination metrics, SIAM evaluates:
- Brier score
- reliability curves
- calibration stability across folds

This allows assessment of **predictive trustworthiness**, not just correctness.

---

## Demographic Confounding

SIAM explicitly characterizes demographic sensitivity at the feature level.

Rather than attempting to algorithmically “correct” confounding in underpowered cohorts, SIAM:
- quantifies demographic sensitivity
- reports its potential impact
- contextualizes observed system behavior

This design choice prioritizes **transparency over false adjustment**.

---

## Intended Use

SIAM is intended for:
- methodological research
- validation benchmarking
- reproducibility studies
- doctoral and master’s research
- biomedical systems engineering

It is **not intended for clinical deployment or diagnostic use**.

---

## Installation (Research Use)

```bash
git clone https://github.com/nietoisra/SIAM.git
cd SIAM
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Reproducibility Note
All preprocessing parameters, model configurations, and validation strategies are fixed a priori and documented to ensure reproducible system-level evaluation.


## Citation
If you use this framework in academic work, please cite:

```bibtex
@article{Nieto2025SIAM,
  author  = {Nieto, Israel Edgar and Alarcón Paredes, Antonio and Silverio, Gustavo Alonso},
  title   = {SIAM: An Open-Source Architecture for Inertial Gait Analysis and Characterization in Alzheimer’s Disease},
  journal = {Medical \& Biological Engineering \& Computing},
  year    = {2025},
  note    = {Manuscript under review}
}
```

## Disclaimer
This software is provided strictly for research and educational purposes.
No clinical claims are made or implied.
