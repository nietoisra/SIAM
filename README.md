# SIAM  
### A System-Level Validation Framework for Inertial Gait Analysis

SIAM (System for Inertial Gait Analysis and Modeling) is an open-source, modular, and reproducible framework designed for **system-level methodological validation of inertial gait-analysis pipelines**. The framework prioritizes **subject-independent evaluation, probabilistic reliability, and explicit characterization of demographic confounding**, particularly in small-sample wearable sensing studies.

SIAM is **not** proposed as a clinically validated diagnostic system.  
Its purpose is methodological: to provide a **reference validation architecture** for evaluating how complete inertial gait-analysis systems behave under **strict and deployment-relevant validation conditions**.

---

## Rationale

Inertial measurement units (IMUs) are widely adopted for quantitative gait analysis in ageing and neurological research. Despite their accessibility and growing use, methodological progress in this field is frequently constrained by:

- Subject-dependent or mixed-subject validation strategies  
- Optimistic performance estimates driven by subject-specific signal leakage  
- Limited reporting of probabilistic calibration and predictive reliability  
- Insufficient analysis of demographic confounding (e.g., biological sex)  
- Lack of reproducible, system-level validation pipelines  

As a result, reported classification performance often reflects favorable validation conditions rather than genuine subject-independent generalization.

SIAM addresses these issues by shifting emphasis from algorithmic novelty to **validation architecture**, treating the computational pipeline itself as a biomedical system whose robustness must be empirically demonstrated.

---

## Scope and Intended Contribution

SIAM demonstrates system behavior using a pilot cohort of older adults with and without Alzheimer’s disease as a **representative use case**. This cohort is intentionally small and demographically constrained, reflecting realistic conditions encountered in exploratory wearable studies.

Accordingly:

- SIAM does **not** aim to establish diagnostic validity  
- SIAM does **not** identify disease-specific biomarkers  
- Reported metrics describe **system-level behavior**, not clinical performance  

The framework is intended to support **transparent methodological evaluation**, not clinical decision-making.

---

## Architectural Principles

SIAM is guided by five design principles aligned with best practices in biomedical systems engineering:

1. **Strict subject independence**  
   Leave-one-subject-out (LOSO) validation is mandatory and treated as an architectural stress test.

2. **Reliability beyond discrimination**  
   Probabilistic calibration is evaluated alongside accuracy-based metrics.

3. **Explicit confounding analysis**  
   Demographic sensitivity is quantified and reported rather than implicitly assumed away.

4. **Fixed and transparent pipelines**  
   Preprocessing, feature extraction, and model parameters are defined *a priori*.

5. **Reproducibility over optimization**  
   No dataset-specific hyperparameter tuning or model selection using test data.

---
## System Overview

SIAM is organized as a **fixed, end-to-end validation architecture** in which each stage is executed sequentially and evaluated under strict subject-independent conditions. The pipeline is intentionally designed to expose architectural fragility, uncertainty, and demographic sensitivity rather than to optimize task-specific performance.


Inertial Data Acquisition

└─ Standardized single-IMU recording (trunk-mounted)

↓

Signal Preprocessing
└─ Fixed band-pass filtering and windowing (no subject-specific tuning)
↓
Window-Based Feature Extraction
└─ Time-, frequency-, variability-, and complexity-based descriptors
↓
Trial-Level Feature Aggregation
└─ Robust median aggregation across windows
↓
Classical Machine Learning Models
└─ Interpretable models with fixed hyperparameters
↓
Leave-One-Subject-Out Validation (LOSO)
└─ Subject-independent architectural stress test
↓
Probabilistic Calibration Assessment
└─ Reliability curves and Brier score on held-out subjects
↓
Demographic Confounding Characterization
└─ Feature-level sensitivity analysis (e.g., biological sex)

---

## System Overview

Inertial Data Acquisition
↓
Signal Preprocessing
↓
Window-Based Feature Extraction
↓
Trial-Level Aggregation
↓
Classical Machine Learning Models
↓
Leave-One-Subject-Out Validation
↓
Probabilistic Calibration Assessment
↓
Demographic Confounding Characterization


Each stage is modular, documented, and hardware-agnostic.

---

## Data Acquisition

- **Sensor**: Single wearable IMU (Bosch BMI270)  
- **Signals**: Triaxial acceleration and triaxial angular velocity  
- **Placement**: Lumbar region (L4–L5)  
- **Sampling frequency**: 100 Hz  
- **Protocol**: Overground walking at slow, preferred, and fast speeds  
- **Output format**: CSV  

Walking speed conditions are treated as controlled within-subject variability rather than explicit experimental factors, reflecting deployment-like heterogeneity.

---

## Signal Preprocessing

All preprocessing parameters are fixed prior to analysis and applied identically across subjects:

- Zero-phase Butterworth band-pass filtering (0.5–20 Hz)  
- Fixed-length windowing:
  - Window duration: 2.0 s  
  - Overlap: 50%  

No subject-specific filtering, normalization, or tuning is performed.

---

## Feature Extraction and Aggregation

SIAM employs window-based feature extraction to prioritize robustness and reproducibility over fine-grained event detection.

Extracted features include:

- Time-domain amplitude measures  
- Variability descriptors  
- Frequency-domain and spectral metrics  
- Complexity and entropy-based measures  
- Asymmetry indices  

Features are computed per axis and aggregated at the trial level using the median to reduce sensitivity to transient artifacts and outliers.

This design avoids reliance on gait event detection, which is known to be unstable in pathological or heterogeneous populations.

---

## Machine Learning Models

SIAM uses classical, well-characterized models suitable for small-sample biomedical studies:

- Logistic Regression  
- Linear Support Vector Machines  
- Random Forests  

Key constraints:

- Fixed hyperparameters defined *a priori*  
- Feature scaling performed within each LOSO training fold  
- Balanced class weighting  
- Probabilistic outputs enabled for all models  

The objective is to evaluate **architectural stability**, not to maximize performance.

---

## Validation Strategy

### Leave-One-Subject-Out (LOSO)

Each validation fold excludes all data from one participant, ensuring complete subject independence.

In small-sample wearable studies, LOSO represents a deliberately adverse validation regime that amplifies inter-subject variability and uncertainty. Stability under LOSO is therefore interpreted as evidence of **system-level robustness**, not favorable dataset structure.

---

## Probabilistic Calibration

SIAM explicitly evaluates the reliability of probabilistic predictions:

- Brier score  
- Reliability (calibration) curves  
- Post-hoc calibration using isotonic regression  

All calibration procedures are performed exclusively within training folds, and evaluation is conducted on held-out subjects only.

This ensures that reported probabilities reflect genuine subject-independent generalization.

---

## Demographic Confounding Analysis

Given well-established biomechanical sex differences in gait, SIAM includes explicit characterization of sex-related feature sensitivity:

- Feature-level effect sizes (Cohen’s *d*)  
- Comparison across feature families  
- Restricted sex-homogeneous robustness analyses  

Confounding is **quantified and reported**, not algorithmically corrected when cohort design precludes causal disentanglement.

---

## Statistical Perspective

Analyses are descriptive and methodological in nature:

- Emphasis on effect sizes and uncertainty  
- Bootstrap-based confidence intervals  
- No null-hypothesis significance testing for clinical inference  
- No multiple-comparison correction  

This approach is consistent with exploratory system-validation studies.

---

## Software Implementation

- **Language**: Python 3.10  
- **Libraries**:
  - NumPy  
  - SciPy  
  - scikit-learn  
  - Streamlit (visualization)  
- Modular, hardware-agnostic design  
- Fixed random seeds for reproducibility  
- Version-controlled using Git  

---

## Computational Performance

End-to-end processing time is approximately **3 seconds per 60-second walking trial** on standard workstation hardware. Runtime and memory usage remain stable across validation folds and subjects.

These results indicate feasibility for research-scale deployment but do not imply clinical readiness.

---

## Limitations

The framework is demonstrated using a small, demographically imbalanced pilot cohort and a single lumbar-mounted sensor under controlled conditions. These constraints are intentionally retained and explicitly documented to reflect common limitations in exploratory wearable research.

SIAM does not attempt to overcome cohort limitations algorithmically; instead, it provides tools to **expose and contextualize** their impact on system behavior.

---

## Intended Use

SIAM is intended for:

- Methodological research in inertial gait analysis  
- Validation benchmarking under subject-independent conditions  
- Studies of probabilistic calibration and uncertainty  
- Reproducible biomedical systems engineering research  

It is **not intended for clinical diagnosis or decision support**.

---

## Citation

If you use SIAM in academic work, please cite:

> Nieto Granados, I. E., Alarcón Paredes, A., & Alonso Silverio, G.  
> *SIAM: An Open-Source Architecture for Inertial Gait Analysis and Characterization in Alzheimer’s Disease.*  
> Manuscript under review.

---

## Closing Remark

SIAM formalizes the view that **validation strategy is a first-class component of biomedical systems**, not a post hoc procedural detail. By embedding subject-independent evaluation, probabilistic reliability, and demographic transparency into a unified architecture, SIAM supports reproducible and trustworthy development of inertial gait-analysis systems.


SIAM integrates the complete inertial gait-analysis pipeline within a single reproducible architecture:

