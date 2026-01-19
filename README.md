Below is the **updated full README**, with the **inspiration properly integrated** in a dedicated **“Acknowledgements and References”** section, keeping the tone **professional, senior, and defensible**.

You can copy-paste this directly as `README.md`.

---

# Engine Fault Detection using Autoencoder (Anomaly Detection)

## Overview

This project implements an **Autoencoder-based anomaly detection approach** to identify faulty engine behavior from multivariate sensor data.
The model is trained exclusively on **healthy operating conditions**, learning a compact representation of normal behavior. Samples that deviate from this learned manifold are detected through **reconstruction error**.

The notebook focuses on:

* principled data preparation,
* model design choices aligned with anomaly detection theory,
* transparent and defensible evaluation using reconstruction error distributions and thresholding.

---

## Problem Statement

In many industrial systems, **fault data is scarce, imbalanced, or incomplete**, making supervised classification difficult or unreliable.
Anomaly detection offers an alternative by modeling **normal behavior only**, allowing detection of both known and unknown failure modes.

The objective of this project is to:

* learn normal engine behavior from healthy samples,
* detect anomalous (faulty) samples via reconstruction error,
* evaluate detection quality using decision-oriented metrics.

---

## Dataset

* **Samples**: ~16,000 healthy samples used for training and validation
* **Features**: 14 numerical sensor variables
* **Faulty samples**: held out for evaluation only

### Data Preparation

* Only **healthy data** is used during training.
* Features are scaled using **MinMaxScaler** to the ([0,1]) range.
* Sample-level normalization (`Normalizer`) was intentionally **not used**, as it removes magnitude information critical for anomaly detection.

---

## Modeling Approach

### Autoencoder Architecture

* Fully connected (Dense) Autoencoder
* Symmetric encoder–decoder structure
* Bottleneck layer to enforce dimensionality reduction
* Output layer configured to reconstruct normalized inputs

The Autoencoder is optimized using reconstruction loss and trained with **early stopping** to prevent memorization of healthy data.

---

## Reconstruction Error

For each sample, reconstruction error is computed as a **per-sample Mean Absolute Error (MAE)** across features:

[
\text{Reconstruction Error}*i = \frac{1}{d} \sum*{j=1}^{d} |x_{ij} - \hat{x}_{ij}|
]

This error serves as the **anomaly score**:

* low values → normal behavior
* high values → anomalous behavior

---

## Model Evaluation

### Separation Analysis

Reconstruction error distributions are compared between:

* Healthy (train / validation)
* Faulty (test)

A clear rightward shift of faulty samples indicates that the Autoencoder has successfully learned the normal operating manifold.

### Threshold Selection

An anomaly threshold is defined using **only healthy validation data**, based on a high percentile (e.g., 99th percentile).
This approach enables explicit control of the expected **false positive rate** on healthy data and avoids information leakage.

### Metrics

Once the threshold is applied, performance is evaluated using:

* False Positive Rate (healthy samples)
* Recall (faulty samples)
* Precision
* F1-score
* Confusion Matrix visualization

These metrics support **decision-oriented evaluation**, rather than purely descriptive analysis.

---

## Interpretability

To provide insight into detected anomalies:

* Feature-wise absolute reconstruction error is computed
* Bar plots highlight which variables contribute most to the anomaly score

This analysis supports **expert interpretation** and exploratory diagnosis, without implying direct causality.

---

## Key Design Principles

* Train only on healthy data
* Preserve physical meaning and scale of features
* Avoid assumptions about reconstruction error distributions
* Use percentile-based thresholding
* Favor interpretability and operational relevance

---

## Limitations and Next Steps

* The model is **predictive**, not diagnostic or prescriptive
* Root cause analysis is not inferred directly
* Temporal dynamics are not explicitly modeled

Potential extensions include:

* Sequential autoencoders (LSTM / GRU)
* Hybrid architectures combining anomaly detection and fault classification
* Prescriptive optimization based on controllable variables

---

## How to Run

1. Install dependencies:

   ```bash
   pip install numpy pandas scikit-learn matplotlib tensorflow
   ```
2. Open the notebook:

   ```bash
   jupyter notebook Engine_Fault_DB_Dataset_Autoencoder_for_Anomaly_Detection.ipynb
   ```
3. Run the notebook cells sequentially.

---

## Acknowledgements and References

This notebook was **inspired by** the open-source repository:

* **Autoencoders for Anomaly Detection**
  [https://github.com/amnahhebrahim/Autoencoders-for-Anomaly-Detection](https://github.com/amnahhebrahim/Autoencoders-for-Anomaly-Detection)

The referenced repository provided an initial conceptual baseline for applying autoencoders to anomaly detection problems. The current notebook **extends and adapts** those ideas through:

* a different dataset and industrial context,
* refined data preparation decisions,
* explicit reconstruction error analysis,
* percentile-based threshold selection,
* and decision-oriented evaluation metrics.

All modeling choices, evaluations, and interpretations presented here were **independently implemented** and tailored to the specific problem addressed in this project.

---

## Author Notes

This notebook is intended as a **technical proof of concept**, demonstrating sound anomaly detection practices rather than a fully deployed production system.
Design decisions prioritize **robustness, interpretability, and defensibility**, aligning with real-world industrial AI constraints.

---