# Dimensionality Reduction: PCA & LDA

A learning-focused project that explains and implements **Dimensionality Reduction techniques** using both **theoretical notebooks** and **Python utility functions**.

This repository aims to build a **strong conceptual and mathematical understanding** of two widely used dimensionality reduction techniques:

* **Principal Component Analysis (PCA)**
* **Linear Discriminant Analysis (LDA)**

The project combines:

* theoretical explanations
* visualization notebooks
* low-level mathematical implementations

instead of relying solely on high-level libraries like `sklearn`.

---

# Project Goals

This project helps learners understand:

* Why dimensionality reduction is important
* The **curse of dimensionality**
* Linear dependence and independence
* Variance and covariance relationships
* Eigenvectors and eigenvalues intuition
* Mathematical steps behind **PCA**
* Mathematical steps behind **LDA**

The implementation focuses on **clarity of concepts rather than production optimization**.

---

# Repository Structure

```text
Dimensionality-Reduction-PCA-LDA/
│
├── notebooks/
│   ├── notes.ipynb
│   ├── pca_visualizations.ipynb
│   └── lda_visualizations.ipynb
│
├── utils/
│   └── dimensionality_reduction_utils.py
│
└── README.md
```

### Notebooks

The notebooks cover the theoretical flow:

1. Dimensionality Reduction
2. Curse of Dimensionality
3. Linear Dependence and Independence
4. Variance and Covariance
5. Covariance Matrix
6. Eigenvectors and Eigenvalues
7. Principal Component Analysis (PCA)
8. Explained Variance
9. Linear Discriminant Analysis (LDA)
10. PCA vs LDA

Separate notebooks demonstrate **visual examples and projections**.

---

# Utility Functions

The `dimensionality_reduction_utils.py` file
contains Python functions implementing core mathematical operations required for PCA and LDA.
These functions help illustrate how dimensionality reduction algorithms work internally.

---

# Implemented Functions

## Standard Deviation Calculation

```python
calculate_standard_deviation(X)
```

Computes the standard deviation of each feature.
This step is commonly used during **data standardization** before applying PCA.

---

## Class Mean Calculation

```python
calculate_class_means(X, y)
```

Computes the **mean vector for each class**.
Used in LDA for computing class statistics.

---

## Covariance Matrix Calculation

```python
calculate_covariance_matrix(X)
```

Computes the covariance matrix of the dataset.
The covariance matrix captures how features vary with respect to each other.
This matrix is the **foundation for PCA**.

---

## Class Covariance Matrices

```python
compute_class_covariances(X, y)
```

Computes covariance matrices for each class.
These are used to construct the **within-class scatter matrix** in LDA.

---

## Eigenvalue and Eigenvector Decomposition

```python
calculate_eigen_decompostion(dat_matrix)
```

Computes eigenvalues and eigenvectors of a matrix.

Eigen decomposition is required in:

* PCA → principal component directions
* LDA → discriminant directions

The function also sorts eigenvalues in **descending order** to identify the most important components.

---

# PCA Workflow

The conceptual workflow of PCA is:

```
Dataset
   │
   ▼
Standardization
   │
   ▼
Covariance Matrix
   │
   ▼
Eigenvalue Decomposition
   │
   ▼
Select Top k Components
   │
   ▼
Project Data onto New Feature Space
```

Goal:

Maximize **variance preservation** while reducing dimensionality.

---

# LDA Workflow

The conceptual workflow of LDA is:

```
Dataset + Class Labels
        │
        ▼
Compute Class Means
        │
        ▼
Compute Within-Class Scatter Matrix
        │
        ▼
Compute Between-Class Scatter Matrix
        │
        ▼
Solve Eigenvalue Problem
        │
        ▼
Select Discriminant Components
        │
        ▼
Project Data
```

Goal:

Maximize **class separability**.

---

# PCA vs LDA

| Feature     | PCA                  | LDA                       |
| ----------- | -------------------- | ------------------------- |
| Type        | Unsupervised         | Supervised                |
| Uses Labels | No                   | Yes                       |
| Objective   | Maximize variance    | Maximize class separation |
| Components  | ≤ number of features | ≤ classes − 1             |

---

# Example Usage

```python
import numpy as np
from dimensionality_reduction_utils import calculate_covariance_matrix

X = np.random.rand(100,5)
cov_matrix = calculate_covariance_matrix(X)
print(cov_matrix)
```

---

# Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

---

### TODO Dataset Experiments 

Apply PCA and LDA to:

* Iris dataset
* MNIST dataset
* Wine dataset

---

### Performance Improvements

* Vectorized implementations
* Optimized matrix computations
* Benchmark comparisons

---

# License

This project is intended for **educational and learning purposes**.

---