# Handling Missing Values Using Multivariate Imputation

This repository contains my practical work on **handling missing values using multivariate imputation techniques in Machine Learning**.

Missing values are common in real-world datasets. Simply removing rows or filling missing values with the mean, median, or mode is not always the best approach. Multivariate imputation provides another way to estimate missing values by considering information from other features in the dataset.

## What is Multivariate Imputation?

Multivariate imputation is a technique where missing values in one feature are estimated using the values of other features.

For example, suppose a dataset contains:

```text
Age    Income    Experience
25     40000     2
30     NaN       5
35     60000     NaN
```

Instead of looking only at the `Income` column, a multivariate method can use other available features such as `Age` and `Experience` to estimate the missing value.

This can provide more meaningful estimates when features have relationships with each other.

## Techniques Practiced

In this repository, I practiced different multivariate imputation approaches, including:

- Iterative Imputation
- KNN Imputation
- Feature-based imputation
- Comparison of different imputation methods

---

## 1. Iterative Imputation

**Iterative Imputer** estimates missing values by modeling each feature with missing values as a function of the other features.

For example:

```text
Feature A ─┐
Feature B ─┼──> Model ──> Missing Value
Feature C ─┘
```

The process is repeated iteratively to improve the estimates.

Example using Scikit-learn:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()

X_imputed = imputer.fit_transform(X)
```

---

## 2. KNN Imputation

**KNN Imputer** uses similar observations to estimate missing values.

KNN stands for **K-Nearest Neighbors**.

The basic idea is:

```text
Find similar rows
       ↓
Look at their available values
       ↓
Use those values to estimate the missing value
```

Example:

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

X_imputed = imputer.fit_transform(X)
```

The `n_neighbors` parameter controls how many nearby observations are considered.

---

## Why Use Multivariate Imputation?

Multivariate imputation can be useful when:

- Features have relationships with each other
- A dataset contains several missing values
- We want to use information from multiple features
- Simple mean or median imputation may lose useful information
- We want a more data-driven approach to estimating missing values

However, the best imputation technique depends on the dataset and the relationships between its features.

## Important Practice

When applying imputation in Machine Learning, the imputer should generally be **fitted only on the training data**.

```text
Training Data
     ↓
Fit Imputer
     ↓
Transform Training Data
     ↓
Transform Test Data
```

This helps prevent information from the test set from influencing the preprocessing process.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

## Learning Goal

The goal of this repository is to develop a practical understanding of **multivariate missing-value imputation** and learn how techniques such as **Iterative Imputer and KNN Imputer** can be used to estimate missing values using information from other features.

This practice is part of my ongoing **Machine Learning and AI learning journey**.
