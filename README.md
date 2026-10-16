# Linear Regression from Scratch

Predict house prices using the **normal equation** and **vectorized gradient descent**, built with **only NumPy** and no ML library doing the work. The result is verified against `scikit-learn` at the end.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Problem

Calling `LinearRegression().fit()` hides what actually happens. This project implements linear regression from first principles to show how matrix operations, cost functions and optimization produce a trained model.

The data is generated from a **known formula**, so the model can be checked on whether it recovers the true weights:

```
price (lakhs) = 20 + 0.05 * area + 8 * bedrooms - 0.6 * age + noise
```

In matrix form the model is `y_hat = X @ w`, with a column of ones in `X` for the bias.

## What I built

| Step | What it does | Concepts |
|---|---|---|
| 1 | Seeded housing data (500 houses) from a known formula | `np.random`, `np.column_stack` |
| 2 | Train/test split, RMSE, R², standardization and bias helpers | permutation indexing, train-only statistics |
| 3 | **Normal equation** with an invertibility check (determinant, condition number) | `np.linalg.inv`, `np.linalg.solve`, `.T`, `@` |
| 4 | **Gradient descent** with loss curve and learning-rate comparison | vectorized gradients, feature scaling |
| 5 | Evaluation with predicted-vs-actual and residual plots | RMSE, R², residual analysis |
| 6 | Verification against `scikit-learn` | `LinearRegression`, `np.allclose` |
| 7 | Same code run on a real dataset (California housing, with a bundled fallback) | generalizing the pipeline |

## The math

**Cost function**

```
J(w) = (1 / 2m) * Σ (Xw - y)²
```

**Normal equation** (closed form)

```
w = (XᵀX)⁻¹ Xᵀ y
```

**Gradient descent** (iterative)

```
w ← w - α * (1/m) * Xᵀ(Xw - y)
```

## Key learnings

- **Gradient descent needs feature scaling.** `area` is in the thousands while `bedrooms` is 1-5, which makes the cost surface a long narrow valley. Standardizing the features fixes this. The normal equation doesn't need it.
- **Learning rate matters.** Too small is slow, and too large diverges (see the learning-rate plot).
- **No test-set leakage.** Scaling uses the training mean and std only.
- **`solve` beats `inv`.** `np.linalg.solve` is numerically safer than computing the inverse explicitly. The condition number of `XᵀX` (about 5e7 here) shows why this matters.
- **Know your baseline.** A model must beat the mean-prediction baseline, and RMSE near the injected noise level (10) means the model can't do better on this data.
- **Normal equation vs gradient descent.** The closed form costs roughly O(n³) in the number of features, while gradient descent scales to many more features.

## Results

| Parameter | True | Normal Equation | Gradient Descent | scikit-learn |
|---|---:|---:|---:|---:|
| bias | 20.000 | 21.426 | 21.426 | 21.426 |
| area | 0.050 | 0.049 | 0.049 | 0.049 |
| bedrooms | 8.000 | 7.320 | 7.320 | 7.320 |
| age | -0.600 | -0.540 | -0.540 | -0.540 |

| Method | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---:|---:|---:|---:|
| Normal Equation | 10.237 | 9.914 | 0.9329 | 0.9462 |
| Gradient Descent | 10.237 | 9.914 | 0.9329 | 0.9462 |

Both solvers match each other and `scikit-learn`. The small gaps from the true weights come from the injected noise.

## Output

**Gradient descent loss curve** (cost falls from about 7,814 to about 52)

<img width="923" height="586" alt="01_loss_curve" src="https://github.com/user-attachments/assets/2a962d9c-a772-494c-bb2a-ee890ef33629" />


**Learning rate comparison**

<img width="923" height="586" alt="02_learning_rate_comparison" src="https://github.com/user-attachments/assets/4a7c6621-2a37-4b9f-9d5f-f0c3d23ebea3" />


**Evaluation: predicted vs actual, residuals, residual distribution**

<img width="2384" height="656" alt="03_evaluation_plots" src="https://github.com/user-attachments/assets/81fee976-f744-44c2-86de-6aec3fc66978" />


## How to run

Option 1: Google Colab (no setup). Click the Open In Colab badge at the top and run all cells (Runtime > Run all).

Option 2: locally

Requirements: Python 3.8+, NumPy, Matplotlib, scikit-learn.
```bash
git clone https://github.com/jayy0101/linear_regression_from_scratch.git
cd linear_regression_from_scratch
pip install -r requirements.txt
jupyter notebook linear_regression_from_scratch.ipynb
```

Run all cells from top to bottom. A fixed random seed makes the results reproducible. The real-dataset step needs internet on first run; without it the notebook falls back to a dataset bundled with scikit-learn.

Requirements: Python 3.8+, NumPy, Matplotlib, scikit-learn.

Limitations
- The main results use synthetic data with a perfectly linear relationship. Real data is rarely this clean, which is why the notebook also runs the same pipeline on a real dataset.
- Linear regression can only model linear relationships between features and the target.
- There is no regularization, so it can overfit when there are many features or little data.

## Ideas for extension

- Polynomial features
- L2 regularization (Ridge regression)
- Mini-batch and stochastic gradient descent

## Author

Jaydev Singh | [GitHub](https://github.com/jayy0101)
