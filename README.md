# Machine Learning Fundamentals: Regression & Classification Benchmarks

A hands-on repository exploring core supervised machine learning workflows in **Python** and **Scikit-Learn**, comparing linear and tree-based algorithms across continuous regression and medical binary classification tasks.

## Overview

Rather than treating algorithms as black boxes, this project focuses on reproducible benchmarking, proper preprocessing pipelines to prevent data leakage, cross-validation, and model interpretability:

1. **Regression (`diabetes_2.py`)**: Predicts continuous disease progression scores one year after baseline using the Scikit-Learn Diabetes dataset (442 samples, 10 physiological features).
2. **Classification (`cancer.py`)**: Classifies tumors as **Malignant** or **Benign** using the Wisconsin Breast Cancer dataset (569 samples, 30 cell nucleus measurements) with standardized preprocessing pipelines, 5-fold cross-validation, and feature importance extraction.

---

## Benchmark Results

### 1. Breast Cancer Classification (`cancer.py`)
All models were wrapped in a `StandardScaler` + estimator `Pipeline` and evaluated using an 80/20 train-test split (`random_state=42`) alongside **5-Fold Cross-Validation**:

| Model | Test Accuracy | 5-Fold CV Mean (± Std) | Malignant Recall | False Negatives (Missed Tumors) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **97.37%** | **0.98 ± 0.01** | **0.95** | **2 / 43** |
| **Random Forest** | 96.49% | 0.96 ± 0.02 | 0.93 | 3 / 43 |
| **Decision Tree** | 94.74% | 0.92 ± 0.02 | 0.93 | 3 / 43 |

### 2. Diabetes Progression Regression (`diabetes_2.py`)
Evaluated on an 80/20 train-test split (`random_state=42`) using $R^2$ Score and Mean Absolute Error (MAE):

| Model | $R^2$ Score | Mean Absolute Error (MAE) |
| :--- | :---: | :---: |
| **Linear Regression** | **0.453** | **42.79** |
| **Random Forest Regressor** | 0.443 | 44.05 |
| **Decision Tree Regressor** | 0.061 | 54.53 |

---

## Key Engineering Takeaways

* **Feature Scaling Matters for Linear Models, Not Trees:** Wrapping `StandardScaler` inside `make_pipeline()` eliminated convergence warnings and pushed `LogisticRegression` to **98% CV accuracy**, whereas `DecisionTree` and `RandomForest` remained invariant because threshold splits are scale-independent.
* **Single Trees Overfit Easily:** On the small Diabetes dataset, an unconstrained `DecisionTreeRegressor` scored just `0.061` $R^2$, while averaging 100 trees in a `RandomForestRegressor` jumped performance to `0.443` $R^2$.
* **Accuracy Alone Isn't Enough in Medicine:** Inspecting the confusion matrix and `classification_report` highlights **Recall** on the malignant class—ensuring malignant cases aren't misclassified as benign.
* **Model Interpretability:** Extracting `.feature_importances_` from the trained `RandomForestClassifier` revealed that **`worst area` (~15.5%)**, **`worst concave points` (~14.5%)**, and **`mean concave points` (~10.7%)** drive over **40%** of the ensemble's decisions.

*(Optional: Once you save your fixed Matplotlib chart to `assets/feature_importance.png`, uncomment the line below to display it directly in your README)*
<!-- ![Random Forest Feature Importance](assets/feature_importance.png) -->

---

## Tech Stack & Setup

* **Language:** Python 3.10+
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`

### Run Locally

```bash
# Clone the repository
git clone [https://github.com/redaisvibin11/Machine-Learning.git](https://github.com/redaisvibin11/Machine-Learning.git)
cd Machine-Learning

# Install dependencies
pip install scikit-learn pandas numpy matplotlib

# Run benchmarks
python diabetes_2.py
python cancer.py
