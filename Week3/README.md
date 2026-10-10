# Week 3: Unsupervised Learning & Model Evaluation

## Project Overview

This project focuses on applying unsupervised learning techniques and evaluating machine learning models using Python and Scikit-learn. It includes clustering, dimensionality reduction, cross-validation, classification metrics, and hyperparameter tuning.

The project uses the processed Week 2 produce dataset to explore patterns, reduce data dimensionality, and demonstrate machine learning model evaluation techniques.

## Objectives

- Implement K-Means clustering.
- Implement Hierarchical (Agglomerative) Clustering.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Evaluate clustering quality using the Silhouette Score and inertia.
- Perform cross-validation for model evaluation.
- Generate a confusion matrix and calculate accuracy, precision, recall, F1-score, and ROC-AUC where applicable.
- Apply GridSearchCV for hyperparameter tuning.
- Generate visualizations and evaluation reports.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Project Structure

```text
Week3/
├── data/
│   └── processed/
├── src/
│   ├── clustering.py
│   ├── dimensionality_reduction.py
│   └── model_evaluation.py
├── results/
├── screenshots/
├── report/
├── requirements.txt
└── README.md
```

## Project Modules

### 1. Clustering

K-Means and Hierarchical Clustering are applied to identify groups and patterns in the numerical features. The Silhouette Score and K-Means inertia are used to assess clustering results.

### 2. Dimensionality Reduction

Principal Component Analysis (PCA) reduces the numerical feature space to two principal components for visualization and exploration.

### 3. Model Evaluation

A Random Forest Classifier is evaluated using cross-validation, a confusion matrix, classification metrics, and GridSearchCV for hyperparameter tuning.

**Important note:** The classification labels Low, Medium, and High are derived from the `3-2014` target values for demonstration purposes. They are not original dataset labels. Therefore, the classification results demonstrate evaluation techniques rather than validated real-world predictive performance.

## How to Run

### Step 1: Install dependencies

```bash
pip install -r Week3/requirements.txt
```

### Step 2: Run clustering

```bash
python Week3/src/clustering.py
```

### Step 3: Run dimensionality reduction

```bash
python Week3/src/dimensionality_reduction.py
```

### Step 4: Run model evaluation

```bash
python Week3/src/model_evaluation.py
```

Run the commands from the repository root directory. Ensure the processed Week 2 dataset is available at the expected path.

## Expected Outputs

The `Week3/results/` directory contains outputs such as:

- Clustering metrics and cluster assignments
- K-Means and Hierarchical Clustering visualizations
- PCA-reduced data and explained variance
- Classification evaluation metrics
- Classification predictions
- Confusion matrix visualization
- Saved best classifier model

Actual output files depend on successful execution of the scripts.

## Learning Outcomes

- Understanding unsupervised learning algorithms.
- Exploring patterns in numerical datasets.
- Applying PCA for dimensionality reduction.
- Evaluating machine learning models with multiple metrics.
- Performing cross-validation and hyperparameter tuning.
- Creating visualizations and documenting experimental results.

## Conclusion

Week 3 demonstrates clustering, dimensionality reduction, and machine learning model evaluation using Python. The project provides practical experience with Scikit-learn workflows and emphasizes interpreting model results carefully, especially when demonstration labels are generated from numerical target values.

## Author

