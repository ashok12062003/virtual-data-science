# Week 2: Supervised Machine Learning Models

## Project Overview

This project focuses on implementing supervised machine learning algorithms using Python and Scikit-learn. The objective is to understand regression models, train them using a prepared dataset, evaluate their performance, and compare their prediction results.

The project uses the `produce_cleaned_week2.csv` dataset from the Week 1 data preprocessing workflow.

## Objectives

- Understand supervised machine learning and regression.
- Load and clean the dataset using Pandas.
- Handle missing values and preprocess features.
- Split the dataset into training and testing sets.
- Implement and train multiple regression models.
- Evaluate model performance using standard regression metrics.
- Compare model results and save trained models and predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Visual Studio Code
- Git and GitHub

## Machine Learning Models

The following regression models are implemented:

1. **Linear Regression** – predicts continuous values using a linear relationship between features and the target.
2. **Decision Tree Regressor** – predicts values using a tree-based decision process.
3. **Random Forest Regressor** – combines multiple decision trees to produce predictions.
4. **K-Nearest Neighbors (KNN) Regressor** – predicts values using nearby training samples.

## Project Structure

```text
Week2/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   ├── model_comparison.png
│   ├── model_comparison_summary.txt
│   └── run_details.json
├── src/
│   ├── clean_dataset.py
│   ├── preprocess_dataset.py
│   ├── train_models.py
│   └── compare_models.py
├── report/
└── README.md
```

*The structure above describes the intended project layout; some files may be optional or not yet created.*

## Implementation Steps

1. Load the prepared dataset.
2. Remove duplicate records and handle missing values.
3. Select the target column and input features.
4. Divide the dataset into training and testing sets.
5. Apply imputation, feature scaling, and categorical encoding.
6. Train the regression models.
7. Calculate evaluation metrics.
8. Compare model performance.
9. Save the trained models, predictions, and comparison results.

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Measures the average absolute prediction error. |
| MSE | Measures the average squared prediction error. |
| RMSE | Measures prediction error in the target's units; lower is generally better. |
| R² Score | Measures how well the model explains variation in the target. |

The models are compared using the actual evaluation results generated during training. The model with the lowest test RMSE is identified as the best model by that metric, while R² and other metrics are also considered.

## How to Run the Project

### 1. Install the required libraries

```bash
python -m pip install pandas numpy scikit-learn matplotlib joblib
```

### 2. Navigate to the project directory

```bash
cd /d "E:\New folder\Virtual-Data-Science"
```

### 3. Run the training script

```bash
python Week2/src/train_models.py
```

On Windows Command Prompt, you can also use:

```bat
python Week2\src\train_models.py
```

### 4. Compare the models

```bat
python Week2\src\compare_models.py
```

## Output Files

- `models/` – stores trained machine learning pipelines.
- `results/model_comparison.csv` – stores evaluation metrics for each model.
- `results/test_predictions.csv` – stores actual and predicted test values.
- `results/model_comparison.png` – displays the model RMSE comparison chart.
- `results/model_comparison_summary.txt` – contains the comparison summary.
- `results/run_details.json` – records the training configuration and dataset details.

## Learning Outcomes

Through this project, I gained practical experience in supervised machine learning, data preprocessing, regression model development, model evaluation, and performance comparison. I also learned to organize project files and use GitHub for version control.

## Conclusion

This project demonstrates a supervised machine learning workflow using four regression algorithms. The evaluation results provide a basis for comparing the models and identifying areas for improvement. Further work can include cross-validation, hyperparameter tuning, feature selection, and verification that the selected target and features support meaningful predictions.

## Repository

GitHub: https://github.com/ashok12062003/virtual-data-science