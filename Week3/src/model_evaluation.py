
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

BASE = Path(r"E:\New folder\Virtual-Data-Science")
DATA_DIR = BASE / "Week2" / "data" / "processed"
RESULTS_DIR = BASE / "Week3" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "3-2014"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ============================================================
# 2. LOAD DATASET
# ============================================================

files = sorted(DATA_DIR.glob("*_week2.csv"))

if not files:
    files = sorted(DATA_DIR.glob("*.csv"))

if len(files) != 1:
    raise ValueError(
        f"Expected exactly one CSV in {DATA_DIR}, "
        f"but found {len(files)}."
    )

df = pd.read_csv(files[0])
df.columns = df.columns.astype(str).str.strip()

print("=" * 65)
print("WEEK 3 - MODEL EVALUATION")
print("=" * 65)
print("Dataset:", files[0].name)
print("Dataset shape:", df.shape)

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column {TARGET_COLUMN!r} not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# ============================================================
# 3. CREATE DEMONSTRATION CLASSES
# ============================================================
# These classes are created only to demonstrate classification
# evaluation. They are not genuine pre-existing dataset labels.

df = df.drop_duplicates().copy()

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN], errors="coerce"
)
df = df.dropna(subset=[TARGET_COLUMN]).copy()

if df[TARGET_COLUMN].nunique() < 3:
    raise ValueError(
        "At least three distinct numeric target values are required."
    )

# Rank-based qcut avoids duplicate quantile boundary errors.
ranked_values = df[TARGET_COLUMN].rank(method="first")

df["Demo_Class"] = pd.qcut(
    ranked_values,
    q=3,
    labels=["Low", "Medium", "High"]
)

print("\nDemonstration class distribution:")
print(df["Demo_Class"].value_counts())

# Exclude all year columns, including the target source, to
# prevent the target value from appearing directly in the features.
year_columns = [
    col for col in df.columns if col.startswith("3-")
]

X = df.drop(
    columns=year_columns + ["Demo_Class"],
    errors="ignore"
).copy()

y = df["Demo_Class"].astype(str)

X = X.dropna(axis=1, how="all")

constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]
X = X.drop(columns=constant_columns)

if X.shape[1] == 0:
    raise ValueError(
        "No usable features remain. Use a labeled classification dataset."
    )

for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].astype("string")

print("\nFeatures used:", X.columns.tolist())

label_encoder = LabelEncoder()
y = pd.Series(
    label_encoder.fit_transform(y),
    index=y.index
)

if y.value_counts().min() < 2:
    raise ValueError("Each class needs at least two samples.")

# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

number_of_classes = y.nunique()
test_rows = int(np.ceil(len(y) * TEST_SIZE))
train_rows = len(y) - test_rows

if (
    test_rows < number_of_classes
    or train_rows < number_of_classes
):
    raise ValueError(
        "Not enough samples for a stratified train/test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

transformers = []

if numeric_columns:
    transformers.append((
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric_columns
    ))

if categorical_columns:
    transformers.append((
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_columns
    ))

preprocessor = ColumnTransformer(
    transformers=transformers
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE))
])

# ============================================================
# 6. CROSS-VALIDATION
# ============================================================

n_splits = min(5, int(y_train.value_counts().min()))

if n_splits < 2:
    raise ValueError(
        "Insufficient examples per class for cross-validation."
    )

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    },
    n_jobs=-1
)

print("\nCROSS-VALIDATION RESULTS")

for metric in [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]:
    scores = cv_results[f"test_{metric}"]
    print(
        f"{metric}: {scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )

# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

parameter_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print("\nBEST PARAMETERS:", search.best_params_)
print("Best CV F1-score:", round(search.best_score_, 4))

# ============================================================
# 8. TEST SET METRICS
# ============================================================

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\nTEST SET METRICS")
print("Accuracy:", round(accuracy, 4))
print("Precision (macro):", round(precision, 4))
print("Recall (macro):", round(recall, 4))
print("F1-score (macro):", round(f1, 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm_display = ConfusionMatrixDisplay.from_predictions(
    label_encoder.inverse_transform(y_test.to_numpy()),
    label_encoder.inverse_transform(y_pred),
    labels=label_encoder.classes_,
    display_labels=label_encoder.classes_,
    xticks_rotation="vertical"
)

plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)
plt.close()

# ============================================================
# 10. ROC-AUC
# ============================================================

try:
    if len(label_encoder.classes_) == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(
            y_test,
            y_prob,
            labels=np.arange(len(label_encoder.classes_)),
            multi_class="ovr",
            average="macro"
        )

    print("ROC-AUC:", round(auc, 4))

except ValueError as error:
    auc = np.nan
    print("ROC-AUC unavailable:", error)

# ============================================================
# 11. SAVE RESULTS
# ============================================================

summary = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "Accuracy": accuracy,
    "Precision_Macro": precision,
    "Recall_Macro": recall,
    "F1_Macro": f1,
    "ROC_AUC": auc,
    "Cross_Validation_F1_Mean":
        cv_results["test_f1_macro"].mean(),
    "Best_CV_F1": search.best_score_
}])

summary.to_csv(
    RESULTS_DIR / "classification_evaluation.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "Actual_Label": label_encoder.inverse_transform(
        y_test.to_numpy()
    ),
    "Predicted_Label": label_encoder.inverse_transform(y_pred)
})

predictions_df.to_csv(
    RESULTS_DIR / "classification_predictions.csv",
    index=False
)

joblib.dump(
    {
        "model": best_model,
        "label_encoder": label_encoder,
        "target_column": TARGET_COLUMN,
        "target_type": "demonstration_only"
    },
    RESULTS_DIR / "best_classifier.joblib"
)

print("\nResults saved in:", RESULTS_DIR)
print(" - classification_evaluation.csv")
print(" - classification_predictions.csv")
print(" - confusion_matrix.png")
print(" - best_classifier.joblib")
print("\nModel evaluation completed.")
