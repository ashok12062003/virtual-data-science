
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

BASE = Path(r"E:\New folder\Virtual-Data-Science")

DATA_DIR = BASE / "Week2" / "data" / "processed"
MODELS_DIR = BASE / "Week2" / "models"
RESULTS_DIR = BASE / "Week2" / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = " 3-2014"
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

dataset_files = sorted(DATA_DIR.glob("*_week2.csv"))

if not dataset_files:
    dataset_files = sorted(DATA_DIR.glob("*.csv"))

if len(dataset_files) == 0:
    raise FileNotFoundError(
        f"No CSV dataset found in {DATA_DIR}. "
        "Run clean_dataset.py first."
    )

if len(dataset_files) > 1:
    raise ValueError(
        "More than one CSV file was found:\n"
        + "\n".join(file.name for file in dataset_files)
        + "\nKeep only the intended dataset in the folder."
    )

dataset_path = dataset_files[0]
df = pd.read_csv(dataset_path)

# Remove accidental spaces from column names.
df.columns = df.columns.astype(str).str.strip()

# Match the target after trimming column-name spaces.
TARGET_COLUMN = TARGET_COLUMN.strip()

print("=" * 60)
print("WEEK 2 - SUPERVISED MACHINE LEARNING")
print("=" * 60)
print("Dataset:", dataset_path.name)
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column {TARGET_COLUMN!r} not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# 3. CLEAN DATA
# ============================================================

df = df.dropna(how="all")
df = df.dropna(axis=1, how="all")
df = df.drop_duplicates()

# Convert the target to numeric values.
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN], errors="coerce"
)

# Remove rows where the target is missing or non-numeric.
df = df.dropna(subset=[TARGET_COLUMN]).copy()

if df.empty:
    raise ValueError(
        f"No numeric target values found in {TARGET_COLUMN!r}."
    )

if df[TARGET_COLUMN].nunique() < 2:
    raise ValueError(
        "The target contains fewer than two different values. "
        "Choose another numeric target."
    )

print("\nCleaned shape:", df.shape)
print("Target column:", TARGET_COLUMN)
print("\nTarget statistics:")
print(df[TARGET_COLUMN].describe())


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

# Convert year columns to numeric when possible.
for column in df.columns:
    if column != TARGET_COLUMN:
        converted = pd.to_numeric(df[column], errors="coerce")

        # Convert only when at least some values are numeric.
        if converted.notna().sum() > 0:
            df[column] = converted

X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].copy()

# Remove columns containing only one unique value.
constant_columns = [
    column for column in X.columns
    if X[column].nunique(dropna=True) <= 1
]

if constant_columns:
    X = X.drop(columns=constant_columns)
    print("\nRemoved constant columns:", constant_columns)

# Drop rows where all feature values are missing.
valid_rows = X.notna().any(axis=1)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

if X.shape[1] == 0:
    raise ValueError("No usable feature columns remain.")

if len(X) < 10:
    raise ValueError(
        "Too few usable rows to perform a reliable train/test split."
    )

# Convert remaining non-numeric features to categorical strings.
for column in X.columns:
    if not pd.api.types.is_numeric_dtype(X[column]):
        X[column] = X[column].astype("string")

print("\nFeature count:", X.shape[1])
print("Usable rows:", len(X))


# ============================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 6. PREPROCESS FEATURES
# ============================================================

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

transformers = []

if numeric_columns:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    transformers.append(
        ("numeric", numeric_pipeline, numeric_columns)
    )

if categorical_columns:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    transformers.append(
        ("categorical", categorical_pipeline, categorical_columns)
    )

if not transformers:
    raise ValueError("No usable numeric or categorical features found.")

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# 7. DEFINE REGRESSION MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=min(5, len(X_train))
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

predictions_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True)
})

for model_name, estimator in models.items():

    print(f"\nTraining {model_name}...")

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2
    })

    predictions_df[model_name] = predictions

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_path = MODELS_DIR / f"{safe_name}.joblib"
    joblib.dump(model_pipeline, model_path)

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 score:", round(r2, 4))
    print("Saved model:", model_path)


# ============================================================
# 9. COMPARE MODELS AND SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

# Lower RMSE indicates a smaller prediction error.
results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

comparison_path = RESULTS_DIR / "model_comparison.csv"
predictions_path = RESULTS_DIR / "test_predictions.csv"
details_path = RESULTS_DIR / "run_details.json"

results_df.to_csv(comparison_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

details = {
    "dataset": dataset_path.name,
    "task": "regression",
    "target_column": TARGET_COLUMN,
    "feature_columns": X.columns.tolist(),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "models": list(models.keys())
}

with open(details_path, "w", encoding="utf-8") as file:
    json.dump(details, file, indent=4)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

print("\nBest model by test RMSE:", results_df.iloc[0]["Model"])
print("\nSaved comparison:", comparison_path)
print("Saved predictions:", predictions_path)
print("Saved model files:", MODELS_DIR)
print("\nTraining and evaluation completed successfully.")
