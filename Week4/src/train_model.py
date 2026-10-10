
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Week2" / "data" / "processed" / "produce_cleaned_week2.csv"
MODEL_DIR = ROOT / "Week4" / "models"
RESULTS_DIR = ROOT / "Week4" / "results"

FEATURES = ["Particulars", "Frequency", "Unit"]
TARGET = "3-2014"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    X = df[FEATURES].copy()
    y = df[TARGET]

    for col in FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessing = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), FEATURES)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions)
    }

    joblib.dump(model, MODEL_DIR / "produce_model.joblib")

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "evaluation_metrics.csv", index=False
    )

    pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions
    }).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    print("Model training completed successfully.")
    print("Model saved:", MODEL_DIR / "produce_model.joblib")
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
