
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE = Path(r"E:\New folder\Virtual-Data-Science")

RESULTS_DIR = BASE / "Week2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"

CHART_FILE = RESULTS_DIR / "model_comparison.png"
SUMMARY_FILE = RESULTS_DIR / "model_comparison_summary.txt"


# ============================================================
# 2. LOAD MODEL RESULTS
# ============================================================

if not COMPARISON_FILE.exists():
    raise FileNotFoundError(
        f"Comparison file not found: {COMPARISON_FILE}\n"
        "Run train_models.py first."
    )

df = pd.read_csv(COMPARISON_FILE)

if df.empty:
    raise ValueError("The model comparison file contains no results.")

required_columns = {"Model", "MAE", "RMSE", "R2_Score"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in comparison file: {sorted(missing_columns)}"
    )

# Sort models by lowest RMSE.
df = df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)


# ============================================================
# 3. DISPLAY MODEL COMPARISON
# ============================================================

print("=" * 70)
print("WEEK 2 - REGRESSION MODEL COMPARISON")
print("=" * 70)

print("\nModel performance:")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best_model = df.iloc[0]

print("\n" + "-" * 70)
print("BEST MODEL BASED ON TEST RMSE")
print("-" * 70)
print("Model:", best_model["Model"])
print(f"MAE:      {best_model['MAE']:.4f}")
print(f"RMSE:     {best_model['RMSE']:.4f}")
print(f"R2 Score: {best_model['R2_Score']:.4f}")


# ============================================================
# 4. SAVE A MODEL COMPARISON CHART
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(df["Model"], df["RMSE"])

plt.title("Regression Model Comparison - RMSE")
plt.xlabel("Machine Learning Model")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, df["RMSE"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=300)
plt.close()

print("\nComparison chart saved to:")
print(CHART_FILE)


# ============================================================
# 5. SAVE A TEXT SUMMARY
# ============================================================

with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
    file.write("WEEK 2 - SUPERVISED MACHINE LEARNING\n")
    file.write("REGRESSION MODEL COMPARISON\n")
    file.write("=" * 50 + "\n\n")

    file.write(df.to_string(index=False))
    file.write("\n\nBest model based on lowest test RMSE:\n")
    file.write(f"Model: {best_model['Model']}\n")
    file.write(f"MAE: {best_model['MAE']:.4f}\n")
    file.write(f"RMSE: {best_model['RMSE']:.4f}\n")
    file.write(f"R2 Score: {best_model['R2_Score']:.4f}\n")

print("Summary saved to:")
print(SUMMARY_FILE)


# ============================================================
# 6. OPTIONAL: CHECK SAVED PREDICTIONS
# ============================================================

if PREDICTIONS_FILE.exists():
    predictions = pd.read_csv(PREDICTIONS_FILE)

    print("\nPrediction file preview:")
    print(predictions.head().to_string(index=False))
else:
    print("\nPrediction file not found:", PREDICTIONS_FILE)

print("\nModel comparison completed successfully.")
