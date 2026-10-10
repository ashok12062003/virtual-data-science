
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE = Path(r"E:\New folder\Virtual-Data-Science")

DATA_DIR = BASE / "Week2" / "data" / "processed"
RESULTS_DIR = BASE / "Week3" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================

files = sorted(DATA_DIR.glob("*_week2.csv"))

if not files:
    files = sorted(DATA_DIR.glob("*.csv"))

if len(files) == 0:
    raise FileNotFoundError(
        f"No CSV dataset found in {DATA_DIR}"
    )

if len(files) > 1:
    raise ValueError(
        "Multiple CSV files found. Keep only the intended dataset "
        "in Week2/data/processed or select one explicitly."
    )

df = pd.read_csv(files[0])
df.columns = df.columns.astype(str).str.strip()

print("=" * 60)
print("WEEK 3 - DIMENSIONALITY REDUCTION USING PCA")
print("=" * 60)

print("Dataset:", files[0].name)
print("Original dataset shape:", df.shape)

# ============================================================
# 3. SELECT NUMERIC FEATURES
# ============================================================

numeric_df = df.select_dtypes(include=["number"]).copy()

# Remove constant columns.
numeric_df = numeric_df.loc[
    :, numeric_df.nunique(dropna=True) > 1
]

if numeric_df.shape[1] < 2:
    raise ValueError(
        "PCA requires at least two usable numeric features."
    )

# Replace infinite values with missing values.
numeric_df = numeric_df.replace(
    [np.inf, -np.inf], np.nan
)

# Remove rows where all selected features are missing.
numeric_df = numeric_df.dropna(how="all")

if len(numeric_df) < 2:
    raise ValueError(
        "At least two usable rows are required for PCA."
    )

print("\nSelected numeric features:")
print(numeric_df.columns.tolist())

# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

imputed_array = imputer.fit_transform(numeric_df)

# SimpleImputer may omit columns that are entirely missing.
if imputed_array.shape[1] < 2:
    raise ValueError(
        "Fewer than two usable numeric features remain after imputation."
    )

# ============================================================
# 5. STANDARDIZE THE FEATURES
# ============================================================

scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed_array)

print("\nShape after preprocessing:", scaled_data.shape)

# ============================================================
# 6. APPLY PCA
# ============================================================

pca = PCA(n_components=2)

reduced_data = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(
    reduced_data,
    columns=["Principal_Component_1", "Principal_Component_2"],
    index=numeric_df.index
)

print("\nReduced dataset shape:", pca_df.shape)

# ============================================================
# 7. EXPLAINED VARIANCE
# ============================================================

variance = pca.explained_variance_ratio_

print("\nExplained variance:")
print("Principal Component 1:", round(variance[0], 4))
print("Principal Component 2:", round(variance[1], 4))
print(
    "Total variance retained:",
    round(variance.sum() * 100, 2),
    "%"
)

variance_df = pd.DataFrame({
    "Principal_Component": [
        "Principal Component 1",
        "Principal Component 2"
    ],
    "Explained_Variance_Ratio": variance,
    "Explained_Variance_Percentage": variance * 100
})

variance_df.to_csv(
    RESULTS_DIR / "pca_explained_variance.csv",
    index=False
)

# ============================================================
# 8. SAVE REDUCED DATASET
# ============================================================

reduced_path = RESULTS_DIR / "pca_reduced_data.csv"

pca_df.to_csv(reduced_path, index=False)

print("\nReduced data saved to:")
print(reduced_path)

# ============================================================
# 9. VISUALIZE THE TWO PRINCIPAL COMPONENTS
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    pca_df["Principal_Component_1"],
    pca_df["Principal_Component_2"],
    s=35
)

plt.title("Dimensionality Reduction Using PCA")
plt.xlabel(
    f"Principal Component 1 ({variance[0] * 100:.2f}% variance)"
)
plt.ylabel(
    f"Principal Component 2 ({variance[1] * 100:.2f}% variance)"
)

plt.tight_layout()

chart_path = RESULTS_DIR / "pca_visualization.png"

plt.savefig(chart_path, dpi=300)
plt.close()

print("\nPCA visualization saved to:")
print(chart_path)

print("\nDimensionality reduction completed successfully.")
