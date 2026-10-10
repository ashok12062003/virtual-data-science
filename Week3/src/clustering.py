
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE = Path(r"E:\New folder\Virtual-Data-Science")

DATA_DIR = BASE / "Week2" / "data" / "processed"
RESULTS_DIR = BASE / "Week3" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

files = sorted(DATA_DIR.glob("*_week2.csv"))

if not files:
    files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No CSV dataset found in {DATA_DIR}"
    )

if len(files) > 1:
    raise ValueError(
        "Multiple datasets found. Keep only the intended CSV "
        "in Week2/data/processed or select a file explicitly."
    )

df = pd.read_csv(files[0])
df.columns = df.columns.astype(str).str.strip()

print("Dataset:", files[0].name)
print("Original shape:", df.shape)

# ------------------------------------------------------------
# 3. PREPARE NUMERIC FEATURES
# ------------------------------------------------------------

# Use numeric columns only. Do not cluster using identifiers.
numeric_df = df.select_dtypes(include=["number"]).copy()

# Remove constant columns.
numeric_df = numeric_df.loc[
    :, numeric_df.nunique(dropna=True) > 1
]

if numeric_df.shape[1] == 0:
    raise ValueError(
        "No suitable numeric features found for clustering."
    )

# Convert infinite values to missing values.
numeric_df = numeric_df.replace(
    [np.inf, -np.inf], np.nan
)

# Remove rows where every selected feature is missing.
numeric_df = numeric_df.dropna(how="all")

# Impute remaining missing values.
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(numeric_df)

# Standardize features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if len(X_scaled) < 3:
    raise ValueError(
        "At least 3 usable rows are needed for this clustering analysis."
    )

print("Usable rows:", len(X_scaled))
print("Features:", numeric_df.columns.tolist())

# ------------------------------------------------------------
# 4. K-MEANS CLUSTERING
# ------------------------------------------------------------

n_clusters = min(3, len(X_scaled) - 1)

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

# ------------------------------------------------------------
# 5. HIERARCHICAL CLUSTERING
# ------------------------------------------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

# ------------------------------------------------------------
# 6. EVALUATE CLUSTERING
# ------------------------------------------------------------

def calculate_silhouette(data, labels):
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) >= len(data):
        return np.nan

    return silhouette_score(data, labels)


kmeans_silhouette = calculate_silhouette(
    X_scaled, kmeans_labels
)

hierarchical_silhouette = calculate_silhouette(
    X_scaled, hierarchical_labels
)

metrics = pd.DataFrame([
    {
        "Algorithm": "K-Means",
        "Number_of_Clusters": len(np.unique(kmeans_labels)),
        "Silhouette_Score": kmeans_silhouette,
        "Inertia": kmeans.inertia_
    },
    {
        "Algorithm": "Hierarchical Clustering",
        "Number_of_Clusters": len(np.unique(hierarchical_labels)),
        "Silhouette_Score": hierarchical_silhouette,
        "Inertia": np.nan
    }
])

metrics_path = RESULTS_DIR / "clustering_metrics.csv"
metrics.to_csv(metrics_path, index=False)

print("\nClustering evaluation:")
print(metrics.to_string(index=False))

# ------------------------------------------------------------
# 7. SAVE CLUSTER LABELS
# ------------------------------------------------------------

cluster_results = df.loc[numeric_df.index].copy()

cluster_results["KMeans_Cluster"] = kmeans_labels
cluster_results["Hierarchical_Cluster"] = hierarchical_labels

cluster_path = RESULTS_DIR / "clustering_results.csv"
cluster_results.to_csv(cluster_path, index=False)

print("\nSaved cluster assignments:", cluster_path)
print("Saved clustering metrics:", metrics_path)

# ------------------------------------------------------------
# 8. PCA VISUALIZATION
# ------------------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_results = pd.DataFrame({
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "KMeans_Cluster": kmeans_labels,
    "Hierarchical_Cluster": hierarchical_labels
})

pca_results.to_csv(
    RESULTS_DIR / "pca_coordinates.csv",
    index=False
)

# K-Means chart.
plt.figure(figsize=(9, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    s=35
)
plt.title("K-Means Clustering Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "kmeans_pca.png",
    dpi=300
)
plt.close()

# Hierarchical chart.
plt.figure(figsize=(9, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=hierarchical_labels,
    s=35
)
plt.title("Hierarchical Clustering Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "hierarchical_pca.png",
    dpi=300
)
plt.close()

print("\nPCA explained variance ratio:",
      pca.explained_variance_ratio_)

print("\nClustering completed successfully.")
