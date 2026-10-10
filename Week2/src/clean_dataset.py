
from pathlib import Path
import pandas as pd

# Project paths
BASE = Path(r"E:\New folder\Virtual-Data-Science")
WEEK1_DATA = BASE / "Week1" / "data" / "processed"
WEEK2_OUTPUT = BASE / "Week2" / "data" / "processed"

WEEK2_OUTPUT.mkdir(parents=True, exist_ok=True)

# Find the Week 1 cleaned dataset
files = list(WEEK1_DATA.glob("*_cleaned.csv"))

if not files:
    raise FileNotFoundError(
        "No Week 1 cleaned CSV found in Week1/data/processed."
    )

if len(files) > 1:
    raise ValueError(
        "Multiple cleaned CSV files found. Select the correct dataset."
    )

input_file = files[0]
df = pd.read_csv(input_file)

print("Using Week 1 dataset:", input_file.name)
print("Original shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Remove fully empty rows and columns
df = df.dropna(how="all").dropna(axis=1, how="all").copy()

# Remove duplicate rows
duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# Clean whitespace in text columns
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].map(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    df[col] = df[col].replace("", pd.NA)

# Save Week 2 cleaned dataset
output_file = WEEK2_OUTPUT / f"{input_file.stem}_week2.csv"
df.to_csv(output_file, index=False)

print("\nDuplicates removed:", duplicates)
print("Final shape:", df.shape)
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nFirst five rows:\n", df.head())
print("\nSaved to:", output_file)
