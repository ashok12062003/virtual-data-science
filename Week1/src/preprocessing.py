
from pathlib import Path
import pandas as pd

# 1. Define folders
ARCHIVE = Path(r"E:\New folder\archive")
OUTPUT = Path(
    r"E:\New folder\Virtual-Data-Science\Week1\data\processed"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

# 2. Find CSV files
csv_files = list(ARCHIVE.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in archive folder.")

# 3. Select the largest CSV file
input_file = max(csv_files, key=lambda f: f.stat().st_size)
print("Selected dataset:", input_file.name)

# 4. Load the dataset
df = pd.read_csv(input_file)
print("\nOriginal shape:", df.shape)

# 5. Inspect missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 6. Remove duplicate rows
duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()
print("\nDuplicate rows removed:", duplicates)

# 7. Handle missing values
for column in df.columns:
    if df[column].isnull().any():
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])
            else:
                df[column] = df[column].fillna("Unknown")

# 8. Save cleaned dataset
output_file = OUTPUT / f"{input_file.stem}_cleaned.csv"
df.to_csv(output_file, index=False)

# 9. Verify the result
print("\nFinal shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nCleaned dataset saved to:", output_file)
print("\nFirst five cleaned rows:")
print(df.head())
