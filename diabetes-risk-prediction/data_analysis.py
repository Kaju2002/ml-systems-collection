# ============================================================
#   DIABETES RISK PREDICTION — DATA ANALYSIS & PREPROCESSING
#   Purpose: Load, explore, clean, and prepare dataset
# ============================================================

# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 1 — IMPORT LIBRARIES                           │
# └─────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("All libraries imported successfully!")


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 2 — LOAD DATASET                              │
# └─────────────────────────────────────────────────────────┘

df = pd.read_csv("diabetes.csv")

print("\n" + "="*60)
print(" DATASET LOADED SUCCESSFULLY")
print("="*60)
print(f"\n   Total Rows    : {df.shape[0]}")
print(f"   Total Columns : {df.shape[1]}")
print(f"\n   Dataset Size  : {df.shape[0]} patients × {df.shape[1]} measurements")

print("\n Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

print("\n First 5 rows of data:")
print(df.head())


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)           │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# --- Step 3.1: Basic Statistics ---
print("\nSummary Statistics (Min, Max, Mean, Median):")
print(df.describe())

# --- Step 3.2: Check for Missing Values ---
print("\nMissing Values (NaN count per column):")
missing = df.isnull().sum()
print(missing)

if missing.sum() == 0:
    print("   No missing values found (good!)")
else:
    print(f"   {missing.sum()} missing values found")

# --- Step 3.3: Target Variable Distribution ---
print("\n TARGET VARIABLE DISTRIBUTION (Outcome):")
target_counts = df["Outcome"].value_counts()
print(f"   0 = Non-Diabetic : {target_counts[0]} patients ({target_counts[0]/len(df)*100:.1f}%)")
print(f"   1 = Diabetic     : {target_counts[1]} patients ({target_counts[1]/len(df)*100:.1f}%)")

class_ratio = target_counts[1] / target_counts[0]
print(f"   Class Ratio      : 1:{1/class_ratio:.1f} (imbalanced, but acceptable)")

# --- Step 3.4: Check for Zero Values (THE KEY INSIGHT!) ---
print("\n  CHECKING FOR IMPOSSIBLE ZERO VALUES:")
print("   (Zeros in medical columns = missing data, NOT literal zeros)")

zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_columns:
    zero_count = (df[col] == 0).sum()
    zero_pct = (zero_count / len(df)) * 100
    print(f"   {col:>16} → {zero_count:>3} zeros ({zero_pct:>5.1f}%) [IMPOSSIBLE!]")



