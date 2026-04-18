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
