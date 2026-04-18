# ============================================================
#   DIABETES RISK PREDICTION — MODEL TRAINING & EVALUATION
#   Algorithm: Logistic Regression
# ============================================================


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 1 — IMPORT LIBRARIES                           │
# └─────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

print(" All libraries imported successfully!")


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 2 — LOAD TRAIN/TEST DATA & SCALER              │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("  LOADING PREPROCESSED DATA")
print("="*60)

# Load the train/test splits from data_analysis.py
with open("data/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("data/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("data/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
with open("data/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Load the fitted scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"\n Data loaded successfully!")
print(f"   X_train shape : {X_train.shape}")
print(f"   X_test shape  : {X_test.shape}")
print(f"   y_train shape : {y_train.shape}")
print(f"   y_test shape  : {y_test.shape}")

# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 3 — BUILD LOGISTIC REGRESSION MODEL           │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("  BUILDING LOGISTIC REGRESSION MODEL")
print("="*60)

model = LogisticRegression(
    max_iter=1000,        # Max iterations to converge
    random_state=42,      # Reproducibility
    solver="lbfgs",       # Efficient for small datasets
    class_weight=None     # Can use 'balanced' if class imbalance issues
)

print("\n Model Configuration:")
print(f"   Algorithm      : Logistic Regression")
print(f"   Solver         : lbfgs")
print(f"   Max Iterations : 1000")
print(f"   Random State   : 42 (reproducible)")


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 4 — TRAIN THE MODEL                           │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("  TRAINING MODEL ON TRAINING DATA")
print("="*60)

model.fit(X_train, y_train)

print(f"\n Model trained successfully!")
print(f"   Bias (Intercept)    : {model.intercept_[0]:.4f}")
print(f"\n Feature Coefficients (weights):")
print(f"   (Positive = increases risk, Negative = decreases risk)")

coef_df = pd.DataFrame({
    "Feature"     : X_train.columns,
    "Coefficient" : model.coef_[0]
}).sort_values("Coefficient", ascending=False)

print(coef_df.to_string(index=False))