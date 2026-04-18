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

