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
    max_iter=1000,
    random_state=42,
    solver="lbfgs",
    class_weight="balanced"  
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



# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 5 — MAKE PREDICTIONS ON TEST SET              │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("  GENERATING PREDICTIONS ON TEST SET")
print("="*60)

y_pred = model.predict(X_test)                              # Class labels (0 or 1)
y_pred_proba = model.predict_proba(X_test)[:, 1]           # Probability of class 1

print(f"\n✓ Predictions generated!")
print(f"   Sample predictions (first 10):")
for i in range(10):
    print(f"      Test sample {i+1} → Predicted: {y_pred[i]}, Probability: {y_pred_proba[i]:.2%}")


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 5B — ADJUSTED PREDICTIONS WITH LOWER THRESHOLD│
# └─────────────────────────────────────────────────────────┘

# Default: 0.5 threshold
# Custom: 0.4 threshold (predict diabetic if probability > 0.4)

custom_threshold = 0.4
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

print("\n" + "="*60)
print("  THRESHOLD ADJUSTMENT TEST")
print("="*60)

from sklearn.metrics import confusion_matrix as cm_func

cm_custom = cm_func(y_test, y_pred_custom)
tn_c, fp_c, fn_c, tp_c = cm_custom.ravel()

print(f"\nWith Threshold = 0.4:")
print(f"   True Positives (TP)   : {tp_c}  (caught diabetic)")
print(f"   False Negatives (FN)  : {fn_c}  (missed diabetic) ↓ LOWER")
print(f"   False Positives (FP)  : {fp_c}  (false alarm) ↑ HIGHER")

accuracy_custom = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c)
print(f"   Accuracy : {accuracy_custom * 100:.2f}%")


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 6 — CALCULATE EVALUATION METRICS              │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("  MODEL EVALUATION METRICS")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n Key Performance Metrics:")
print(f"   Accuracy  : {accuracy * 100:.2f}%   (How many predictions correct)")
print(f"   ROC-AUC   : {roc_auc:.4f}     (0.5=random, 1.0=perfect)")

print(f"\n Detailed Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Non-Diabetic (0)", "Diabetic (1)"],
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n Confusion Matrix Breakdown:")
print(f"   True Negatives  (TN)  : {tn}  (Correctly predicted healthy)")
print(f"   False Positives (FP)  : {fp}  (Incorrectly predicted diabetic)")
print(f"   False Negatives (FN)  : {fn}  (Incorrectly predicted healthy)")
print(f"   True Positives  (TP)  : {tp}  (Correctly predicted diabetic)")


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 7 — PLOT 1: CONFUSION MATRIX                 │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("  CREATING VISUALIZATIONS")
print("="*60)

fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Diabetic", "Diabetic"]
)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title("Confusion Matrix — Diabetes Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_4_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("    Saved: plot_4_confusion_matrix.png")
plt.close()