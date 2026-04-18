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

# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 4 — VISUALIZATION 1: TARGET DISTRIBUTION      │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

# Plot 1: Outcome Distribution (Bar Chart)
plt.figure(figsize=(8, 5))
colors = ["#4CAF50", "#F44336"]  # Green for healthy, red for diabetic
sns.countplot(x="Outcome", data=df, palette=colors, hue="Outcome", legend=False)
plt.title("Target Distribution: Healthy vs Diabetic", fontsize=14, fontweight="bold")
plt.xlabel("Outcome", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.xticks([0, 1], ["Non-Diabetic (0)", "Diabetic (1)"])
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_1_target_distribution.png", dpi=150, bbox_inches="tight")
print("   Saved: plot_1_target_distribution.png")
plt.close()


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 5 — VISUALIZATION 2: FEATURE DISTRIBUTIONS    │
# └─────────────────────────────────────────────────────────┘

# Plot 2: Feature Histograms (8 subplots)
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()  # Flatten 2D array to 1D for easier iteration

features = df.columns[:-1]  # All columns except "Outcome"

for i, col in enumerate(features):
    axes[i].hist(df[col], bins=25, color="#5C6BC0", edgecolor="white", alpha=0.7)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    axes[i].grid(axis="y", alpha=0.3)

plt.suptitle("Feature Distributions (Histograms)", fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig("plot_2_feature_distributions.png", dpi=150, bbox_inches="tight")
print("   Saved: plot_2_feature_distributions.png")
plt.close()


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 6 — VISUALIZATION 3: CORRELATION HEATMAP      │
# └─────────────────────────────────────────────────────────┘

# Plot 3: Correlation Matrix
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,  # Show correlation values
    fmt=".2f",   # 2 decimal places
    cmap="coolwarm",  # Red/Blue color scheme
    square=True,
    linewidths=1,
    cbar_kws={"label": "Correlation"}
)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_3_correlation_heatmap.png", dpi=150, bbox_inches="tight")
print("   Saved: plot_3_correlation_heatmap.png")
plt.close()

print("\n    What does the heatmap show?")
print("      • Dark red   = Strong positive correlation")
print("      • Dark blue  = Strong negative correlation")
print("      • White      = No correlation")

# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 7 — DATA PREPROCESSING (HANDLE ZEROS)         │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print(" DATA PREPROCESSING ")
print("="*60)

df_clean = df.copy()

print("\n  Replacing impossible zeros with MEDIAN values...")

# Replace zeros with median for each column
for col in zero_columns:
    # Count zeros before replacement
    zero_before = (df_clean[col] == 0).sum()
    
    # Calculate median (ignoring the zeros)
    median_val = df_clean[col].replace(0, np.nan).median()
    
    # Replace zeros with median
    df_clean[col] = df_clean[col].replace(0, median_val)
    
    # Count zeros after replacement
    zero_after = (df_clean[col] == 0).sum()
    
    print(f"   {col:>16} → {zero_before:>3} zeros replaced with median={median_val:>6.2f}")

print("\n   All impossible zeros replaced!")

# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 8 — STANDARD SCALING (Normalize Features)     │
# └─────────────────────────────────────────────────────────┘

from sklearn.preprocessing import StandardScaler

print("\n" + "="*60)
print("  FEATURE SCALING (StandardScaler)")
print("="*60)

print("\n Why scale features?")
print("   • Logistic Regression is distance-based")
print("   • Features with large values (e.g., Glucose) dominate")
print("   • Scaling makes all features equal importance")
print("   • Formula: (x - mean) / std_dev → mean=0, std=1")

# Separate features (X) and target (y)
X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]

print(f"\n   X shape (features) : {X.shape}")
print(f"   y shape (target)   : {y.shape}")

# Initialize and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for easier handling
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\n    Features scaled!")
print(f"\n   Check: Scaled data stats")
print(f"   • Mean : {X_scaled.mean().mean():.6f} (should be ~0)")
print(f"   • Std  : {X_scaled.std().mean():.6f} (should be ~1)")


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 9 — TRAIN/TEST SPLIT                          │
# └─────────────────────────────────────────────────────────┘

from sklearn.model_selection import train_test_split

print("\n" + "="*60)
print("  TRAIN/TEST SPLIT (80/20)")
print("="*60)

print("\n Why split data?")
print("   • Training set (80%): teach the model")
print("   • Testing set (20%): evaluate performance")
print("   • Tests on unseen data (real-world scenario)")
print("   • Stratify: keeps class ratio in both sets")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n   Total samples         : {len(X_scaled)}")
print(f"   Training samples (80%): {len(X_train)}")
print(f"   Testing samples (20%) : {len(X_test)}")

print(f"\n   Training set class distribution:")
print(f"   • Non-Diabetic : {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"   • Diabetic     : {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

print(f"\n   Testing set class distribution:")
print(f"   • Non-Diabetic : {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"   • Diabetic     : {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 10 — SAVE SCALER FOR MODEL TRAINING          │
# └─────────────────────────────────────────────────────────┘

import pickle

print("\n" + "="*60)
print("  SAVING SCALER & PREPROCESSED DATA")
print("="*60)

# Save the fitted scaler (needed later in model_training.py and app.py)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n Files Saved:")
print("    scaler.pkl  → StandardScaler (for model training & web app)")

# Also save the processed data splits for later use (optional but helpful)
import os
if not os.path.exists("data"):
    os.makedirs("data")

# Save train/test splits
with open("data/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("data/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("data/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("data/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("   ✓ data/X_train.pkl → Training features (614 samples)")
print("   ✓ data/X_test.pkl  → Testing features (154 samples)")
print("   ✓ data/y_train.pkl → Training labels")
print("   ✓ data/y_test.pkl  → Testing labels")



# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 11 — QUICK PREDICTION TEST                    │
# └─────────────────────────────────────────────────────────┘

print("\n" + "="*60)
print("  QUICK PREDICTION TEST")
print("="*60)

# Test: Create a sample patient and see if preprocessing works
sample_patient = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
# Features: Pregnancies, Glucose, BloodPressure, SkinThickness, 
#           Insulin, BMI, DiabetesPedigreeFunction, Age

sample_scaled = scaler.transform(sample_patient)

print("\n Sample Patient Test:")
print(f"   Input (raw)     : {sample_patient[0]}")
print(f"   After scaling   : {sample_scaled[0]}")
print(f"    Scaling works correctly!")

print("\n" + "="*60)
print("   DATA PREPROCESSING COMPLETE!")
print("="*60)
print("="*60)