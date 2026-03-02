"""
MFI Risk Scoring - Model Training Script
=========================================
Trains a RandomForest classifier on consortium MFI data (data.csv)
and saves the model + artefacts to model/

Usage:
    python train_model.py

Outputs (saved to ./model/):
    risk_model.pkl       - Trained RandomForestClassifier pipeline
    label_encoder.pkl    - LabelEncoder for Risk Label
    feature_cols.pkl     - Ordered list of feature names
    training_report.txt  - Classification report + feature importances
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ----------------------------------------------
#  Config
# ----------------------------------------------
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "Current Monthly Salary (USD)",
    "Total Previous Loans",
    "Active Loans",
    "Total Outstanding Balance (USD)",
    "Avg Loan Amount (USD)",
    "Historical Return Rate (%)",
    "Days Past Due (Max)",
    "MFI Diversity Score",
]

CATEGORICAL_FEATURES = [
    "Employment Sector",
    "Common Loan Reason",
]

TARGET = "Risk Label"

# ----------------------------------------------
#  Step 1 - Load & Validate Data
# ----------------------------------------------
print("=" * 60)
print("  MFI RISK MODEL TRAINING PIPELINE")
print("=" * 60)

print(f"\n[1/6] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"      ok {len(df)} records loaded, {len(df.columns)} columns")
print(f"      ok Risk Label distribution:")
dist = df[TARGET].value_counts()
for label, count in dist.items():
    pct = count / len(df) * 100
    bar = "#" * int(pct / 3)
    print(f"        {label:<8} {bar} {count} ({pct:.1f}%)")

missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"      warn  Missing values detected:\n{missing[missing > 0]}")
else:
    print("      ok No missing values")

# ----------------------------------------------
#  Step 2 - Feature Engineering
# ----------------------------------------------
print("\n[2/6] Engineering features...")

df = df.copy()

# Derived numeric features
df["Debt_to_Income"]      = df["Total Outstanding Balance (USD)"] / df["Current Monthly Salary (USD)"].clip(lower=1)
df["Loan_to_Income"]      = df["Avg Loan Amount (USD)"] / df["Current Monthly Salary (USD)"].clip(lower=1)
df["Active_Loan_Density"] = df["Active Loans"] / df["Total Previous Loans"].clip(lower=1)
df["Return_Rate_Norm"]    = df["Historical Return Rate (%)"] / 100.0
df["Is_Overdue"]          = (df["Days Past Due (Max)"] > 0).astype(int)
df["Overdue_Severity"]    = pd.cut(
    df["Days Past Due (Max)"],
    bins=[-1, 0, 30, 60, 90, float("inf")],
    labels=[0, 1, 2, 3, 4]
).astype(int)

ENGINEERED_FEATURES = [
    "Debt_to_Income", "Loan_to_Income", "Active_Loan_Density",
    "Return_Rate_Norm", "Is_Overdue", "Overdue_Severity"
]

# One-hot encode categorical features
df_encoded = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
cat_feature_names = list(df_encoded.columns)

ALL_FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES + cat_feature_names
X = pd.concat([df[NUMERIC_FEATURES + ENGINEERED_FEATURES], df_encoded], axis=1)

print(f"      ok {len(NUMERIC_FEATURES)} numeric features")
print(f"      ok {len(ENGINEERED_FEATURES)} engineered features")
print(f"      ok {len(cat_feature_names)} one-hot encoded features")
print(f"      ok Total feature vector size: {X.shape[1]}")

# Encode target
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
print(f"      ok Label classes: {list(le.classes_)}")

# ----------------------------------------------
#  Step 3 - Train / Test Split
# ----------------------------------------------
print("\n[3/6] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      ok Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ----------------------------------------------
#  Step 4 - Train Model
# ----------------------------------------------
print("\n[4/6] Training RandomForest model...")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ))
])

model.fit(X_train, y_train)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"      ok CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# ----------------------------------------------
#  Step 5 - Evaluate
# ----------------------------------------------
print("\n[5/6] Evaluating on test set...")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"      ok Test Accuracy: {acc:.3f}")
print(f"      ok AUC-ROC (macro): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'):.3f}")

report = classification_report(y_test, y_pred, target_names=le.classes_)
print("\n      Classification Report:")
for line in report.split("\n"):
    print(f"        {line}")

# Feature importances (top 15)
rf = model.named_steps["clf"]
importances = pd.Series(rf.feature_importances_, index=ALL_FEATURES)
importances_sorted = importances.sort_values(ascending=False).head(15)
print("\n      Top 15 Feature Importances:")
for feat, imp in importances_sorted.items():
    bar = "#" * int(imp * 100)
    print(f"        {feat:<45} {bar} {imp:.4f}")

# ----------------------------------------------
#  Step 6 - Save Artefacts
# ----------------------------------------------
print("\n[6/6] Saving model artefacts...")

# Save model
model_path = os.path.join(MODEL_DIR, "risk_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"      ok Model saved       -> {model_path}")

# Save label encoder
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
with open(le_path, "wb") as f:
    pickle.dump(le, f)
print(f"      ok Label encoder     -> {le_path}")

# Save feature column list
feat_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
with open(feat_path, "wb") as f:
    pickle.dump(ALL_FEATURES, f)
print(f"      ok Feature columns   -> {feat_path}")

# Save metadata for the app (salary buckets, sector profiles)
sector_profiles = df.groupby("Employment Sector").agg(
    avg_salary=("Current Monthly Salary (USD)", "mean"),
    avg_return_rate=("Historical Return Rate (%)", "mean"),
    avg_days_past_due=("Days Past Due (Max)", "mean"),
    avg_active_loans=("Active Loans", "mean"),
    avg_previous_loans=("Total Previous Loans", "mean"),
    avg_outstanding=("Total Outstanding Balance (USD)", "mean"),
    avg_loan_amount=("Avg Loan Amount (USD)", "mean"),
    avg_mfi_score=("MFI Diversity Score", "mean"),
    common_reason=("Common Loan Reason", lambda x: x.mode()[0])
).reset_index().to_dict("records")

salary_percentiles = {
    "p10": float(df["Current Monthly Salary (USD)"].quantile(0.10)),
    "p25": float(df["Current Monthly Salary (USD)"].quantile(0.25)),
    "p50": float(df["Current Monthly Salary (USD)"].quantile(0.50)),
    "p75": float(df["Current Monthly Salary (USD)"].quantile(0.75)),
    "p90": float(df["Current Monthly Salary (USD)"].quantile(0.90)),
}

metadata = {
    "sector_profiles": sector_profiles,
    "salary_percentiles": salary_percentiles,
    "label_classes": list(le.classes_),
    "all_features": ALL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "engineered_features": ENGINEERED_FEATURES,
    "cat_feature_names": cat_feature_names,
    "accuracy": float(acc),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
}
meta_path = os.path.join(MODEL_DIR, "metadata.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"      ok Metadata          -> {meta_path}")

# Save human-readable report
report_path = os.path.join(MODEL_DIR, "training_report.txt")
with open(report_path, "w") as f:
    f.write("MFI RISK MODEL - TRAINING REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Records: {len(df)}\n\n")
    f.write("Risk Label Distribution:\n")
    f.write(str(dist) + "\n\n")
    f.write(f"CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}\n")
    f.write(f"Test Accuracy: {acc:.3f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Feature Importances (Top 15):\n")
    f.write(str(importances_sorted) + "\n")
print(f"      ok Training report   -> {report_path}")

print("\n" + "=" * 60)
print("  ok  TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Model accuracy : {acc:.1%}")
print(f"  CV accuracy    : {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
print(f"  Classes        : {list(le.classes_)}")
print(f"\n  Run app.py to start the dashboard.\n")
