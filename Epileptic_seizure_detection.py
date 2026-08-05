"""
EPILEPTIC SEIZURE RECOGNITION — Binary Classification from EEG Features
Dataset: Epileptic Seizure Recognition (Kaggle / UCI) — 11,500 EEG segments, 178 features each

PIPELINE
  1. Load data, binarize labels (1 = seizure, 0 = non-seizure)
  2. Train/test split (stratified, so both sets keep the ~20/80 class ratio)
  3. Scale features (needed for SVM / Logistic Regression, not for Random Forest)
  4. Handle class imbalance via class_weight='balanced'
  5. Benchmark 3 models with 5-fold stratified cross-validation (mean +- std)
  6. Report precision/recall/F1 specifically on the seizure class (accuracy alone is misleading here)
  7. Plot ROC and Precision-Recall curves for all 3 models on one figure
  8. Explain the best model with SHAP (which EEG feature indices drive predictions)

HOW TO RUN
  pip install pandas scikit-learn matplotlib shap --break-system-packages   (if needed)
  python seizure_classification.py
  Edit DATA_PATH below if your CSV lives somewhere else.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

RANDOM_STATE = 42
DATA_PATH = "C:/Users/hprai/Downloads/archive/Epileptic Seizure Recognition.csv"

# ─────────────────────────────────────────────────────────────────────────
# 1. LOAD + PREP DATA
# ─────────────────────────────────────────────────────────────────────────
data = pd.read_csv(DATA_PATH)
print("Raw shape:", data.shape)

data = data.iloc[:, 1:]  # drop the row-id column (e.g. "X12.V1.764...")
X = data.drop(columns=["y"])
y = data["y"].apply(lambda v: 1 if v == 1 else 0)  # 1 = seizure, 0 = non-seizure

print("Feature matrix:", X.shape)
print("Class balance:\n", y.value_counts(normalize=True).rename("proportion"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────
# 2. SCALING (fit on train only, applied to both — avoids leakage)
# ─────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────
# 3. MODELS — each explicitly named, seeded, and imbalance-aware
# ─────────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        True,  # needs scaled features
    ),
    "SVM": (
        SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        True,
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        False,  # scale-invariant
    ),
}

# ─────────────────────────────────────────────────────────────────────────
# 4. 5-FOLD STRATIFIED CROSS-VALIDATION (mean +- std, not a single split)
# ─────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]

print("\n" + "=" * 65)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 65)

cv_summary = []
for name, (model, needs_scaling) in models.items():
    X_cv = scaler.fit_transform(X) if needs_scaling else X
    scores = cross_validate(model, X_cv, y, cv=cv, scoring=scoring, n_jobs=-1)
    row = {"Model": name}
    for metric in scoring:
        vals = scores[f"test_{metric}"]
        row[metric] = f"{vals.mean():.3f} ± {vals.std():.3f}"
    cv_summary.append(row)
    print(f"\n{name}")
    for metric in scoring:
        vals = scores[f"test_{metric}"]
        print(f"  {metric:<10}: {vals.mean():.3f} ± {vals.std():.3f}")

cv_summary_df = pd.DataFrame(cv_summary)
cv_summary_df.to_csv("cv_results_summary.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────
# 5. FIT ON FULL TRAIN SET, EVALUATE ON HELD-OUT TEST SET
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("HELD-OUT TEST SET RESULTS (seizure = positive class)")
print("=" * 65)

fitted_models = {}
test_results = []

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    fitted_models[name] = (model, needs_scaling)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-seizure", "Seizure"], output_dict=True)
    seizure_recall = report["Seizure"]["recall"]
    seizure_precision = report["Seizure"]["precision"]
    seizure_f1 = report["Seizure"]["f1-score"]
    ap = average_precision_score(y_test, y_prob)

    print(f"\n{name}")
    print(f"  Overall accuracy      : {acc:.3f}")
    print(f"  Seizure recall        : {seizure_recall:.3f}  (share of true seizures caught)")
    print(f"  Seizure precision     : {seizure_precision:.3f}  (share of seizure alerts that are real)")
    print(f"  Seizure F1            : {seizure_f1:.3f}")
    print(f"  Average precision(AP) : {ap:.3f}")
    print("  Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, y_pred))

    test_results.append({
        "Model": name, "Accuracy": acc, "Seizure Recall": seizure_recall,
        "Seizure Precision": seizure_precision, "Seizure F1": seizure_f1, "Avg Precision": ap,
    })

test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv("test_results_summary.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────
# 6. ROC + PRECISION-RECALL CURVES — ALL 3 MODELS, ONE FIGURE
# ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, (model, needs_scaling) in fitted_models.items():
    Xte = X_test_scaled if needs_scaling else X_test
    y_prob = model.predict_proba(Xte)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})")

axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=8)

axes[1].axhline(y_test.mean(), color="k", ls="--", lw=0.8, label="Chance (baseline rate)")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision–Recall Curve"); axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("roc_pr_curves.png", dpi=150)
print("\nSaved: roc_pr_curves.png")

# ─────────────────────────────────────────────────────────────────────────
# 7. SHAP EXPLAINABILITY ON THE BEST MODEL (by test seizure-F1)
# ─────────────────────────────────────────────────────────────────────────
best_name = test_results_df.sort_values("Seizure F1", ascending=False).iloc[0]["Model"]
best_model, best_needs_scaling = fitted_models[best_name]
print(f"\nBest model by seizure F1: {best_name}")

try:
    import shap
    Xte_shap = X_test_scaled if best_needs_scaling else X_test.values
    feature_names = X.columns.tolist()

    if best_name == "Random Forest":
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(Xte_shap)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    else:
        # KernelExplainer is slow — subsample background + test rows
        bg = shap.sample(X_train_scaled if best_needs_scaling else X_train.values, 100, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(best_model.predict_proba, bg)
        sv = explainer.shap_values(Xte_shap[:200], nsamples=100)
        Xte_shap = Xte_shap[:200]

    # Normalize sv to a 2D (n_samples, n_features) array of |SHAP| values.
    # Different shap versions return: a list of per-class arrays, a 3D array
    # (n_samples, n_features, n_classes), or a plain 2D array.
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
        mean_abs_shap = np.abs(sv).mean(axis=0)
    else:
        sv = np.array(sv)
        if sv.ndim == 3:
            # last axis = classes -> take the seizure (positive) class if present
            class_idx = 1 if sv.shape[-1] > 1 else 0
            mean_abs_shap = np.abs(sv[:, :, class_idx]).mean(axis=0)
        else:
            mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]

    shap_df = pd.DataFrame({
        "feature": [feature_names[i] for i in top_idx],
        "mean_abs_shap": mean_abs_shap[top_idx],
    })
    shap_df.to_csv("shap_top_features.csv", index=False)
    print("Saved: shap_top_features.csv (top 15 EEG features driving predictions)")

    plt.figure(figsize=(8, 6))
    plt.barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top 15 EEG Features — {best_name}")
    plt.tight_layout()
    plt.savefig("shap_top_features.png", dpi=150)
    print("Saved: shap_top_features.png")

except ImportError:
    print("shap not installed — skipping explainability step. "
          "Run: pip install shap --break-system-packages")

# ─────────────────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY PRINTOUT
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY (copy these numbers for the README / CV)")
print("=" * 65)
print("\n-- 5-fold CV results --")
print(cv_summary_df.to_string(index=False))
print("\n-- Held-out test set results --")
print(test_results_df.round(3).to_string(index=False))
print(f"\nBest model: {best_name}")
print("=" * 65)
