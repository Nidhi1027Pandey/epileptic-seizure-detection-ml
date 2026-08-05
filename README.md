# Epileptic Seizure Detection from EEG Signals

Binary classification of epileptic seizure activity from single-channel EEG segments, benchmarked across three model families with rigorous handling of class imbalance and explainability analysis.

**Dataset**: [Epileptic Seizure Recognition](https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition) (UCI / Kaggle) — 11,500 EEG segments × 178 time-domain features, 20% positive (seizure) class.

## Results

Best model: **SVM (RBF kernel)** — selected by seizure-class F1 on held-out test data.

### Held-out test set (2,300 samples, unseen during training/CV)

| Model | Accuracy | Seizure Recall | Seizure Precision | Seizure F1 | Avg. Precision |
|---|---|---|---|---|---|
| Logistic Regression | 0.706 | 0.393 | 0.313 | 0.348 | 0.398 |
| **SVM (RBF)** | **0.975** | **0.965** | **0.915** | **0.940** | **0.981** |
| Random Forest | 0.973 | 0.896 | 0.967 | 0.930 | 0.988 |

### 5-fold stratified cross-validation (mean ± std)

| Model | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.704 ± 0.007 | 0.442 ± 0.017 | 0.324 ± 0.008 | 0.374 ± 0.010 | 0.530 ± 0.017 |
| **SVM (RBF)** | **0.975 ± 0.002** | **0.960 ± 0.006** | **0.918 ± 0.007** | **0.939 ± 0.005** | **0.995 ± 0.001** |
| Random Forest | 0.970 ± 0.003 | 0.884 ± 0.010 | 0.963 ± 0.008 | 0.922 ± 0.007 | 0.996 ± 0.001 |

Held-out and cross-validated numbers agree closely (SVM: 0.940 test F1 vs. 0.939 ± 0.005 CV F1), indicating the result is stable rather than a lucky split.

### Why SVM over Random Forest, despite near-identical accuracy

Both models score ~97% accuracy, but they fail differently — the choice depends on what a false negative costs versus what a false alarm costs:

- **SVM** — 96.5% recall, 91.5% precision. Misses fewer true seizures (16 of 460 in the test set) but raises more false alarms (41). Preferable when missing a seizure is the costlier error, e.g. a clinical monitoring alert system.
- **Random Forest** — 89.6% recall, 96.7% precision. Fewer false alarms (14) but misses more true seizures (48). Preferable when alert fatigue is the dominant concern.

This precision/recall tradeoff is visualized in `roc_pr_curves.png`.

## Method

1. **Preprocessing** — binarized the 5-class label into seizure (1) vs. non-seizure (2–5); stratified 80/20 train/test split to preserve the 80/20 class ratio in both sets.
2. **Scaling** — `StandardScaler` fit on training data only, applied to SVM and Logistic Regression (both scale-sensitive); Random Forest left unscaled (scale-invariant).
3. **Class imbalance** — `class_weight='balanced'` on all three models rather than naive accuracy optimization, given the 80/20 split.
4. **Validation** — 5-fold stratified cross-validation reported alongside a single held-out test set, to confirm results generalize rather than reflect one split.
5. **Explainability** — SHAP (`TreeExplainer` for Random Forest, `KernelExplainer` for SVM) to identify which of the 178 EEG feature indices drive seizure predictions, saved to `shap_top_features.csv` / `.png`.

## Repository contents

```
seizure_classification.py      # full pipeline: load → scale → CV → test eval → SHAP
cv_results_summary.csv         # 5-fold CV metrics per model
test_results_summary.csv       # held-out test metrics per model
roc_pr_curves.png              # ROC + precision-recall curves, all 3 models
shap_top_features.csv          # top 15 EEG features by mean |SHAP value|
shap_top_features.png          # SHAP feature importance chart
```

## Run it

```bash
pip install pandas scikit-learn matplotlib shap
python seizure_classification.py
```

Edit `DATA_PATH` at the top of the script to point to your local copy of the dataset CSV.

## Stack

Python · pandas · scikit-learn (SVM, Random Forest, Logistic Regression) · SHAP · matplotlib
