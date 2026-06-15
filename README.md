
# EEG Seizure Classification

ML models for seizure detection from EEG data, comparing Logistic Regression, SVM, and Random Forest classifiers.

## 📌 Overview
This project uses the **Epileptic Seizure Recognition** dataset to build binary classifiers (seizure vs. non-seizure) based on EEG signal features. Multiple machine learning models are trained and compared, with performance evaluated using accuracy, classification reports, ROC-AUC, and cross-validation.

## 📊 Dataset
- Source: [Add dataset link - e.g., Kaggle/UCI]
- 11,500 samples × 178 EEG features
- Original labels (1–5) converted to binary: `1` = seizure, `0` = non-seizure (labels 2–5)
- Train/Test split: 80% / 20%

## 🛠️ Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 80.39% |
| SVM | 97.87% |
| Random Forest | 97.87% |

**Best Model: Random Forest / SVM**

**Cross-Validation Accuracy (Random Forest, 5-fold):** 97.42%

### Classification Report (Random Forest)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Non-Seizure (0) | 0.99 | 0.99 | 0.99 |
| Seizure (1) | 0.95 | 0.94 | 0.95 |
| **Overall** | **0.98** | **0.98** | **0.98** |

### Visualizations

**Feature Importance (Random Forest)**

![Feature Importance](results/feature_importance.png)

**ROC Curve (AUC = 1.00)**

![ROC Curve](results/roc_curve.png)

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python seizure_detection.py
```

> Download the dataset from [Add link] and place it in a `data/` folder before running.

## 📁 Project Structure
```
eeg-seizure-classification/
├── seizure_detection.py
├── requirements.txt
├── results/
│   ├── feature_importance.png
│   └── roc_curve.png
└── README.md
```

## 🔮 Future Improvements
- Hyperparameter tuning (GridSearchCV)
- Try additional models (XGBoost, Neural Networks)
- Feature selection / dimensionality reduction (PCA)
- Fix Logistic Regression convergence (scaling + higher max_iter)

## 👩‍💻 Author
**Nidhi Pandey** — M.Tech Biomedical Engineering, IIT Bombay
