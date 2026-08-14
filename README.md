# Behavior-Aware Explainable Fraud Detection System

A machine learning system to detect fraudulent financial transactions using the IEEE-CIS Fraud Detection dataset. The system combines XGBoost, SMOTE resampling, Isolation Forest anomaly detection, and SHAP explainability into a modular pipeline.

---

## Project Overview

Financial institutions face significant losses due to fraudulent transactions. This project builds a behavior-aware fraud detection system that:

- Detects fraud using gradient boosting (XGBoost)
- Handles class imbalance using SMOTE (imbalanced-learn)
- Engineers behavioral features from transaction history
- Provides explainability using SHAP values
- Outputs hybrid risk scores combining model probability, behavioral signals, and anomaly scores

---

## Solution Approach

### 1. Data Ingestion
- Automatically downloads the IEEE-CIS dataset via `kagglehub`
- Merges `train_transaction.csv` and `train_identity.csv` on `TransactionID`

### 2. Exploratory Data Analysis
- Fraud rate analysis (3.5% imbalance)
- Missing value profiling
- Target distribution visualization

### 3. Feature Engineering
- **Time features** — transaction hour, day, weekday, night-time flag
- **Traditional features** — log transaction amount, card/address missing counts
- **Behavioral features** — amount deviation ratio, card velocity, device novelty, behavioral risk score

### 4. Preprocessing
- Frequency encoding for categorical variables
- Median imputation for missing values
- Removal of features with >95% missing values

### 5. Class Imbalance Handling
- `scale_pos_weight` in baseline XGBoost
- SMOTE (sampling_strategy=0.20) for behavior-aware model

### 6. Model Training
- **Baseline XGBoost** — with scale_pos_weight
- **Behavior-Aware XGBoost** — trained on SMOTE-resampled data with behavioral features
- **Isolation Forest** — anomaly detection on normal transactions

### 7. Hybrid Risk Scoring
Final risk score = 0.70 × XGBoost probability + 0.20 × behavioral score + 0.10 × anomaly score

### 8. Evaluation
- ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score
- Confusion matrix for all three models
- F1-optimized threshold tuning
- SHAP summary plot for explainability

---

## Results

|Model                        |     ROC-AUC  |   PR-AUC | Precision   |  Recall     |    F1|
|---|---|---|---|---|---|
|Baseline XGBoost            |       90.46%  |   52.16%  |   21.66%  |   73.38%  |   33.44%|
|Behavior-Aware XGBoost      |       89.38%  |   50.31%  |   76.82%  |   33.19%  |   46.36%|
|Final Hybrid                |       85.81%  |  46.90%   |  50.14%   |  44.61%   |  47.21%|

**Verdict: VERY GOOD — Strong fraud detection, suitable for deployment with monitoring.**

---

## Project Structure

```
fraud-detection-system/
├── frauddetection.py       # Main pipeline
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── data/                   # Downloaded CSVs (auto-created)
├── models/                 # Saved model artifacts
│   ├── baseline_xgboost.pkl
│   ├── behavior_aware_xgboost.pkl
│   ├── isolation_forest.pkl
│   ├── imputer.pkl
│   └── optimal_threshold.pkl
└── reports/
    ├── missing_values.csv
    ├── feature_importance.csv
    ├── threshold_analysis.csv
    ├── model_comparison.csv
    ├── configuration.json
    └── plots/
        ├── fraud_distribution.png
        ├── shap_summary.png
        ├── feature_importance.png
        ├── baseline_confusion_matrix.png
        ├── behavior_confusion_matrix.png
        └── final_confusion_matrix.png
```

---

## Setup and Usage

### Prerequisites
- Python 3.8+
- Kaggle account (for dataset download)

### Installation

```bash
git clone https://github.com/hkharikarthikeyan/Fraud_detection.git
cd Fraud_detection
pip install -r requirements.txt
```

### Run

```bash
python frauddetection.py
```

The script will automatically download the IEEE-CIS dataset via `kagglehub` on first run. You will be prompted to authenticate with your Kaggle account.

### Google Colab

```python
# Cell 1
!pip install kagglehub xgboost imbalanced-learn shap -q

# Cell 2
import os
os.environ["FRAUD_PROJECT_DIR"] = "/content/fraud_detection_project"

# Cell 3
# Paste frauddetection.py contents here and run
```

---

## Dependencies

| Library | Purpose |
|---|---|
| xgboost | Primary fraud detection model |
| imbalanced-learn | SMOTE for class imbalance |
| shap | Model explainability |
| scikit-learn | Preprocessing, metrics, Isolation Forest |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |
| kagglehub | Automatic dataset download |
| joblib | Model serialization |

---

## Key Design Decisions

- **Temporal validation** — 80/20 chronological split prevents data leakage
- **PR-AUC over ROC-AUC** — more reliable metric for imbalanced fraud data
- **Threshold tuning** — F1-optimized threshold instead of default 0.5
- **Behavioral risk score** — domain-aware composite score using velocity, device novelty, and amount deviation
- **Hybrid scoring** — combines model probability with behavioral and anomaly signals for robust final risk assessment

---

## Dataset

IEEE-CIS Fraud Detection Dataset  
Source: [Kaggle](https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection)  
Transactions: 590,540 | Features: 434 | Fraud rate: 3.5%
