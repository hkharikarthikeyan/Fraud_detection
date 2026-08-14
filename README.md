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
- Runs locally (optimizing memory usage via row limits) and in Google Colab (self-contained layout)

---

## Solution Approach

### 1. Data Ingestion
- Automatically downloads the IEEE-CIS dataset via `kagglehub`
- Merges `train_transaction.csv` and `train_identity.csv` on `TransactionID`
- Supports custom row subsetting (`nrows`) to prevent Out Of Memory (OOM) errors locally

### 2. Exploratory Data Analysis
- Fraud rate analysis (2.56% in subset, ~3.5% overall)
- Missing value profiling
- Target distribution visualization

### 3. Feature Engineering
- **Time features** — transaction hour, day, weekday, night-time flag
- **Traditional features** — log transaction amount, card/address missing counts
- **Behavioral features** — amount deviation ratio, card velocity, device novelty, behavioral risk score

### 4. Preprocessing
- Frequency encoding for categorical variables
- Mean imputation for missing values
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

## Results (Subsampled Execution)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline XGBoost | 89.51% | 58.76% | 31.15% | 65.37% | 42.19% |
| Behavior-Aware XGBoost | 90.96% | 61.02% | 85.47% | 45.87% | 59.70% |
| Final Hybrid | 85.41% | 58.11% | 78.83% | 49.54% | 60.85% |

**Verdict: VERY GOOD — Strong fraud detection, suitable for deployment with monitoring.**

---

## Project Structure

```
fraud-detection/
├── train.py               # Main training entry point
├── predict.py             # Main inference/prediction entry point
├── fraud_detection.ipynb  # Self-contained Google Colab notebook
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── src/                   # Package sources
│   ├── config.py          # Paths and constant settings
│   ├── data_preprocessing.py # Preprocessing & Feature Engineering
│   ├── models.py          # Model architectures and splitting
│   ├── evaluation.py      # Metrics and plot generation
│   └── utils.py           # Ingestion and helper functions
├── data/                  # Downloaded CSVs (auto-created)
├── models/                # Saved model checkpoints (auto-created)
└── reports/               # Saved reports & plots (auto-created)
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

### Local Training
To run the full preprocessing, training, evaluation, and serialization pipeline locally:
```bash
python train.py
```
*Note: This script automatically limits transaction ingestion to 100,000 rows to fit inside standard RAM environments.*

### Local Inference
To run predictions on a slice of transactions using the saved model checkpoints:
```bash
python predict.py
```

### Google Colab Execution
Open [fraud_detection.ipynb](file:///d:/projects/Fraud%20detection/fraud_detection.ipynb) in Google Colab:
- Cell 1 installs dependencies.
- Cells 2 to 6 load all configuration, feature engineering, modeling, and evaluation functions locally in memory.
- Cell 7 loads the **full dataset** (no row limits) and runs the entire training/testing pipeline.
- Does not require mounting Google Drive or uploading any Python package dependencies.

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
