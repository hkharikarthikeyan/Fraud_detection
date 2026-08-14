import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from src.config import RANDOM_STATE
from src.utils import print_section

def temporal_split(X, y, df):
    print_section("TEMPORAL TRAIN / VALIDATION SPLIT")

    order = np.argsort(df["TransactionDT"].values)

    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)

    split_point = int(len(X) * 0.80)
    X_train = X.iloc[:split_point].copy()
    X_valid = X.iloc[split_point:].copy()
    y_train = y.iloc[:split_point].copy()
    y_valid = y.iloc[split_point:].copy()
    
    print("Training shape:", X_train.shape)
    print("Validation shape:", X_valid.shape)
    print("Training fraud rate:", f"{y_train.mean():.4f}")
    print("Validation fraud rate:", f"{y_valid.mean():.4f}")

    return X_train, X_valid, y_train, y_valid

def impute_data(X_train, X_valid):
    print_section("MISSING VALUE IMPUTATION")
    imputer = SimpleImputer(strategy="mean")
    X_train_processed = imputer.fit_transform(X_train)
    X_valid_processed = imputer.transform(X_valid)
    print("Imputation completed.")

    return imputer, X_train_processed, X_valid_processed

def train_baseline(X_train, y_train):
    print_section("TRAINING BASELINE XGBOOST")

    fraud_count = int(y_train.sum())
    normal_count = len(y_train) - fraud_count
    scale_pos_weight = normal_count / max(fraud_count, 1)

    print("Scale positive weight:", round(scale_pos_weight, 2))

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train)
    print("Baseline XGBoost trained.")

    return model

def apply_smote(X_train, y_train):
    print_section("SMOTE RESAMPLING (imbalanced-learn)")
    print(f"Before SMOTE — Fraud: {y_train.sum():,}  Normal: {(y_train==0).sum():,}")
    smote = SMOTE(sampling_strategy=0.20, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After  SMOTE — Fraud: {y_resampled.sum():,}  Normal: {(y_resampled==0).sum():,}")

    return X_resampled, y_resampled

def train_improved_model(X_train, y_train):
    print_section("TRAINING BEHAVIOR-AWARE XGBOOST")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train)
    print("Behavior-aware XGBoost trained.")
    return model

def train_anomaly_model(X_train, y_train):
    print_section("TRAINING ANOMALY DETECTION MODEL")
    normal_indices = np.where(y_train.values == 0)[0]
    sample_size = min(100000, len(normal_indices))

    rng = np.random.default_rng(RANDOM_STATE)
    selected_indices = rng.choice(normal_indices, size=sample_size, replace=False)
    print("Normal samples used:", sample_size)
    
    anomaly_model = IsolationForest(
        n_estimators=100, 
        contamination="auto", 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    anomaly_model.fit(X_train[selected_indices])
    print("Isolation Forest trained.")
    return anomaly_model

def calculate_anomaly_score(anomaly_model, X_valid):
    raw_score = -anomaly_model.score_samples(X_valid)

    score_min = raw_score.min()
    score_max = raw_score.max()

    normalized_score = (raw_score - score_min) / (score_max - score_min + 1e-8)
    return np.clip(normalized_score, 0, 1)

def get_behavior_scores(df, X_valid):
    # The final validation rows correspond to the final 20% chronological section.
    split_point = int(len(df) * 0.80)
    validation_df = df.iloc[split_point:].copy()

    if "behavioral_risk_score" in validation_df.columns:
        scores = validation_df["behavioral_risk_score"].fillna(0).values
    else:
        scores = np.zeros(len(X_valid))

    return scores

def calculate_final_risk(fraud_probability, behavior_score, anomaly_score):
    final_score = (
        0.70 * fraud_probability
        +
        0.20 * behavior_score
        +
        0.10 * anomaly_score
    )
    return np.clip(final_score, 0, 1)

def risk_category(score):
    if score < 0.30:
        return "LOW"
    elif score < 0.70:
        return "MEDIUM"
    else:
        return "HIGH"
