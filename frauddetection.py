"""
=============================================================
BEHAVIOR-AWARE EXPLAINABLE FRAUD DETECTION SYSTEM
=============================================================
Dataset      : IEEE-CIS Fraud Detection
Algorithm    : XGBoost + SMOTE + Isolation Forest + SHAP
Validation   : Temporal 80/20 split
Risk Scoring : Hybrid (XGBoost 70% + Behavioral 20% + Anomaly 10%)
=============================================================
"""
import os
import gc
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)

from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import shap

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(
    os.environ.get("FRAUD_PROJECT_DIR",str(Path(__file__).parent))
)

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"

DATA_DIR.mkdir(parents=True,exist_ok=True)
MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)
PLOT_DIR.mkdir(parents=True,exist_ok=True)

# UTILITY FUNCTIONS

def print_section(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

def cleanup():
    gc.collect()

# DATASET DOWNLOAD

def check_dataset():
    print_section("DATASET DOWNLOAD")
    expected_files = ["train_transaction.csv","train_identity.csv"]
    if all((DATA_DIR / f).exists() for f in expected_files):
        print("Dataset already exists.")
        return True

    try:
        import kagglehub
        import shutil
        print("Downloading dataset via kagglehub...")
        path = kagglehub.dataset_download("lnasiri007/ieeecis-fraud-detection")
        print("Downloaded to:", path)

        # Copy CSVs into DATA_DIR
        for f in expected_files:
            for candidate in Path(path).rglob(f):
                shutil.copy(candidate, DATA_DIR / f)
                print(f"Copied {f} -> {DATA_DIR}")
                break

        if all((DATA_DIR / f).exists() for f in expected_files):
            print("Dataset ready.")
            return True

        print("CSVs not found in downloaded path:", path)
        return False

    except Exception as e:
        print("kagglehub download failed:", e)
        print("\nEnsure kagglehub is installed:  pip install kagglehub")
        print("And you are logged in:          kagglehub.login()")
        return False

# LOAD DATA

def load_data():
    print_section("LOADING DATA")
    transaction_file = (DATA_DIR /"train_transaction.csv")
    identity_file = (DATA_DIR /"train_identity.csv")

    if not transaction_file.exists():
        raise FileNotFoundError(f"Missing: {transaction_file}")

    if not identity_file.exists():
        raise FileNotFoundError(f"Missing: {identity_file}")
    print("Loading train_transaction.csv...")
    transaction = pd.read_csv(transaction_file,low_memory=False)
    print("Transaction shape:",transaction.shape)
    print("Loading train_identity.csv...")
    identity = pd.read_csv(identity_file,low_memory=False)
    print("Identity shape:",identity.shape)
    print("Merging transaction + identity...")

    df = transaction.merge(identity,on="TransactionID",how="left")
    print("Merged shape:",df.shape)
    del transaction
    del identity
    cleanup()

    return df

# EDA

def perform_eda(df):
    print_section("EXPLORATORY DATA ANALYSIS")
    target = "isFraud"
    fraud_count = int(df[target].sum())
    total_count = len(df)
    fraud_percentage = (
        fraud_count /
        total_count *
        100
    )

    print(f"Total transactions : {total_count:,}")
    print(f"Fraud transactions  : {fraud_count:,}")
    print(f"Fraud percentage    : {fraud_percentage:.2f}%")
    print("\nTarget distribution:")
    print(df[target].value_counts())
    # Target plot
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df,x=target)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.tight_layout()
    plt.savefig(PLOT_DIR /"fraud_distribution.png")
    plt.close()

    # Missing values
    missing = (
        df.isnull()
        .mean()
        .sort_values(
            ascending=False
        ) * 100
    )

    missing_df = pd.DataFrame({"missing_percentage": missing})
    missing_df.to_csv(REPORT_DIR /"missing_values.csv")
    print("\nTop missing-value columns:")
    print(missing_df.head(20))

# TIME FEATURES

def create_time_features(df):
    print_section("CREATING TIME FEATURES")
    seconds_per_day = (24 * 60 * 60)
    df["transaction_day"] = (df["TransactionDT"] // seconds_per_day)
    df["transaction_hour"] = ((df["TransactionDT"] % seconds_per_day) // 3600)
    df["transaction_weekday"] = (df["transaction_day"] % 7)
    df["is_night"] = (
        (df["transaction_hour"] < 6) | (df["transaction_hour"] >= 23)
    ).astype("int8")

    return df

# TRADITIONAL FEATURES

def create_traditional_features(df):
    print_section("CREATING TRADITIONAL FEATURES")
    if "TransactionAmt" in df.columns:
        df["log_transaction_amount"] = (
            np.log1p(
                df["TransactionAmt"]
                .clip(lower=0)
            )
        )
    card_columns = ["card1","card2","card3","card4","card5","card6"]
    existing_cards = [
        column
        for column in card_columns
        if column in df.columns
    ]

    if existing_cards:
        df["card_missing_count"] = (
            df[existing_cards]
            .isna()
            .sum(axis=1)
        )
    address_columns = ["addr1","addr2"]

    existing_addresses = [
        column
        for column in address_columns
        if column in df.columns
    ]
    if existing_addresses:

        df["address_missing_count"] = (
            df[existing_addresses]
            .isna()
            .sum(axis=1)
        )

    return df

# BEHAVIORAL FEATURES

def create_behavioral_features(df):
    print_section("CREATING BEHAVIOR-AWARE FEATURES")

    # Sort chronologically
    df = (
        df.sort_values("TransactionDT")
        .reset_index(
            drop=True
        )
    )

    # CARD KEY
    card_parts = [
        column
        for column in ["card1","card2","card3","card5","card6"]
        if column in df.columns
    ]

    if card_parts:
        df["card_key"] = (
            df[card_parts]
            .astype("string")
            .fillna("UNKNOWN")
            .agg(
                "_".join,
                axis=1
            )
        )
    else:
        df["card_key"] = (
            df["TransactionID"]
            .astype(str)
        )

    # PREVIOUS TRANSACTION AMOUNT

    df["previous_card_amount"] = (
        df.groupby("card_key")["TransactionAmt"]
        .shift(1)
    )

    # HISTORICAL MEAN
    # IMPORTANT:
    # shift(1) prevents current transaction leakage

    df["card_previous_mean_amount"] = (
        df.groupby("card_key")["TransactionAmt"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    # TRANSACTION COUNT

    df["card_previous_transaction_count"] = (
        df.groupby("card_key")
        .cumcount()
    )

    # AMOUNT DEVIATION

    df["amount_deviation_ratio"] = (
        df["TransactionAmt"] / (df["card_previous_mean_amount"] + 1e-6)
    )

    # DEVICE NOVELTY

    if "DeviceInfo" in df.columns:
        df["device_key"] = (
            df["DeviceInfo"]
            .astype("string")
            .fillna("UNKNOWN")
        )
        df["card_device_key"] = (
            df["card_key"].astype(str)
            + "_"
            + df["device_key"].astype(str)
        )
        df["card_device_previous_count"] = (
            df.groupby("card_device_key")
            .cumcount()
        )
        df["is_new_card_device"] = (
            df["card_device_previous_count"] == 0
        ).astype("int8")

    else:
        df["is_new_card_device"] = 0

    # TRANSACTION VELOCITY

    df["card_previous_transaction_time"] = (
        df.groupby("card_key")["TransactionDT"]
        .shift(1)
    )
    df[ "seconds_since_previous_card_transaction"] = (
        df["TransactionDT"] -
        df["card_previous_transaction_time"]
    )
    df["rapid_transaction"] = (
        df["seconds_since_previous_card_transaction"]
        .between(0,300)
        .fillna(False)
        .astype("int8")
    )
    df["very_rapid_transaction"] = (
        df["seconds_since_previous_card_transaction"]
        .between(0,60)
        .fillna(False)
        .astype("int8")
    )

    # BEHAVIORAL RISK

    df["amount_anomaly"] = (
        np.log1p(
            df["amount_deviation_ratio"]
            .clip(lower=0)
        )
        .clip(upper=5) / 5
    )
    df["new_card_behavior"] = (
        df["card_previous_transaction_count"] == 0
    ).astype("int8")
    df["velocity_behavior"] = (
        0.7 *
        df["rapid_transaction"] +
        0.3 *
        df["very_rapid_transaction"]
    )

    df["behavioral_risk_score"] = (
        0.35 *
        df["amount_anomaly"].fillna(0)
        +
        0.25 *
        df["is_new_card_device"]
        +
        0.20 *
        df["velocity_behavior"]
        +
        0.20 *
        df["is_night"]
    )

    df["behavioral_risk_score"] = (
        df["behavioral_risk_score"]
        .clip(0, 1)
    )
    print("Behavioral features created:")

    behavioral_features = ["amount_deviation_ratio","amount_anomaly","is_new_card_device","rapid_transaction","very_rapid_transaction","velocity_behavior","behavioral_risk_score"]
    for feature in behavioral_features:
        if feature in df.columns:
            print(f"  ✓ {feature}")
    return df


# PREPARE MODEL DATA


def prepare_features(df):

    print_section(
        "PREPARING MODEL FEATURES"
    )

    target = "isFraud"

    drop_columns = [
        target,
        "TransactionID",
        "TransactionDT",
        "card_key",
        "card_device_key",
        "device_key",
        "card_previous_transaction_time"
    ]

    drop_columns = [
        column
        for column in drop_columns
        if column in df.columns
    ]

    X = df.drop(
        columns=drop_columns
    )

    y = df[target].astype(
        int
    )


    # FREQUENCY ENCODING

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    print(
        "Categorical columns:",
        len(categorical_columns)
    )

    for column in categorical_columns:

        frequency = (
            X[column]
            .value_counts(
                dropna=False,
                normalize=True
            )
        )

        X[
            column + "_freq"
        ] = (
            X[column]
            .map(frequency)
            .fillna(0)
        )

    X = X.drop(
        columns=categorical_columns
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    X = X.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    for column in X.columns:

        if not pd.api.types.is_numeric_dtype(
            X[column]
        ):

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )
    # REMOVE EXTREMELY SPARSE FEATURES
   

    missing_ratio = (X.isna().mean())

    keep_columns = (
        missing_ratio[
            missing_ratio < 0.95
        ].index
    )

    X = X[keep_columns]
    print("Final feature shape:", X.shape)

    return X, y

# ============================================================
# TEMPORAL TRAIN / VALIDATION SPLIT
# ============================================================

def temporal_split(X, y, df):

    print_section("TEMPORAL TRAIN / VALIDATION SPLIT")

    order = np.argsort(
        df["TransactionDT"]
        .values
    )

    X = (
        X.iloc[order]
        .reset_index(
            drop=True
        )
    )

    y = (
        y.iloc[order]
        .reset_index(
            drop=True
        )
    )

    split_point = int(len(X) * 0.80)
    X_train = X.iloc[ :split_point].copy()
    X_valid = X.iloc[split_point: ].copy()
    y_train = y.iloc[ :split_point].copy()
    y_valid = y.iloc[split_point: ].copy()
    print("Training shape:", X_train.shape)
    print("Validation shape:", X_valid.shape)
    print("Training fraud rate:", f"{y_train.mean():.4f}")
    print("Validation fraud rate:", f"{y_valid.mean():.4f}")

    return (X_train, X_valid, y_train, y_valid)


# ============================================================
# IMPUTATION
# ============================================================

def impute_data(X_train, X_valid):

    print_section("MISSING VALUE IMPUTATION")
    imputer = SimpleImputer(strategy="median")
    X_train_processed = (imputer.fit_transform(X_train))
    X_valid_processed = (imputer.transform(X_valid))
    print( "Imputation completed.")

    return (imputer, X_train_processed, X_valid_processed)


# ============================================================
# XGBOOST BASELINE
# ============================================================

def train_baseline(X_train, y_train):

    print_section(
        "TRAINING BASELINE XGBOOST"
    )

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


# ============================================================
# SMOTE RESAMPLING
# ============================================================

def apply_smote(X_train, y_train):

    print_section("SMOTE RESAMPLING (imbalanced-learn)")
    print(f"Before SMOTE — Fraud: {y_train.sum():,}  Normal: {(y_train==0).sum():,}")
    smote = SMOTE(sampling_strategy=0.20, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After  SMOTE — Fraud: {y_resampled.sum():,}  Normal: {(y_resampled==0).sum():,}")

    return X_resampled, y_resampled


# ============================================================
# IMPROVED XGBOOST (trained on SMOTE data)
# ============================================================

def train_improved_model(X_train,y_train):

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


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_valid, y_valid, threshold=0.5):

    probability = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    metrics = {
        "ROC-AUC": roc_auc_score(y_valid, probability),
        "PR-AUC": average_precision_score(y_valid, probability),
        "Precision": precision_score(y_valid, prediction, zero_division=0),
        "Recall": recall_score(y_valid, prediction, zero_division=0),
        "F1": f1_score(y_valid, prediction, zero_division=0),
        "Brier Score": brier_score_loss(y_valid, probability)
    }

    return (metrics, probability, prediction)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(probabilities, y_valid):

    print_section("THRESHOLD OPTIMIZATION")
    results = []

    for threshold in np.arange(0.05, 0.96, 0.05):

        prediction = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score( y_valid, prediction, zero_division=0)
        recall = recall_score(y_valid, prediction, zero_division=0)
        f1 = f1_score(y_valid, prediction, zero_division=0)
        results.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})

    threshold_df = pd.DataFrame(
        results
    )

    best_row = (
        threshold_df
        .loc[
            threshold_df["f1"].idxmax()
        ]
    )

    best_threshold = float(
        best_row["threshold"]
    )

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Precision: "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_row['f1']:.4f}"
    )
    threshold_df.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    return (best_threshold, threshold_df)


# ============================================================
# ISOLATION FOREST
# ============================================================

def train_anomaly_model(X_train, y_train):

    print_section("TRAINING ANOMALY DETECTION MODEL")
    normal_indices = np.where(
        y_train.values == 0
    )[0]

    sample_size = min(100000, len(normal_indices))

    rng = np.random.default_rng(RANDOM_STATE)

    selected_indices = (rng.choice(normal_indices, size=sample_size, replace=False))
    print("Normal samples used:",sample_size)
    anomaly_model = IsolationForest(n_estimators=100, contamination="auto", random_state=RANDOM_STATE, n_jobs=-1)
    anomaly_model.fit(X_train[selected_indices])
    print("Isolation Forest trained.")
    return anomaly_model


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(anomaly_model, X_valid):

    raw_score = (
        -anomaly_model.score_samples(X_valid)
    )

    score_min = raw_score.min()
    score_max = raw_score.max()

    normalized_score = (
        raw_score -
        score_min
    ) / (
        score_max -
        score_min +
        1e-8
    )
    return np.clip(normalized_score, 0, 1)


# ============================================================
# BEHAVIOR SCORE
# ============================================================

def get_behavior_scores(df, X_valid):

    # The final validation rows correspond
    # to the final 20% chronological section.

    split_point = int(
        len(df) * 0.80
    )

    validation_df = df.iloc[
        split_point:
    ].copy()

    if (
        "behavioral_risk_score"
        in validation_df.columns
    ):

        scores = (
            validation_df[
                "behavioral_risk_score"
            ]
            .fillna(0)
            .values
        )

    else:
        scores = np.zeros(len(X_valid))

    return scores


# ============================================================
# FINAL HYBRID SCORE
# ============================================================

def calculate_final_risk(fraud_probability, behavior_score, anomaly_score):

    final_score = (
        0.70 *
        fraud_probability
        +
        0.20 *
        behavior_score
        +
        0.10 *
        anomaly_score
    )

    return np.clip(final_score, 0, 1)


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(score):

    if score < 0.30:
        return "LOW"

    elif score < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y_true, prediction, filename, title):

    cm = confusion_matrix(y_true, prediction)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename)
    plt.close()


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def perform_shap_analysis(model, X_valid, feature_names):

    print_section("SHAP EXPLAINABILITY")
    sample_size = min(500,len(X_valid))
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(X_valid), size=sample_size, replace=False)
    X_sample = X_valid[indices]
    print("Calculating SHAP values...")
    explainer = (shap.TreeExplainer(model))
    shap_values = (explainer.shap_values(X_sample))
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig( PLOT_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()
    print("SHAP analysis completed.")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(model, feature_names):

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance":
            model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    importance.to_csv(
        REPORT_DIR /
        "feature_importance.csv",
        index=False
    )

    top_features = (
        importance
        .head(20)
    )
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x="importance", y="feature")
    plt.title( "Top 20 Fraud Detection Features")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png")

    plt.close()

    return importance


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(baseline_model, improved_model, anomaly_model, imputer, threshold, feature_names):

    print_section("SAVING MODELS")
    joblib.dump( baseline_model, MODEL_DIR / "baseline_xgboost.pkl")
    joblib.dump(improved_model, MODEL_DIR / "behavior_aware_xgboost.pkl")
    joblib.dump(anomaly_model, MODEL_DIR / "isolation_forest.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    joblib.dump(threshold, MODEL_DIR / "optimal_threshold.pkl")
    joblib.dump(list(feature_names), MODEL_DIR / "feature_names.pkl")
    print("All models saved.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print_section("BEHAVIOR-AWARE FRAUD DETECTION SYSTEM")
    print("IEEE-CIS Fraud Detection")
    print("Project differentiator:")
    print("Behavior-aware dynamic risk scoring")

    if not check_dataset():
        return
    df = load_data()
    perform_eda(df)
    df = create_time_features(df)
    df = create_traditional_features(df)
    df = create_behavioral_features(df)
    X, y = prepare_features(df)
    (X_train,X_valid,y_train,y_valid) = temporal_split(X,y,df)
    (imputer,X_train_processed,X_valid_processed) = impute_data(X_train,X_valid)

    
    # SMOTE + BASELINE
    
    baseline_model = train_baseline( X_train_processed,y_train)
    (baseline_metrics,baseline_probability,baseline_prediction) = evaluate_model(baseline_model, X_valid_processed, y_valid)
    print_section( "BASELINE RESULTS")

    for metric, value in (baseline_metrics.items()):
        print(f"{metric:<15}: " f"{value:.5f}")

    save_confusion_matrix(y_valid, baseline_prediction,"baseline_confusion_matrix.png","Baseline XGBoost Confusion Matrix"
    )

       # SMOTE + IMPROVED MODEL
   
    X_train_smote, y_train_smote = apply_smote(X_train_processed,y_train)

    improved_model = train_improved_model(X_train_smote,y_train_smote)

    (improved_metrics,improved_probability, improved_prediction) = evaluate_model( improved_model, X_valid_processed, y_valid )

    print_section("BEHAVIOR-AWARE RESULTS")

    for metric, value in (improved_metrics.items()):
        print(f"{metric:<15}: "f"{value:.5f}")
    save_confusion_matrix(y_valid, improved_prediction, "behavior_confusion_matrix.png", "Behavior-Aware XGBoost Confusion Matrix")

    # THRESHOLD

    (best_threshold, threshold_df) = optimize_threshold(improved_probability, y_valid)

    # ANOMALY MODEL
    
    anomaly_model = train_anomaly_model(X_train_processed,y_train)
    anomaly_score = (calculate_anomaly_score(anomaly_model,X_valid_processed))

    # BEHAVIOR SCORE
    
    behavior_score = (get_behavior_scores(df,X_valid))

    # Safety check
    min_length = min(len(improved_probability),len(behavior_score),len(anomaly_score))
    improved_probability = (improved_probability[:min_length])

    behavior_score = (behavior_score[:min_length])

    anomaly_score = (anomaly_score[:min_length])
    y_valid_final = (y_valid.iloc[:min_length])

    
    # FINAL HYBRID

    final_risk_score = (calculate_final_risk( improved_probability, behavior_score,anomaly_score))

    final_prediction = ( final_risk_score >= best_threshold
    ).astype(int)

    # FINAL METRICS

    final_metrics = {

        "ROC-AUC": roc_auc_score(y_valid_final,final_risk_score),
        "PR-AUC": average_precision_score(y_valid_final,final_risk_score),
        "Precision": precision_score(y_valid_final,final_prediction, zero_division=0),
        "Recall": recall_score( y_valid_final,final_prediction, zero_division=0),
        "F1": f1_score( y_valid_final, final_prediction, zero_division=0),
        "Brier Score": brier_score_loss(y_valid_final,final_risk_score)
    }

    print_section("FINAL HYBRID MODEL RESULTS")

    for metric, value in (final_metrics.items()):
        print(f"{metric:<15}: "f"{value:.5f}")

    save_confusion_matrix(y_valid_final,final_prediction,"final_confusion_matrix.png","Final Hybrid Fraud Detection")

    print("\nClassification Report:")

    print(classification_report(y_valid_final,final_prediction,target_names=[ "Legitimate","Fraud"],zero_division=0))

    # MODEL COMPARISON
    
    comparison = pd.DataFrame({"Metric": list(baseline_metrics.keys()
        ),
        "Baseline_XGBoost": [
            baseline_metrics[m]
            for m in baseline_metrics
        ],

        "Behavior_Aware_XGBoost": [
            improved_metrics[m]
            for m in improved_metrics
        ],

        "Final_Hybrid": [
            final_metrics[m]
            for m in baseline_metrics
        ]
    })

    comparison["Behavior_Improvement_%"] = (
        (
            comparison["Behavior_Aware_XGBoost"]
            -
            comparison["Baseline_XGBoost"]
        )
        /
        comparison["Baseline_XGBoost"].replace(0,np.nan) 
	* 100
    )

    comparison.to_csv(REPORT_DIR /"model_comparison.csv",index=False)

    print_section("MODEL COMPARISON")

    print(comparison.to_string(index=False))

    
    # FEATURE IMPORTANCE

    importance = (save_feature_importance(improved_model, X.columns))

    # SHAP

    try:
        perform_shap_analysis(improved_model, X_valid_processed, X.columns)

    except Exception as error:
        print("SHAP analysis could not be completed:")
        print(error)

    # SAVE MODELS

    save_models(baseline_model, improved_model, anomaly_model, imputer, best_threshold, X.columns
    )

    
    # SAVE CONFIGURATION

    configuration = {

        "dataset":"lnasiri007/ieeecis-fraud-detection",

        "random_state":RANDOM_STATE,

        "optimal_threshold": best_threshold,

        "behavior_weights": {
            "amount_anomaly": 0.35,
            "device_novelty": 0.25,
            "velocity": 0.20,
            "night_behavior": 0.20
        },

        "final_risk_weights": {
            "xgboost": 0.70,
            "behavior": 0.20,
            "anomaly": 0.10
        }
    }

    with open(
        REPORT_DIR / "configuration.json", "w"
    ) as file:

        json.dump(
            configuration, file,indent=4)

    
    # FINAL SUMMARY

    print_section("PROJECT COMPLETED" )

    print("Dataset              : IEEE-CIS")
    print("Baseline             : XGBoost")
    print("Improved Model       : Behavior-Aware XGBoost + SMOTE")
    print("Anomaly Detection    : Isolation Forest")
    print("Explainability       : SHAP")
    print("Validation           : Temporal")
    print("Risk Scoring         : Dynamic Hybrid")

    print("\n" + "=" * 70)
    print("FINAL MODEL ACCURACY SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<30} {'ROC-AUC':>10} {'PR-AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 75)

    for label, metrics in [
        ("Baseline XGBoost",        baseline_metrics),
        ("Behavior-Aware XGBoost",  improved_metrics),
        ("Final Hybrid",            final_metrics),
    ]:
        print(
            f"{label:<30}"
            f" {metrics['ROC-AUC']*100:>9.2f}%"
            f" {metrics['PR-AUC']*100:>9.2f}%"
            f" {metrics['Precision']*100:>9.2f}%"
            f" {metrics['Recall']*100:>9.2f}%"
            f" {metrics['F1']*100:>9.2f}%"
        )

    print("-" * 75)
    print(f"\n{'Best ROC-AUC':<30}: {max(baseline_metrics['ROC-AUC'], improved_metrics['ROC-AUC'], final_metrics['ROC-AUC'])*100:.2f}%")
    print(f"{'Best F1 Score':<30}: {max(baseline_metrics['F1'], improved_metrics['F1'], final_metrics['F1'])*100:.2f}%")
    print(f"{'Best Precision':<30}: {max(baseline_metrics['Precision'], improved_metrics['Precision'], final_metrics['Precision'])*100:.2f}%")
    print(f"{'Best Recall':<30}: {max(baseline_metrics['Recall'], improved_metrics['Recall'], final_metrics['Recall'])*100:.2f}%")
    print(f"{'Optimal Threshold':<30}: {best_threshold:.2f}")
    print(f"{'Brier Score (Hybrid)':<30}: {final_metrics['Brier Score']:.5f} (lower is better)")

    # Overall accuracy verdict
    best_roc = max(baseline_metrics['ROC-AUC'], improved_metrics['ROC-AUC'], final_metrics['ROC-AUC'])
    best_f1  = max(baseline_metrics['F1'], improved_metrics['F1'], final_metrics['F1'])

    if best_roc >= 0.95:
        verdict = "EXCELLENT — Production-ready fraud detection."
    elif best_roc >= 0.90:
        verdict = "VERY GOOD — Strong fraud detection, suitable for deployment with monitoring."
    elif best_roc >= 0.85:
        verdict = "GOOD — Reliable detection, consider further tuning."
    else:
        verdict = "FAIR — Model needs improvement before deployment."

    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    print(f"  ROC-AUC : {best_roc*100:.2f}%")
    print(f"  F1 Score: {best_f1*100:.2f}%")
    print(f"  Verdict : {verdict}")
    print("=" * 70)
    print("Output directory:", BASE_DIR)
    print("Project completed successfully.")
    print("=" * 70)

#RUN PROGRAM

if __name__ == "__main__":

    main()