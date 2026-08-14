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
    os.environ.get(
        "FRAUD_PROJECT_DIR",
        str(Path(__file__).parent)
    )
)

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def cleanup():

    gc.collect()


# ============================================================
# DATASET DOWNLOAD
# ============================================================

def check_dataset():

    print_section(
        "DATASET DOWNLOAD"
    )

    expected_files = [
        "train_transaction.csv",
        "train_identity.csv"
    ]

    if all((DATA_DIR / f).exists() for f in expected_files):
        print("Dataset already exists.")
        return True

    try:
        import kagglehub
        import shutil

        print("Downloading dataset via kagglehub...")

        path = kagglehub.dataset_download(
            "lnasiri007/ieeecis-fraud-detection"
        )

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

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print_section(
        "LOADING DATA"
    )

    transaction_file = (
        DATA_DIR /
        "train_transaction.csv"
    )

    identity_file = (
        DATA_DIR /
        "train_identity.csv"
    )

    if not transaction_file.exists():

        raise FileNotFoundError(
            f"Missing: {transaction_file}"
        )

    if not identity_file.exists():

        raise FileNotFoundError(
            f"Missing: {identity_file}"
        )

    print(
        "Loading train_transaction.csv..."
    )

    transaction = pd.read_csv(
        transaction_file,
        low_memory=False
    )

    print(
        "Transaction shape:",
        transaction.shape
    )

    print(
        "Loading train_identity.csv..."
    )

    identity = pd.read_csv(
        identity_file,
        low_memory=False
    )

    print(
        "Identity shape:",
        identity.shape
    )

    print(
        "Merging transaction + identity..."
    )

    df = transaction.merge(
        identity,
        on="TransactionID",
        how="left"
    )

    print(
        "Merged shape:",
        df.shape
    )

    del transaction
    del identity

    cleanup()

    return df


# ============================================================
# 8. EDA
# ============================================================

def perform_eda(df):

    print_section(
        "EXPLORATORY DATA ANALYSIS"
    )

    target = "isFraud"

    fraud_count = int(
        df[target].sum()
    )

    total_count = len(df)

    fraud_percentage = (
        fraud_count /
        total_count *
        100
    )

    print(
        f"Total transactions : {total_count:,}"
    )

    print(
        f"Fraud transactions  : {fraud_count:,}"
    )

    print(
        f"Fraud percentage    : {fraud_percentage:.2f}%"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df[target].value_counts()
    )

    # Target plot
    plt.figure(
        figsize=(7, 5)
    )

    sns.countplot(
        data=df,
        x=target
    )

    plt.title(
        "Fraud vs Non-Fraud Transactions"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        "fraud_distribution.png"
    )

    plt.close()

    # Missing values
    missing = (
        df.isnull()
        .mean()
        .sort_values(
            ascending=False
        ) * 100
    )

    missing_df = pd.DataFrame({
        "missing_percentage": missing
    })

    missing_df.to_csv(
        REPORT_DIR /
        "missing_values.csv"
    )

    print(
        "\nTop missing-value columns:"
    )

    print(
        missing_df.head(20)
    )


# ============================================================
# 9. TIME FEATURES
# ============================================================

def create_time_features(df):

    print_section(
        "CREATING TIME FEATURES"
    )

    seconds_per_day = (
        24 * 60 * 60
    )

    df["transaction_day"] = (
        df["TransactionDT"] //
        seconds_per_day
    )

    df["transaction_hour"] = (
        (
            df["TransactionDT"] %
            seconds_per_day
        ) // 3600
    )

    df["transaction_weekday"] = (
        df["transaction_day"] % 7
    )

    df["is_night"] = (
        (
            df["transaction_hour"] < 6
        ) |
        (
            df["transaction_hour"] >= 23
        )
    ).astype("int8")

    return df


# ============================================================
# 10. TRADITIONAL FEATURES
# ============================================================

def create_traditional_features(df):

    print_section(
        "CREATING TRADITIONAL FEATURES"
    )

    if "TransactionAmt" in df.columns:

        df["log_transaction_amount"] = (
            np.log1p(
                df["TransactionAmt"]
                .clip(lower=0)
            )
        )

    card_columns = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6"
    ]

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

    address_columns = [
        "addr1",
        "addr2"
    ]

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

# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

def create_behavioral_features(df):

    print_section(
        "CREATING BEHAVIOR-AWARE FEATURES"
    )

    # Sort chronologically
    df = (
        df.sort_values(
            "TransactionDT"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # CARD KEY
    # --------------------------------------------------------

    card_parts = [
        column
        for column in [
            "card1",
            "card2",
            "card3",
            "card5",
            "card6"
        ]
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

    # --------------------------------------------------------
    # PREVIOUS TRANSACTION AMOUNT
    # --------------------------------------------------------

    df["previous_card_amount"] = (
        df.groupby(
            "card_key"
        )["TransactionAmt"]
        .shift(1)
    )

    # --------------------------------------------------------
    # HISTORICAL MEAN
    # IMPORTANT:
    # shift(1) prevents current transaction leakage
    # --------------------------------------------------------

    df["card_previous_mean_amount"] = (
        df.groupby(
            "card_key"
        )["TransactionAmt"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    # --------------------------------------------------------
    # TRANSACTION COUNT
    # --------------------------------------------------------

    df["card_previous_transaction_count"] = (
        df.groupby(
            "card_key"
        )
        .cumcount()
    )

    # --------------------------------------------------------
    # AMOUNT DEVIATION
    # --------------------------------------------------------

    df["amount_deviation_ratio"] = (
        df["TransactionAmt"] /
        (
            df[
                "card_previous_mean_amount"
            ] + 1e-6
        )
    )

    # --------------------------------------------------------
    # DEVICE NOVELTY
    # --------------------------------------------------------

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

        df[
            "card_device_previous_count"
        ] = (
            df.groupby(
                "card_device_key"
            )
            .cumcount()
        )

        df["is_new_card_device"] = (
            df[
                "card_device_previous_count"
            ] == 0
        ).astype("int8")

    else:

        df["is_new_card_device"] = 0

    # --------------------------------------------------------
    # TRANSACTION VELOCITY
    # --------------------------------------------------------

    df[
        "card_previous_transaction_time"
    ] = (
        df.groupby(
            "card_key"
        )["TransactionDT"]
        .shift(1)
    )

    df[
        "seconds_since_previous_card_transaction"
    ] = (
        df["TransactionDT"] -
        df[
            "card_previous_transaction_time"
        ]
    )

    df["rapid_transaction"] = (
        df[
            "seconds_since_previous_card_transaction"
        ]
        .between(
            0,
            300
        )
        .fillna(False)
        .astype("int8")
    )

    df["very_rapid_transaction"] = (
        df[
            "seconds_since_previous_card_transaction"
        ]
        .between(
            0,
            60
        )
        .fillna(False)
        .astype("int8")
    )

