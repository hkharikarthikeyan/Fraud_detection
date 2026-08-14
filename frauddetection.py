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