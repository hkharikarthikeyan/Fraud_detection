import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_DIR, PLOT_DIR, REPORT_DIR
from src.utils import print_section, cleanup

def load_data(nrows=None):
    print_section("LOADING DATA")
    transaction_file = (DATA_DIR / "train_transaction.csv")
    identity_file = (DATA_DIR / "train_identity.csv")

    if not transaction_file.exists():
        raise FileNotFoundError(f"Missing: {transaction_file}")

    if not identity_file.exists():
        raise FileNotFoundError(f"Missing: {identity_file}")
    if nrows:
        print(f"Loading train_transaction.csv (subset of {nrows} rows to prevent OOM)...")
    else:
        print("Loading train_transaction.csv (full dataset)...")
    transaction = pd.read_csv(transaction_file, low_memory=False, nrows=nrows)
    print("Transaction shape:", transaction.shape)
    print("Loading train_identity.csv...")
    identity = pd.read_csv(identity_file, low_memory=False)
    print("Identity shape:", identity.shape)
    print("Merging transaction + identity...")

    df = transaction.merge(identity, on="TransactionID", how="left")
    print("Merged shape:", df.shape)
    del transaction
    del identity
    cleanup()

    return df

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
    sns.countplot(data=df, x=target)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fraud_distribution.png")
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
    missing_df.to_csv(REPORT_DIR / "missing_values.csv")
    print("\nTop missing-value columns:")
    print(missing_df.head(20))

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

def create_traditional_features(df):
    print_section("CREATING TRADITIONAL FEATURES")
    if "TransactionAmt" in df.columns:
        df["log_transaction_amount"] = (
            np.log1p(
                df["TransactionAmt"]
                .clip(lower=0)
            )
        )
    card_columns = ["card1", "card2", "card3", "card4", "card5", "card6"]
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
    address_columns = ["addr1", "addr2"]

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
        for column in ["card1", "card2", "card3", "card5", "card6"]
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
    # shift(1) prevents current transaction leakage
    shifted_amt = df.groupby("card_key")["TransactionAmt"].shift(1)
    df["card_previous_mean_amount"] = (
        shifted_amt.groupby(df["card_key"]).cumsum() / 
        (shifted_amt.groupby(df["card_key"]).cumcount() + 1)
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
    df["seconds_since_previous_card_transaction"] = (
        df["TransactionDT"] -
        df["card_previous_transaction_time"]
    )
    df["rapid_transaction"] = (
        df["seconds_since_previous_card_transaction"]
        .between(0, 300)
        .fillna(False)
        .astype("int8")
    )
    df["very_rapid_transaction"] = (
        df["seconds_since_previous_card_transaction"]
        .between(0, 60)
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

    behavioral_features = ["amount_deviation_ratio", "amount_anomaly", "is_new_card_device", "rapid_transaction", "very_rapid_transaction", "velocity_behavior", "behavioral_risk_score"]
    for feature in behavioral_features:
        if feature in df.columns:
            print(f"  [OK] {feature}")
    return df

def prepare_features(df):
    print_section("PREPARING MODEL FEATURES")

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

    X = df.drop(columns=drop_columns)
    y = df[target].astype(int)

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

    print("Categorical columns:", len(categorical_columns))

    for column in categorical_columns:
        frequency = (
            X[column]
            .value_counts(
                dropna=False,
                normalize=True
            )
        )
        X[column + "_freq"] = (
            X[column]
            .map(frequency)
            .fillna(0)
        )

    X = X.drop(columns=categorical_columns)

    # NUMERIC CONVERSION
    X = X.replace([np.inf, -np.inf], np.nan)

    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            X[column] = pd.to_numeric(X[column], errors="coerce")
        # Downcast numerical columns to conserve memory
        if pd.api.types.is_float_dtype(X[column]):
            X[column] = X[column].astype(np.float32)
        elif pd.api.types.is_integer_dtype(X[column]):
            X[column] = X[column].astype(np.int32)

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
