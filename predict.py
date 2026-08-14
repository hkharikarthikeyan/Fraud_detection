from src.data_preprocessing import load_data
import os
import joblib
import pandas as pd
import numpy as np
from src.config import DATA_DIR, MODEL_DIR
from src.utils import print_section
from src.data_preprocessing import (
    create_time_features,
    create_traditional_features,
    create_behavioral_features,
    prepare_features
)
from src.models import (
    calculate_anomaly_score,
    get_behavior_scores,
    calculate_final_risk,
    risk_category
)

def main():
    print_section("RUNNING INFERENCE PIPELINE (predict.py)")
    
    # 1. Load trained artifacts
    artifacts = [
        "baseline_xgboost.pkl",
        "behavior_aware_xgboost.pkl",
        "isolation_forest.pkl",
        "imputer.pkl",
        "optimal_threshold.pkl",
        "feature_names.pkl"
    ]
    
    missing_artifacts = [f for f in artifacts if not (MODEL_DIR / f).exists()]
    if missing_artifacts:
        print(f"Error: Missing model artifacts: {missing_artifacts}")
        print("Please run train.py first to train and serialize the models.")
        return

    print("Loading models and configuration...")
    improved_model = joblib.load(MODEL_DIR / "behavior_aware_xgboost.pkl")
    anomaly_model = joblib.load(MODEL_DIR / "isolation_forest.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    best_threshold = joblib.load(MODEL_DIR / "optimal_threshold.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    print("All artifacts loaded successfully.")

    # 2. Ingest actual transactions
    try:
        df = load_data(nrows=100000)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    # 3. Pipeline pre-processing
    print("Running feature engineering pipeline on full dataset for correct context...")
    df = create_time_features(df)
    df = create_traditional_features(df)
    df = create_behavioral_features(df)
    
    X_all, y_all = prepare_features(df)
    
    # Take a 5000-row slice from the validation split (last 20%) for testing prediction
    split_point = int(len(df) * 0.80)
    slice_start = split_point
    slice_end = min(split_point + 5000, len(df))
    
    print(f"Slicing {slice_end - slice_start} actual transactions from validation split for prediction...")
    X_eval = X_all.iloc[slice_start:slice_end].copy()
    y_true = y_all.iloc[slice_start:slice_end].values
    has_labels = True
    
    # Align features with expected trained columns
    X_eval_aligned = pd.DataFrame(index=X_eval.index)
    for col in feature_names:
        if col in X_eval.columns:
            X_eval_aligned[col] = X_eval[col]
        else:
            X_eval_aligned[col] = 0.0
            
    X_eval = X_eval_aligned[feature_names]
    
    # Impute missing values using the trained imputer
    X_eval_processed = imputer.transform(X_eval)

    # 4. Predict
    print("Running predictions...")
    # Probability from XGBoost
    xgb_probs = improved_model.predict_proba(X_eval_processed)[:, 1]
    
    # Anomaly score from Isolation Forest
    anomaly_scores = calculate_anomaly_score(anomaly_model, X_eval_processed)
    
    # Behavioral score (aligned with validation slice)
    behavior_scores = df["behavioral_risk_score"].iloc[slice_start:slice_end].fillna(0).values
    
    # Align lengths
    min_len = min(len(xgb_probs), len(behavior_scores), len(anomaly_scores))
    xgb_probs = xgb_probs[:min_len]
    behavior_scores = behavior_scores[:min_len]
    anomaly_scores = anomaly_scores[:min_len]
    
    # Final risk score calculation
    final_risk_scores = calculate_final_risk(xgb_probs, behavior_scores, anomaly_scores)
    predictions = (final_risk_scores >= best_threshold).astype(int)

    # 5. Output results
    results_df = pd.DataFrame({
        "TransactionID": df["TransactionID"].iloc[slice_start:slice_end].iloc[:min_len].values,
        "TransactionAmt": df["TransactionAmt"].iloc[slice_start:slice_end].iloc[:min_len].values,
        "XGBoost_Prob": xgb_probs,
        "Anomaly_Score": anomaly_scores,
        "Behavior_Score": behavior_scores,
        "Hybrid_Risk_Score": final_risk_scores,
        "Prediction": predictions
    })
    
    results_df["Risk_Category"] = results_df["Hybrid_Risk_Score"].apply(risk_category)
    
    if has_labels:
        results_df["Actual_Fraud"] = y_true[:min_len]
        
    print("\nInference Results Sample (Top 10):")
    print(results_df.head(10).to_string(index=False))
    
    # Summary of findings
    fraud_detected = int(results_df["Prediction"].sum())
    print("\n" + "=" * 50)
    print("INFERENCE RUN SUMMARY")
    print("=" * 50)
    print(f"Total Transactions Processed : {min_len}")
    print(f"Fraudulent Flags Raised      : {fraud_detected} ({fraud_detected/min_len*100:.2f}%)")
    print(f"Risk Levels Breakdown:")
    print(results_df["Risk_Category"].value_counts().to_string())
    print("=" * 50)

if __name__ == "__main__":
    main()
