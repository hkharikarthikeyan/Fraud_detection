import json
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    classification_report
)

from src.config import (
    RANDOM_STATE,
    BASE_DIR,
    MODEL_DIR,
    REPORT_DIR
)
from src.utils import (
    print_section,
    check_dataset
)
from src.data_preprocessing import (
    load_data,
    perform_eda,
    create_time_features,
    create_traditional_features,
    create_behavioral_features,
    prepare_features
)
from src.models import (
    temporal_split,
    impute_data,
    train_baseline,
    apply_smote,
    train_improved_model,
    train_anomaly_model,
    calculate_anomaly_score,
    get_behavior_scores,
    calculate_final_risk
)
from src.evaluation import (
    evaluate_model,
    optimize_threshold,
    save_confusion_matrix,
    perform_shap_analysis,
    save_feature_importance
)

warnings.filterwarnings("ignore")

def save_models(baseline_model, improved_model, anomaly_model, imputer, threshold, feature_names):
    print_section("SAVING MODELS")
    joblib.dump(baseline_model, MODEL_DIR / "baseline_xgboost.pkl")
    joblib.dump(improved_model, MODEL_DIR / "behavior_aware_xgboost.pkl")
    joblib.dump(anomaly_model, MODEL_DIR / "isolation_forest.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    joblib.dump(threshold, MODEL_DIR / "optimal_threshold.pkl")
    joblib.dump(list(feature_names), MODEL_DIR / "feature_names.pkl")
    print("All models saved.")

def main():
    print_section("BEHAVIOR-AWARE FRAUD DETECTION SYSTEM")
    print("IEEE-CIS Fraud Detection")
    print("Project differentiator: Behavior-aware dynamic risk scoring")

    if not check_dataset():
        return
        
    df = load_data(nrows=100000)
    perform_eda(df)
    
    df = create_time_features(df)
    df = create_traditional_features(df)
    df = create_behavioral_features(df)
    
    X, y = prepare_features(df)
    X_train, X_valid, y_train, y_valid = temporal_split(X, y, df)
    imputer, X_train_processed, X_valid_processed = impute_data(X_train, X_valid)

    # 1. Baseline Model
    baseline_model = train_baseline(X_train_processed, y_train)
    baseline_metrics, baseline_probability, baseline_prediction = evaluate_model(
        baseline_model, X_valid_processed, y_valid
    )
    print_section("BASELINE RESULTS")
    for metric, value in baseline_metrics.items():
        print(f"{metric:<15}: {value:.5f}")
    save_confusion_matrix(
        y_valid, baseline_prediction, "baseline_confusion_matrix.png", "Baseline XGBoost Confusion Matrix"
    )

    # 2. SMOTE + Improved Model
    X_train_smote, y_train_smote = apply_smote(X_train_processed, y_train)
    improved_model = train_improved_model(X_train_smote, y_train_smote)
    improved_metrics, improved_probability, improved_prediction = evaluate_model(
        improved_model, X_valid_processed, y_valid
    )
    print_section("BEHAVIOR-AWARE RESULTS")
    for metric, value in improved_metrics.items():
        print(f"{metric:<15}: {value:.5f}")
    save_confusion_matrix(
        y_valid, improved_prediction, "behavior_confusion_matrix.png", "Behavior-Aware XGBoost Confusion Matrix"
    )

    # 3. Threshold Optimization
    best_threshold, threshold_df = optimize_threshold(improved_probability, y_valid)

    # 4. Anomaly Model (Isolation Forest)
    anomaly_model = train_anomaly_model(X_train_processed, y_train)
    anomaly_score = calculate_anomaly_score(anomaly_model, X_valid_processed)

    # 5. Behavior Score
    behavior_score = get_behavior_scores(df, X_valid)

    # Alignment safety check
    min_length = min(len(improved_probability), len(behavior_score), len(anomaly_score))
    improved_probability = improved_probability[:min_length]
    behavior_score = behavior_score[:min_length]
    anomaly_score = anomaly_score[:min_length]
    y_valid_final = y_valid.iloc[:min_length]

    # 6. Final Hybrid Risk Score
    final_risk_score = calculate_final_risk(improved_probability, behavior_score, anomaly_score)
    final_prediction = (final_risk_score >= best_threshold).astype(int)

    # Final Metrics
    final_metrics = {
        "ROC-AUC": roc_auc_score(y_valid_final, final_risk_score),
        "PR-AUC": average_precision_score(y_valid_final, final_risk_score),
        "Precision": precision_score(y_valid_final, final_prediction, zero_division=0),
        "Recall": recall_score(y_valid_final, final_prediction, zero_division=0),
        "F1": f1_score(y_valid_final, final_prediction, zero_division=0),
        "Brier Score": brier_score_loss(y_valid_final, final_risk_score)
    }

    print_section("FINAL HYBRID MODEL RESULTS")
    for metric, value in final_metrics.items():
        print(f"{metric:<15}: {value:.5f}")
    save_confusion_matrix(
        y_valid_final, final_prediction, "final_confusion_matrix.png", "Final Hybrid Fraud Detection"
    )

    print("\nClassification Report:")
    print(classification_report(y_valid_final, final_prediction, target_names=["Legitimate", "Fraud"], zero_division=0))

    # Model comparison report
    comparison = pd.DataFrame({
        "Metric": list(baseline_metrics.keys()),
        "Baseline_XGBoost": [baseline_metrics[m] for m in baseline_metrics],
        "Behavior_Aware_XGBoost": [improved_metrics[m] for m in improved_metrics],
        "Final_Hybrid": [final_metrics[m] for m in baseline_metrics]
    })
    comparison["Behavior_Improvement_%"] = (
        (comparison["Behavior_Aware_XGBoost"] - comparison["Baseline_XGBoost"])
        / comparison["Baseline_XGBoost"].replace(0, np.nan)
        * 100
    )
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    print_section("MODEL COMPARISON")
    print(comparison.to_string(index=False))

    # Save feature importance
    importance = save_feature_importance(improved_model, X.columns)

    # SHAP Explainability
    try:
        perform_shap_analysis(improved_model, X_valid_processed, X.columns)
    except Exception as error:
        print("SHAP analysis could not be completed:")
        print(error)

    # Save final artifacts
    save_models(baseline_model, improved_model, anomaly_model, imputer, best_threshold, X.columns)

    configuration = {
        "dataset": "lnasiri007/ieeecis-fraud-detection",
        "random_state": RANDOM_STATE,
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

    with open(REPORT_DIR / "configuration.json", "w") as file:
        json.dump(configuration, file, indent=4)

    # Final summary output
    print_section("PROJECT COMPLETED")
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
        ("Baseline XGBoost", baseline_metrics),
        ("Behavior-Aware XGBoost", improved_metrics),
        ("Final Hybrid", final_metrics),
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
    
    best_roc = max(baseline_metrics['ROC-AUC'], improved_metrics['ROC-AUC'], final_metrics['ROC-AUC'])
    best_f1 = max(baseline_metrics['F1'], improved_metrics['F1'], final_metrics['F1'])

    print(f"\n{'Best ROC-AUC':<30}: {best_roc*100:.2f}%")
    print(f"{'Best F1 Score':<30}: {best_f1*100:.2f}%")
    print(f"{'Best Precision':<30}: {max(baseline_metrics['Precision'], improved_metrics['Precision'], final_metrics['Precision'])*100:.2f}%")
    print(f"{'Best Recall':<30}: {max(baseline_metrics['Recall'], improved_metrics['Recall'], final_metrics['Recall'])*100:.2f}%")
    print(f"{'Optimal Threshold':<30}: {best_threshold:.2f}")
    print(f"{'Brier Score (Hybrid)':<30}: {final_metrics['Brier Score']:.5f} (lower is better)")

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

if __name__ == "__main__":
    main()
