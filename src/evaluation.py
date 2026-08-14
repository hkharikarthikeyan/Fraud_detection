import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)
from src.config import PLOT_DIR, REPORT_DIR, RANDOM_STATE
from src.utils import print_section

def evaluate_model(model, X_valid, y_valid, threshold=0.5):
    probability = model.predict_proba(X_valid)[:, 1]
    prediction = (probability >= threshold).astype(int)

    metrics = {
        "ROC-AUC": roc_auc_score(y_valid, probability),
        "PR-AUC": average_precision_score(y_valid, probability),
        "Precision": precision_score(y_valid, prediction, zero_division=0),
        "Recall": recall_score(y_valid, prediction, zero_division=0),
        "F1": f1_score(y_valid, prediction, zero_division=0),
        "Brier Score": brier_score_loss(y_valid, probability)
    }

    return metrics, probability, prediction

def optimize_threshold(probabilities, y_valid):
    print_section("THRESHOLD OPTIMIZATION")
    results = []

    for threshold in np.arange(0.05, 0.96, 0.05):
        prediction = (probabilities >= threshold).astype(int)

        precision = precision_score(y_valid, prediction, zero_division=0)
        recall = recall_score(y_valid, prediction, zero_division=0)
        f1 = f1_score(y_valid, prediction, zero_division=0)
        results.append({
            "threshold": threshold, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1
        })

    threshold_df = pd.DataFrame(results)
    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall: {best_row['recall']:.4f}")
    print(f"F1: {best_row['f1']:.4f}")
    
    threshold_df.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    return best_threshold, threshold_df

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

def perform_shap_analysis(model, X_valid, feature_names):
    print_section("SHAP EXPLAINABILITY")
    sample_size = min(500, len(X_valid))
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(X_valid), size=sample_size, replace=False)
    X_sample = X_valid[indices]
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()
    print("SHAP analysis completed.")

def save_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    top_features = importance.head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x="importance", y="feature")
    plt.title("Top 20 Fraud Detection Features")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png")
    plt.close()

    return importance
