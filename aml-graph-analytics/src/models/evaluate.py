"""
Model Evaluation and Baseline Benchmark Module.

Evaluates models using financial crime surveillance standards:
  - Precision, Recall, F1
  - Precision-Recall Area Under Curve (PR-AUC)
  - Confusion Matrix
  - Feature Importance Rankings & Visualization
  - Rigorous side-by-side comparison with the Rule-Based Baseline
"""

from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


def evaluate_models_and_baseline(
    training_artifacts: Dict[str, Any],
    rule_baseline_metrics: Dict[str, Any],
    output_report_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Evaluate Random Forest, XGBoost, and Rule-Based Baseline on the held-out test split.

    Parameters:
    -----------
    training_artifacts : Dict[str, Any]
        Output of `train_aml_models` with models and test sets.
    rule_baseline_metrics : Dict[str, Any]
        Performance metrics from the deterministic rule baseline.
    output_report_dir : str
        Directory to save comparative charts and markdown logs.

    Returns:
    --------
    Dict[str, Any]:
        Comparison table and comprehensive performance metrics.
    """
    report_path = Path(output_report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    X_test = training_artifacts["X_test"]
    y_test = training_artifacts["y_test"]
    rf = training_artifacts["rf_model"]
    xgb = training_artifacts["xgb_model"]
    feat_names = training_artifacts["feature_names"]

    # 1. Predictions & Probabilities
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    # 2. Performance Metric Computations
    def _calc_metrics(y_true, y_pred, y_prob):
        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    rf_metrics = _calc_metrics(y_test, rf_pred, rf_prob)
    xgb_metrics = _calc_metrics(y_test, xgb_pred, xgb_prob)

    # 3. Compile Comparative Summary Table
    comparison_table = pd.DataFrame(
        [
            {
                "Model / System": "Rule-Based Baseline",
                "Precision": rule_baseline_metrics["precision"],
                "Recall": rule_baseline_metrics["recall"],
                "F1 Score": rule_baseline_metrics["f1_score"],
                "PR-AUC": float("nan"),
            },
            {
                "Model / System": "Random Forest (Graph+Txn)",
                "Precision": rf_metrics["precision"],
                "Recall": rf_metrics["recall"],
                "F1 Score": rf_metrics["f1_score"],
                "PR-AUC": rf_metrics["pr_auc"],
            },
            {
                "Model / System": "XGBoost (Graph+Txn)",
                "Precision": xgb_metrics["precision"],
                "Recall": xgb_metrics["recall"],
                "F1 Score": xgb_metrics["f1_score"],
                "PR-AUC": xgb_metrics["pr_auc"],
            },
        ]
    )

    # 4. Feature Importance Extraction & Plot
    xgb_importances = xgb.feature_importances_
    feat_imp_df = pd.DataFrame(
        {"feature": feat_names, "importance": xgb_importances}
    ).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_imp_df["feature"], feat_imp_df["importance"], color="#1f77b4")
    ax.set_title("XGBoost AML Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontsize=12)
    plt.tight_layout()
    chart_path = report_path / "feature_importance.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)

    return {
        "comparison_table": comparison_table,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "feature_importance": feat_imp_df.sort_values(by="importance", ascending=False).to_dict(orient="records"),
        "chart_path": str(chart_path),
    }
