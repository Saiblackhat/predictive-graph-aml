"""
Unsupervised Anomaly Detection Module (Sprint 1).

Trains an Isolation Forest model to detect statistical outliers across account behaviors.
Flags accounts with low supervised score (< 35) but high anomaly score as:
"Unusual, no known pattern" (rendered with neutral gray-violet #8B5CF6 badge).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_anomaly_detector(
    X: pd.DataFrame,
    contamination: float = 0.03,
    random_seed: int = 42,
    model_save_dir: str = "reports/models",
) -> IsolationForest:
    """
    Train an Isolation Forest unsupervised outlier detector on the account feature space.

    Parameters:
    -----------
    X : pd.DataFrame
        Account feature matrix.
    contamination : float
        Expected proportion of outliers (~3%).
    random_seed : int
        Seed for reproducibility.
    model_save_dir : str
        Directory to save the serialized model.

    Returns:
    --------
    IsolationForest:
        Fitted anomaly detection model.
    """
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_seed,
        n_jobs=-1,
    )
    iso.fit(X)

    save_path = Path(model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, save_path / "isolation_forest.joblib")

    return iso


def compute_anomaly_flags(
    iso_model: IsolationForest,
    X: pd.DataFrame,
    scored_accounts_df: pd.DataFrame,
    supervised_score_threshold: float = 35.0,
) -> pd.DataFrame:
    """
    Identify accounts with low supervised risk score but high anomaly severity:
    "Unusual, no known pattern".

    Parameters:
    -----------
    iso_model : IsolationForest
        Trained Isolation Forest model.
    X : pd.DataFrame
        Feature matrix.
    scored_accounts_df : pd.DataFrame
        Table containing 'account_id' and 'aml_risk_score'.
    supervised_score_threshold : float
        Upper bound on supervised score to qualify as an unknown anomaly.

    Returns:
    --------
    pd.DataFrame:
        Enhanced scored accounts DataFrame with anomaly metrics and pattern labels.
    """
    # Decision function: lower values mean more anomalous.
    raw_anomaly_scores = iso_model.decision_function(X)
    # Normalize anomaly score to 0-100 (where higher = more anomalous)
    min_s, max_s = raw_anomaly_scores.min(), raw_anomaly_scores.max()
    normalized_anomaly = 100.0 * (1.0 - (raw_anomaly_scores - min_s) / (max_s - min_s if max_s > min_s else 1.0))

    anomaly_preds = iso_model.predict(X)  # -1 for anomaly, +1 for inlier
    is_outlier = (anomaly_preds == -1).astype(int)

    df_anom = pd.DataFrame(
        {
            "account_id": X.index,
            "anomaly_score": np.round(normalized_anomaly, 1),
            "is_anomaly": is_outlier,
        }
    )

    merged = pd.merge(scored_accounts_df, df_anom, on="account_id", how="left")

    # Flag "Unusual, no known pattern"
    is_unknown_pattern = (merged["aml_risk_score"] < supervised_score_threshold) & (
        merged["is_anomaly"] == 1
    )
    merged["is_unknown_anomaly"] = is_unknown_pattern.astype(int)

    # If an account was previously marked BENIGN or low risk, update pattern designation
    merged["pattern_display"] = merged.apply(
        lambda r: "Unusual, no known pattern"
        if r["is_unknown_anomaly"] == 1 and (r.get("typology", "BENIGN") in ["BENIGN", "None", None])
        else r.get("typology", "BENIGN"),
        axis=1,
    )

    return merged
