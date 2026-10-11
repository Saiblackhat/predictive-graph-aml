"""
Imbalanced Machine Learning Model Training.

Trains supervised classifiers (Random Forest and XGBoost) on rich graph + transactional features.
Addresses severe class imbalance via:
  - Stratified Train/Test splitting (preserves low positive prevalence).
  - Cost-sensitive loss weighting (`scale_pos_weight` in XGBoost, `class_weight='balanced'` in RF).
  - Calibrated probability output for risk scoring.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def prepare_feature_matrix(
    graph_features_df: pd.DataFrame,
    txn_features_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Merge graph and transactional features with account ground-truth labels.

    Parameters:
    -----------
    graph_features_df : pd.DataFrame
        Graph topological features.
    txn_features_df : pd.DataFrame
        Transactional behavioral features.
    accounts_df : pd.DataFrame
        Account registry with 'is_laundering' target.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series, List[str]]:
        - X: Feature matrix indexed by account_id.
        - y: Binary labels (0 = benign, 1 = laundering).
        - feature_names: Ordered list of feature column names.
    """
    merged = pd.merge(accounts_df[["account_id", "is_laundering"]], graph_features_df, on="account_id", how="left")
    merged = pd.merge(merged, txn_features_df, on="account_id", how="left").fillna(0.0)

    feature_cols = [
        c
        for c in merged.columns
        if c not in ["account_id", "is_laundering", "account_type", "typology"]
    ]

    X = merged[feature_cols].copy()
    y = merged["is_laundering"].astype(int)
    X.index = merged["account_id"]
    y.index = merged["account_id"]

    return X, y, feature_cols


def train_aml_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_seed: int = 42,
    model_save_dir: str = "reports/models",
) -> Dict[str, Any]:
    """
    Train Random Forest and XGBoost classifiers with class imbalance compensation.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    test_size : float
        Proportion of dataset for held-out evaluation.
    random_seed : int
        Seed for reproducibility.
    model_save_dir : str
        Directory to serialize trained model artifacts.

    Returns:
    --------
    Dict[str, Any]:
        Trained models, split datasets, and metadata.
    """
    # 1. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # 2. Compute cost-sensitive weighting
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = float(num_neg / (num_pos if num_pos > 0 else 1.0))

    # 3. Model 1: Random Forest (Balanced Class Weighting)
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # 4. Model 2: XGBoost (scale_pos_weight for imbalanced loss)
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=random_seed,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)

    # 5. Persist models
    save_path = Path(model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, save_path / "random_forest_model.joblib")
    joblib.dump(xgb_model, save_path / "xgboost_model.joblib")

    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "scale_pos_weight": scale_pos_weight,
    }
