"""
Account Calibrated Risk Scorer.

Transforms model output probabilities into an interpretable 0–100 AML Risk Score:
  - Low Risk (0 - 25)
  - Medium Risk (26 - 50)
  - High Risk (51 - 75)
  - Critical Risk (76 - 100)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def compute_account_risk_scores(
    model,
    X: pd.DataFrame,
    accounts_df: pd.DataFrame,
    rules_eval_df: pd.DataFrame,
    save_path: str = "data/processed/account_risk_scores.parquet",
) -> pd.DataFrame:
    """
    Score all accounts in the universe on a 0–100 scale and assign risk tiers.

    Parameters:
    -----------
    model : trained classifier
        Fitted XGBoost or Random Forest model.
    X : pd.DataFrame
        Account feature matrix.
    accounts_df : pd.DataFrame
        Account registry.
    rules_eval_df : pd.DataFrame
        Output of rule baseline with triggered rules.
    save_path : str
        Target destination for scored account database.

    Returns:
    --------
    pd.DataFrame:
        Table of scored accounts with risk tiers, triggered rules, and metrics.
    """
    probabilities = model.predict_proba(X)[:, 1]
    
    # Scale probabilities to 0-100 risk score
    raw_scores = np.round(probabilities * 100.0, 1)

    scored_df = pd.DataFrame(
        {
            "account_id": X.index,
            "aml_risk_score": raw_scores,
            "raw_probability": np.round(probabilities, 4),
        }
    )

    # Risk Tier assignment
    def _assign_tier(score: float) -> str:
        if score >= 75.0:
            return "CRITICAL"
        elif score >= 50.0:
            return "HIGH"
        elif score >= 25.0:
            return "MEDIUM"
        else:
            return "LOW"

    scored_df["risk_tier"] = scored_df["aml_risk_score"].apply(_assign_tier)

    # Merge with accounts metadata & triggered rules
    merged = pd.merge(scored_df, accounts_df, on="account_id", how="left")
    merged = pd.merge(
        merged,
        rules_eval_df[["account_id", "triggered_rules", "rule_count", "rule_flagged"]],
        on="account_id",
        how="left",
    )
    merged["rule_flagged"] = merged["rule_flagged"].fillna(0).astype(int)
    merged["rule_count"] = merged["rule_count"].fillna(0).astype(int)

    # Sort descending by risk score
    merged = merged.sort_values(by="aml_risk_score", ascending=False).reset_index(drop=True)

    # Persist
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(p, index=False)
    # Also save a CSV for quick manual inspection or downstream reporting
    merged.to_csv(p.with_suffix(".csv"), index=False)

    return merged
