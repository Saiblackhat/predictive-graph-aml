"""
SHAP TreeExplainer & Plain-Language Explanation Engine (Sprint 1).

Computes local feature contributions for any account using shap.TreeExplainer on XGBoost.
Produces:
  1. Signed SHAP numerical contribution values (positive pushes risk UP, negative pushes risk DOWN).
  2. Plain-language, investigator-friendly translation:
     - "Part of a 4-account directed cycle: raised risk (+0.42)"
     - "High structuring deposit frequency: raised risk (+0.38)"
     - "Balanced historical transaction volume: lowered risk (-0.15)"
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import shap


class AMLExplainabilityEngine:
    """
    SHAP-based explanation engine for anti-money laundering risk predictions.
    """

    FEATURE_TRANSLATIONS: Dict[str, Dict[str, str]] = {
        "cycle_participation_count": {
            "pos": "Participates in directed closed transfer cycle (round-tripping)",
            "neg": "No participation in circular loops",
        },
        "total_structuring_count": {
            "pos": "Frequent transactions clustered just beneath $10k reporting limit (structuring)",
            "neg": "Transactions not clustered near reporting threshold",
        },
        "structuring_ratio": {
            "pos": "High proportion of near-threshold structuring transactions",
            "neg": "Standard proportion of transactions",
        },
        "flow_retention_ratio": {
            "pos": "Rapid pass-through with minimal capital retention (layering behavior)",
            "neg": "Healthy capital retention balance",
        },
        "pagerank": {
            "pos": "High graph topological centrality in transaction network",
            "neg": "Peripheral topological positioning",
        },
        "betweenness_centrality": {
            "pos": "Acts as an intermediary bridge/broker between disconnected accounts",
            "neg": "Low intermediary brokerage centrality",
        },
        "burst_score": {
            "pos": "High transaction burst velocity over short hourly window",
            "neg": "Consistent, steady transaction flow",
        },
        "dormant_then_active": {
            "pos": "Account was dormant for >= 14 days then activated by large transfer surge",
            "neg": "Continuous regular activity with no abnormal activation",
        },
        "off_hours_ratio": {
            "pos": "Elevated off-hours transaction activity (10 PM - 6 AM)",
            "neg": "Transacts during standard banking business hours",
        },
        "in_degree": {
            "pos": "Unusually high number of incoming transfer counterparties (fan-in)",
            "neg": "Moderate counterparty in-degree",
        },
        "out_degree": {
            "pos": "Unusually high number of outgoing transfer counterparties (fan-out)",
            "neg": "Moderate counterparty out-degree",
        },
    }

    def __init__(self, xgb_model, feature_names: List[str]):
        self.model = xgb_model
        self.feature_names = feature_names
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    def explain_account(
        self,
        account_id: str,
        account_features: pd.Series,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate top SHAP positive and negative feature contributions in plain English.

        Parameters:
        -----------
        account_id : str
            The target account ID.
        account_features : pd.Series
            Feature values for the account.
        top_k : int
            Number of top contributing features to return.

        Returns:
        --------
        Dict[str, Any]:
            Dictionary containing base value, account score, and top feature contributions.
        """
        X_row = pd.DataFrame([account_features[self.feature_names]])
        shap_values = self.explainer.shap_values(X_row)[0]
        base_val = float(self.explainer.expected_value)

        # Pair features with SHAP values
        contributions = []
        for feat, val, shap_val in zip(self.feature_names, X_row.iloc[0], shap_values):
            contributions.append(
                {
                    "feature": feat,
                    "value": float(val),
                    "shap_value": float(shap_val),
                    "impact": "RAISES_RISK" if shap_val > 0 else "LOWERS_RISK",
                }
            )

        # Sort by absolute SHAP magnitude
        sorted_contribs = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)[:top_k]

        # Generate investigator-friendly plain text
        reasons = []
        for item in sorted_contribs:
            feat = item["feature"]
            shap_v = item["shap_value"]
            val = item["value"]

            trans_dict = self.FEATURE_TRANSLATIONS.get(feat, None)
            if trans_dict:
                desc = trans_dict["pos"] if shap_v > 0 else trans_dict["neg"]
            else:
                desc = f"Feature '{feat}' value ({val:,.2f})"

            direction_text = "raised risk" if shap_v > 0 else "lowered risk"
            sign_str = f"+{shap_v:.3f}" if shap_v > 0 else f"{shap_v:.3f}"
            plain_phrase = f"{desc}: {direction_text} ({sign_str})"

            reasons.append(
                {
                    "feature": feat,
                    "feature_value": val,
                    "contribution": shap_v,
                    "impact": item["impact"],
                    "plain_language": plain_phrase,
                }
            )

        return {
            "account_id": account_id,
            "base_value": base_val,
            "top_contributions": reasons,
        }
