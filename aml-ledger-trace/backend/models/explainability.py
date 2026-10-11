"""
SHAP TreeExplainer Module for aml-ledger-trace (Phase 4).
"""

from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import shap


class LedgerTraceExplainer:
    def __init__(self, xgb_model, feature_names: List[str]):
        self.model = xgb_model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, account_id: str, row: pd.Series, top_k: int = 5) -> Dict[str, Any]:
        X_df = pd.DataFrame([row[self.feature_names]])
        vals = self.explainer.shap_values(X_df)[0]
        base_val = float(self.explainer.expected_value)

        contribs = []
        for feat, val, shap_v in zip(self.feature_names, X_df.iloc[0], vals):
            contribs.append({"feature": feat, "value": float(val), "contribution": float(shap_v)})

        sorted_c = sorted(contribs, key=lambda x: abs(x["contribution"]), reverse=True)[:top_k]

        translations = {
            "cycle_participation_count": "Part of directed closed circular loop: raised risk",
            "total_structuring_count": "Multiple deposits clustered just below ₹50,000 threshold: raised risk",
            "structuring_ratio": "High proportion of near-limit transactions: raised risk",
            "burst_score": "High transaction burst velocity: raised risk",
            "dormant_then_active": "Activated after dormancy by high-value transfer: raised risk",
            "flow_retention_ratio": "Rapid pass-through with minimal capital retention: raised risk",
            "pagerank": "High centrality in transaction graph: raised risk",
        }

        reasons = []
        for c in sorted_c:
            feat = c["feature"]
            shap_v = c["contribution"]
            direction = "raised risk" if shap_v > 0 else "lowered risk"
            default_txt = f"{feat.replace('_', ' ').capitalize()}: {direction}"
            plain = translations.get(feat, default_txt)
            sign = f"+{shap_v:.2f}" if shap_v > 0 else f"{shap_v:.2f}"
            reasons.append(
                {
                    "feature": feat,
                    "contribution": shap_v,
                    "impact": "RAISES_RISK" if shap_v > 0 else "LOWERS_RISK",
                    "plain_language": f"{plain} ({sign})",
                }
            )

        return {"account_id": account_id, "base_value": base_val, "top_reasons": reasons}
