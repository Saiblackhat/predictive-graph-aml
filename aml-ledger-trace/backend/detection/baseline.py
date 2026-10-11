"""
Rule-Based Baseline Runner and Benchmarking Module for aml-ledger-trace (Phase 3).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import networkx as nx
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from backend.detection.rules import (
    detect_structuring,
    detect_fan_in_fan_out,
    detect_circular_transfers,
    detect_rapid_layering,
)


def run_rule_based_detection(
    transactions_df: pd.DataFrame,
    G: nx.DiGraph,
    accounts_df: pd.DataFrame,
    thresholds: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    s_df = detect_structuring(transactions_df, thresholds)
    f_df = detect_fan_in_fan_out(G, thresholds)
    c_df, cycles = detect_circular_transfers(G, thresholds)
    l_df = detect_rapid_layering(transactions_df, thresholds)

    all_flags = pd.concat([s_df, f_df, c_df, l_df], ignore_index=True)
    if all_flags.empty:
        agg_flags = pd.DataFrame(columns=["account_id", "triggered_rules", "rule_count", "rule_flagged"])
    else:
        agg_flags = (
            all_flags.groupby("account_id")["rule_name"]
            .apply(lambda r: sorted(list(set(r))))
            .reset_index(name="triggered_rules")
        )
        agg_flags["rule_count"] = agg_flags["triggered_rules"].apply(len)
        agg_flags["rule_flagged"] = 1

    eval_df = accounts_df[["account_id", "is_laundering"]].copy()
    eval_df = eval_df.merge(agg_flags, on="account_id", how="left")
    eval_df["rule_flagged"] = eval_df["rule_flagged"].fillna(0).astype(int)
    eval_df["rule_count"] = eval_df["rule_count"].fillna(0).astype(int)
    eval_df["triggered_rules"] = eval_df["triggered_rules"].apply(lambda x: x if isinstance(x, list) else [])

    y_true = eval_df["is_laundering"].values
    y_pred = eval_df["rule_flagged"].values

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "total_flagged": int(y_pred.sum()),
        "cycles_found": len(cycles),
    }

    return eval_df, metrics
