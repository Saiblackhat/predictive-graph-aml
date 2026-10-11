"""
Rule-Based Baseline Runner and Benchmarking Module.

Executes all 4 typology rules, aggregates flagged accounts, records triggering rules,
and computes baseline classification metrics (Precision, Recall, F1) against true ground truth.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import networkx as nx
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.detection.rules import (
    detect_structuring,
    detect_fan_in_fan_out,
    detect_circular_transfers,
    detect_rapid_layering_passthrough,
)


def run_rule_based_detection(
    transactions_df: pd.DataFrame,
    G: nx.DiGraph,
    accounts_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run all deterministic rules and evaluate benchmark performance.

    Parameters:
    -----------
    transactions_df : pd.DataFrame
        Transaction table.
    G : nx.DiGraph
        Directed transaction graph.
    accounts_df : pd.DataFrame
        Account registry with ground truth `is_laundering`.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]:
        - flagged_accounts_summary: Account-level table with aggregated flags and triggered rules.
        - metrics: Precision, Recall, F1, and Confusion Matrix.
    """
    # 1. Execute individual rule detectors
    struct_df = detect_structuring(transactions_df)
    fan_df = detect_fan_in_fan_out(G)
    cycle_df, detected_cycles = detect_circular_transfers(G)
    layer_df = detect_rapid_layering_passthrough(transactions_df)

    # 2. Combine flags
    all_flags = pd.concat(
        [
            struct_df[["account_id", "rule_name"]],
            fan_df[["account_id", "rule_name"]],
            cycle_df[["account_id", "rule_name"]],
            layer_df[["account_id", "rule_name"]],
        ],
        ignore_index=True,
    )

    if all_flags.empty:
        aggregated_flags = pd.DataFrame(columns=["account_id", "triggered_rules", "rule_count", "rule_flagged"])
    else:
        aggregated_flags = (
            all_flags.groupby("account_id")["rule_name"]
            .apply(lambda rules: sorted(list(set(rules))))
            .reset_index(name="triggered_rules")
        )
        aggregated_flags["rule_count"] = aggregated_flags["triggered_rules"].apply(len)
        aggregated_flags["rule_flagged"] = 1

    # 3. Merge with complete account universe
    eval_df = accounts_df[["account_id", "is_laundering"]].copy()
    eval_df = eval_df.merge(aggregated_flags, on="account_id", how="left")
    eval_df["rule_flagged"] = eval_df["rule_flagged"].fillna(0).astype(int)
    eval_df["rule_count"] = eval_df["rule_count"].fillna(0).astype(int)
    eval_df["triggered_rules"] = eval_df["triggered_rules"].apply(lambda x: x if isinstance(x, list) else [])

    # 4. Compute metrics
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
        "total_flagged_accounts": int(y_pred.sum()),
        "true_laundering_accounts": int(y_true.sum()),
        "detected_cycles_count": len(detected_cycles),
    }

    return eval_df, metrics
