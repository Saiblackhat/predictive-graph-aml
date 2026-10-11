"""
Transactional & Behavioral Feature Engineering.

Aggregates entity transaction histories to capture financial flow behaviors:
  - Total sent/received amount, mean/std amount.
  - Net flow balance (inflow - outflow).
  - Transaction velocity (txns per day / hour).
  - High-risk cash deposit frequency.
  - Near-threshold transaction ratio (potential structuring indicators).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def extract_transaction_features(
    transactions_df: pd.DataFrame,
    structuring_lower: float = 8500.0,
    structuring_upper: float = 10000.0,
) -> pd.DataFrame:
    """
    Extract account-level transactional activity and behavioral flow features.

    Parameters:
    -----------
    transactions_df : pd.DataFrame
        Processed transactions.
    structuring_lower : float
        Lower bound for structuring candidate transactions.
    structuring_upper : float
        Upper bound for structuring candidate transactions.

    Returns:
    --------
    pd.DataFrame:
        DataFrame with transaction features per account_id.
    """
    # 1. Inflow features (as receiver)
    in_grouped = transactions_df.groupby("receiver_id")
    in_stats = in_grouped.agg(
        total_received_amount=("amount", "sum"),
        avg_received_amount=("amount", "mean"),
        std_received_amount=("amount", "std"),
        max_received_amount=("amount", "max"),
        received_txn_count=("amount", "count"),
        cash_deposit_inflow_count=("transaction_type", lambda s: (s == "CASH_DEPOSIT").sum()),
        structuring_inflow_count=(
            "amount",
            lambda s: ((s >= structuring_lower) & (s < structuring_upper)).sum(),
        ),
    ).reset_index().rename(columns={"receiver_id": "account_id"})

    # 2. Outflow features (as sender)
    out_grouped = transactions_df.groupby("sender_id")
    out_stats = out_grouped.agg(
        total_sent_amount=("amount", "sum"),
        avg_sent_amount=("amount", "mean"),
        std_sent_amount=("amount", "std"),
        max_sent_amount=("amount", "max"),
        sent_txn_count=("amount", "count"),
        wire_outflow_count=("transaction_type", lambda s: (s == "WIRE").sum()),
        structuring_outflow_count=(
            "amount",
            lambda s: ((s >= structuring_lower) & (s < structuring_upper)).sum(),
        ),
    ).reset_index().rename(columns={"sender_id": "account_id"})

    # Merge in and out stats
    txn_features = pd.merge(in_stats, out_stats, on="account_id", how="outer").fillna(0.0)

    # 3. Derived flow features
    txn_features["total_transacted_amount"] = (
        txn_features["total_received_amount"] + txn_features["total_sent_amount"]
    )
    txn_features["net_flow_balance"] = (
        txn_features["total_received_amount"] - txn_features["total_sent_amount"]
    )
    txn_features["total_txn_count"] = (
        txn_features["received_txn_count"] + txn_features["sent_txn_count"]
    )
    
    # Flow imbalance ratio
    max_flow = np.maximum(txn_features["total_received_amount"], txn_features["total_sent_amount"])
    txn_features["flow_retention_ratio"] = np.where(
        max_flow > 0,
        np.abs(txn_features["net_flow_balance"]) / max_flow,
        1.0,
    )

    # Structuring ratio
    txn_features["total_structuring_count"] = (
        txn_features["structuring_inflow_count"] + txn_features["structuring_outflow_count"]
    )
    txn_features["structuring_ratio"] = np.where(
        txn_features["total_txn_count"] > 0,
        txn_features["total_structuring_count"] / txn_features["total_txn_count"],
        0.0,
    )

    return txn_features
