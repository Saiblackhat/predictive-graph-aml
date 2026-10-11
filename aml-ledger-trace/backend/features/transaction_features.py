"""
Transaction and Flow Feature Engineering Module for aml-ledger-trace (Phase 4).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def extract_transaction_features(
    transactions_df: pd.DataFrame,
    structuring_lower: float = 45000.0,
    structuring_upper: float = 50000.0,
) -> pd.DataFrame:
    in_grouped = transactions_df.groupby("receiver_id")
    in_stats = in_grouped.agg(
        total_received_amount=("amount", "sum"),
        avg_received_amount=("amount", "mean"),
        std_received_amount=("amount", "std"),
        max_received_amount=("amount", "max"),
        received_txn_count=("amount", "count"),
        structuring_inflow_count=(
            "amount",
            lambda s: ((s >= structuring_lower) & (s < structuring_upper)).sum(),
        ),
    ).reset_index().rename(columns={"receiver_id": "account_id"})

    out_grouped = transactions_df.groupby("sender_id")
    out_stats = out_grouped.agg(
        total_sent_amount=("amount", "sum"),
        avg_sent_amount=("amount", "mean"),
        std_sent_amount=("amount", "std"),
        max_sent_amount=("amount", "max"),
        sent_txn_count=("amount", "count"),
        structuring_outflow_count=(
            "amount",
            lambda s: ((s >= structuring_lower) & (s < structuring_upper)).sum(),
        ),
    ).reset_index().rename(columns={"sender_id": "account_id"})

    txn_feats = pd.merge(in_stats, out_stats, on="account_id", how="outer").fillna(0.0)
    txn_feats["total_transacted_amount"] = txn_feats["total_received_amount"] + txn_feats["total_sent_amount"]
    txn_feats["net_flow_balance"] = txn_feats["total_received_amount"] - txn_feats["total_sent_amount"]
    txn_feats["total_txn_count"] = txn_feats["received_txn_count"] + txn_feats["sent_txn_count"]

    max_f = np.maximum(txn_feats["total_received_amount"], txn_feats["total_sent_amount"])
    txn_feats["flow_retention_ratio"] = np.where(max_f > 0, np.abs(txn_feats["net_flow_balance"]) / max_f, 1.0)
    txn_feats["total_structuring_count"] = txn_feats["structuring_inflow_count"] + txn_feats["structuring_outflow_count"]
    txn_feats["structuring_ratio"] = np.where(
        txn_feats["total_txn_count"] > 0,
        txn_feats["total_structuring_count"] / txn_feats["total_txn_count"],
        0.0,
    )

    return txn_feats
