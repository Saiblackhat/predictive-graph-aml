"""
Preprocessing and Normalization Module for aml-ledger-trace (Phase 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


def preprocess_transactions(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    processed_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    txns = transactions_df.copy()
    accs = accounts_df.copy()

    # Deduplication
    txns = txns.drop_duplicates(subset=["transaction_id"])
    txns = txns.drop_duplicates(subset=["sender_id", "receiver_id", "timestamp", "amount"])

    # Drop nulls
    txns = txns.dropna(subset=["sender_id", "receiver_id", "amount", "timestamp"])
    accs = accs.dropna(subset=["account_id"])

    # String identifiers
    txns["sender_id"] = txns["sender_id"].astype(str).str.strip()
    txns["receiver_id"] = txns["receiver_id"].astype(str).str.strip()
    accs["account_id"] = accs["account_id"].astype(str).str.strip()

    # Filter invalid
    txns = txns[txns["amount"] > 0]
    txns = txns[txns["sender_id"] != txns["receiver_id"]]

    # Timestamp parsing
    if not pd.api.types.is_datetime64_any_dtype(txns["timestamp"]):
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    txns = txns.sort_values(by="timestamp").reset_index(drop=True)
    txns["txn_hour"] = txns["timestamp"].dt.hour
    txns["txn_day_of_week"] = txns["timestamp"].dt.dayofweek
    txns["txn_timestamp_unix"] = txns["timestamp"].astype("int64") // 10**9

    # Amount transformations
    txns["log_amount"] = np.log1p(txns["amount"])
    mean_amt = txns["amount"].mean()
    std_amt = txns["amount"].std()
    txns["amount_zscore"] = (txns["amount"] - mean_amt) / (std_amt if std_amt > 0 else 1.0)

    # Sync accounts
    all_nodes = set(txns["sender_id"]).union(set(txns["receiver_id"]))
    known_nodes = set(accs["account_id"])
    missing = all_nodes - known_nodes
    if missing:
        new_accs = pd.DataFrame(
            [{"account_id": a, "account_type": "INDIVIDUAL", "is_laundering": 0, "typology": "BENIGN"} for a in missing]
        )
        accs = pd.concat([accs, new_accs], ignore_index=True)

    accs = accs.drop_duplicates(subset=["account_id"]).reset_index(drop=True)

    p_dir = Path(processed_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    txns.to_parquet(p_dir / "processed_transactions.parquet", index=False)
    accs.to_parquet(p_dir / "processed_accounts.parquet", index=False)

    return txns, accs
