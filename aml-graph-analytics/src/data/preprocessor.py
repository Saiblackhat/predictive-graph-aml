"""
Transaction Data Preprocessor.

Performs data quality checks, data cleaning, timestamp harmonization,
amount normalization, and saves canonical processed tables ready for graph
construction and downstream modeling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


def preprocess_data(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    processed_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean, harmonize, and normalize transaction and account datasets.

    Steps:
      1. Deduplicate records based on (transaction_id) or (sender, receiver, timestamp, amount).
      2. Handle missing or malformed values (drop or fill with sensible defaults).
      3. Parse and standardize ISO timestamps, extracting temporal helper features:
         - txn_hour, txn_day_of_week, txn_timestamp_unix.
      4. Filter out self-transfers or transactions with non-positive amounts.
      5. Normalize amounts:
         - log_amount: log1p(amount) to stabilize high-variance heavy-tailed distributions.
         - amount_zscore: standardized amount relative to overall transactions.
      6. Save clean parquet files to `processed_dir`.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - processed_txns: Cleaned transaction dataframe.
        - processed_accs: Cleaned account registry.
    """
    txns = transactions_df.copy()
    accs = accounts_df.copy()

    # 1. Deduplication
    initial_txn_count = len(txns)
    txns = txns.drop_duplicates(subset=["transaction_id"])
    txns = txns.drop_duplicates(subset=["sender_id", "receiver_id", "timestamp", "amount"])

    # 2. Missing Value Checks
    txns = txns.dropna(subset=["sender_id", "receiver_id", "amount", "timestamp"])
    accs = accs.dropna(subset=["account_id"])

    # 3. Clean string identifiers
    txns["sender_id"] = txns["sender_id"].astype(str).str.strip()
    txns["receiver_id"] = txns["receiver_id"].astype(str).str.strip()
    txns["transaction_type"] = txns["transaction_type"].astype(str).str.upper()
    accs["account_id"] = accs["account_id"].astype(str).str.strip()

    # 4. Filter invalid transfers (amounts <= 0 or sender == receiver)
    txns = txns[txns["amount"] > 0]
    txns = txns[txns["sender_id"] != txns["receiver_id"]]

    # 5. Timestamp parsing & temporal features
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    txns = txns.sort_values(by="timestamp").reset_index(drop=True)
    txns["txn_hour"] = txns["timestamp"].dt.hour
    txns["txn_day_of_week"] = txns["timestamp"].dt.dayofweek
    txns["txn_timestamp_unix"] = txns["timestamp"].astype("int64") // 10**9

    # 6. Amount normalization
    txns["log_amount"] = np.log1p(txns["amount"])
    mean_amt = txns["amount"].mean()
    std_amt = txns["amount"].std()
    txns["amount_zscore"] = (txns["amount"] - mean_amt) / (std_amt if std_amt > 0 else 1.0)

    # 7. Synchronize account registry
    all_active_accounts = set(txns["sender_id"]).union(set(txns["receiver_id"]))
    known_accounts = set(accs["account_id"])

    # If any accounts appear in transactions but are missing in registry, append them
    missing_accounts = all_active_accounts - known_accounts
    if missing_accounts:
        new_accs = pd.DataFrame(
            [
                {
                    "account_id": a,
                    "account_type": "INDIVIDUAL",
                    "is_laundering": 0,
                    "typology": "BENIGN",
                }
                for a in missing_accounts
            ]
        )
        accs = pd.concat([accs, new_accs], ignore_index=True)

    accs = accs.drop_duplicates(subset=["account_id"]).reset_index(drop=True)

    # 8. Persist to disk
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    txns.to_parquet(proc_path / "processed_transactions.parquet", index=False)
    accs.to_parquet(proc_path / "processed_accounts.parquet", index=False)

    print(
        f"[Preprocessor] Processed {len(txns)} transactions (retained {len(txns)/initial_txn_count:.2%}) "
        f"across {len(accs)} unique accounts."
    )
    return txns, accs
