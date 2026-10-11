"""
Dataset Loader & Ingestion Module for aml-ledger-trace.
Supports caching and loading external Kaggle/IBM datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from backend.data.generator import generate_synthetic_transactions


def load_dataset(
    dataset_type: str = "synthetic",
    raw_dir: str = "data/raw",
    ibm_file_path: Optional[str] = None,
    num_accounts: int = 5000,
    num_transactions: int = 50000,
    force_regenerate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    txns_cache = raw_path / "raw_transactions.parquet"
    accs_cache = raw_path / "raw_accounts.parquet"

    if dataset_type == "synthetic":
        if not force_regenerate and txns_cache.exists() and accs_cache.exists():
            return pd.read_parquet(txns_cache), pd.read_parquet(accs_cache)

        txns, accs = generate_synthetic_transactions(
            num_accounts=num_accounts,
            num_transactions=num_transactions,
            laundering_ratio=0.025,
            random_seed=42,
        )
        txns.to_parquet(txns_cache, index=False)
        accs.to_parquet(accs_cache, index=False)
        return txns, accs

    elif dataset_type == "ibm_kaggle":
        if not ibm_file_path or not os.path.exists(ibm_file_path):
            raise FileNotFoundError(f"IBM Kaggle dataset file not found: {ibm_file_path}")
        df = pd.read_csv(ibm_file_path)
        col_map = {
            "Timestamp": "timestamp",
            "Account": "sender_id",
            "Account.1": "receiver_id",
            "Amount Paid": "amount",
            "Payment Format": "transaction_type",
            "Is Laundering": "is_laundering",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "transaction_id" not in df.columns:
            df["transaction_id"] = [f"TXN_IBM_{i:07d}" for i in range(len(df))]
        all_accounts = list(set(df["sender_id"].dropna().unique()).union(set(df["receiver_id"].dropna().unique())))
        accs_df = pd.DataFrame([{"account_id": str(a), "account_type": "EXTERNAL", "is_laundering": 0, "typology": "BENIGN"} for a in all_accounts])
        return df, accs_df
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
