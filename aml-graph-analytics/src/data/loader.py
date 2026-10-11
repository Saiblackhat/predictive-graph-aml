"""
Dataset Loader & Ingestion Module.

Supports:
1. Synthetic dataset generation and caching.
2. Loading real-world external transaction datasets (e.g. IBM Transactions for AML / Kaggle).
   The IBM AML dataset typically contains columns like:
     - Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency,
       Amount Paid, Payment Currency, Payment Format, Is Laundering.
   This module provides a unified interface mapping varying schemas into a standardized format:
     ['transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount', 'transaction_type', 'is_laundering']
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.data.generator import generate_synthetic_data


def load_dataset(
    dataset_type: str = "synthetic",
    raw_dir: str = "data/raw",
    ibm_file_path: Optional[str] = None,
    num_accounts: int = 5000,
    num_transactions: int = 50000,
    force_regenerate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unified loader for AML transaction data.

    Parameters:
    -----------
    dataset_type : str
        'synthetic' (default) or 'ibm_kaggle'.
    raw_dir : str
        Directory to store and load cached raw datasets.
    ibm_file_path : Optional[str]
        Path to IBM AML CSV/Parquet file if dataset_type == 'ibm_kaggle'.
    num_accounts : int
        Number of accounts for synthetic generation.
    num_transactions : int
        Number of transactions for synthetic generation.
    force_regenerate : bool
        If True, ignores existing cached raw files and regenerates synthetic data.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - transactions_df: Standardized transaction events.
        - accounts_df: Standardized account registry.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    txns_cache = raw_path / "raw_transactions.parquet"
    accs_cache = raw_path / "raw_accounts.parquet"

    if dataset_type == "synthetic":
        if not force_regenerate and txns_cache.exists() and accs_cache.exists():
            print(f"[DataLoader] Loading cached synthetic data from {raw_path}...")
            txns = pd.read_parquet(txns_cache)
            accs = pd.read_parquet(accs_cache)
            return txns, accs

        print(f"[DataLoader] Generating {num_transactions} synthetic transactions...")
        txns, accs = generate_synthetic_data(
            num_accounts=num_accounts,
            num_transactions=num_transactions,
            laundering_ratio=0.025,
            random_seed=42,
        )

        # Cache to parquet for fast downstream reloads
        txns.to_parquet(txns_cache, index=False)
        accs.to_parquet(accs_cache, index=False)
        print(f"[DataLoader] Successfully saved synthetic dataset to {raw_path}")
        return txns, accs

    elif dataset_type == "ibm_kaggle":
        if not ibm_file_path or not os.path.exists(ibm_file_path):
            raise FileNotFoundError(
                f"[DataLoader] Specified IBM Kaggle dataset file does not exist: {ibm_file_path}"
            )

        print(f"[DataLoader] Ingesting IBM AML dataset from {ibm_file_path}...")
        df = pd.read_csv(ibm_file_path)

        # Map typical IBM AML column schema to standardized format
        # Common schema: Timestamp, Account, Account.1, Amount Paid, Payment Format, Is Laundering
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
        
        if "typology" not in df.columns:
            df["typology"] = df["is_laundering"].apply(lambda x: "IBM_LAUNDERING" if x == 1 else "BENIGN")

        # Build account registry from senders and receivers
        all_accounts = list(set(df["sender_id"].dropna().unique()).union(set(df["receiver_id"].dropna().unique())))
        laundering_accounts = set(df[df["is_laundering"] == 1]["sender_id"]).union(
            set(df[df["is_laundering"] == 1]["receiver_id"])
        )
        
        acc_records = []
        for acc in all_accounts:
            is_laund = 1 if acc in laundering_accounts else 0
            acc_records.append({
                "account_id": str(acc),
                "account_type": "EXTERNAL",
                "is_laundering": is_laund,
                "typology": "IBM_LAUNDERING" if is_laund else "BENIGN",
            })
        accs_df = pd.DataFrame(acc_records)
        return df, accs_df

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'synthetic' or 'ibm_kaggle'.")
