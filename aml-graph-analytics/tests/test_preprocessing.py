"""
Unit tests for Data Preprocessing.
"""

import pandas as pd
import numpy as np
from src.data.preprocessor import preprocess_data


def test_preprocess_data_cleaning_and_normalization(tmp_path):
    # Construct raw DataFrame with messy inputs: duplicates, negative amount, self transfer
    raw_txns = pd.DataFrame(
        [
            {"transaction_id": "T1", "timestamp": "2026-01-01 10:00:00", "sender_id": "A", "receiver_id": "B", "amount": 100.0, "transaction_type": "wire", "is_laundering": 0},
            {"transaction_id": "T1", "timestamp": "2026-01-01 10:00:00", "sender_id": "A", "receiver_id": "B", "amount": 100.0, "transaction_type": "wire", "is_laundering": 0},  # duplicate
            {"transaction_id": "T2", "timestamp": "2026-01-01 11:00:00", "sender_id": "A", "receiver_id": "A", "amount": 50.0, "transaction_type": "transfer", "is_laundering": 0},   # self-transfer
            {"transaction_id": "T3", "timestamp": "2026-01-01 12:00:00", "sender_id": "B", "receiver_id": "C", "amount": -10.0, "transaction_type": "transfer", "is_laundering": 0},  # negative amount
            {"transaction_id": "T4", "timestamp": "2026-01-01 13:00:00", "sender_id": "C", "receiver_id": "D", "amount": 500.0, "transaction_type": "payment", "is_laundering": 1},
        ]
    )
    raw_accs = pd.DataFrame(
        [
            {"account_id": "A", "account_type": "INDIVIDUAL", "is_laundering": 0, "typology": "BENIGN"},
            {"account_id": "B", "account_type": "INDIVIDUAL", "is_laundering": 0, "typology": "BENIGN"},
        ]
    )

    proc_dir = str(tmp_path / "processed")
    proc_txns, proc_accs = preprocess_data(raw_txns, raw_accs, processed_dir=proc_dir)

    # Assert invalid transactions removed
    assert len(proc_txns) == 2  # Only T1 and T4 should remain
    assert "log_amount" in proc_txns.columns
    assert "amount_zscore" in proc_txns.columns
    assert "txn_hour" in proc_txns.columns

    # Check missing accounts (C and D) were automatically synchronized into account registry
    registered_ids = set(proc_accs["account_id"])
    assert {"A", "B", "C", "D"}.issubset(registered_ids)
