"""
Unit tests for Synthetic Transaction Generator and Data Loader.
"""

import pytest
import pandas as pd
from src.data.generator import generate_synthetic_data
from src.data.loader import load_dataset


def test_generate_synthetic_data_shapes_and_columns():
    """Verify generated dataset contains expected sizes and mandatory schemas."""
    txns, accs = generate_synthetic_data(num_accounts=1000, num_transactions=5000, laundering_ratio=0.03)

    assert isinstance(txns, pd.DataFrame)
    assert isinstance(accs, pd.DataFrame)
    assert len(txns) >= 5000
    assert len(accs) >= 1000

    required_txn_cols = {
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
        "transaction_type",
        "is_laundering",
        "typology",
    }
    assert required_txn_cols.issubset(set(txns.columns))

    required_acc_cols = {"account_id", "account_type", "is_laundering", "typology"}
    assert required_acc_cols.issubset(set(accs.columns))


def test_injected_typologies_presence():
    """Ensure all 4 requested money laundering typologies are present."""
    txns, accs = generate_synthetic_data(num_accounts=500, num_transactions=2000, laundering_ratio=0.05)

    laundering_txns = txns[txns["is_laundering"] == 1]
    assert len(laundering_txns) > 0

    expected_typologies = {"STRUCTURING", "FAN_IN_FAN_OUT", "CIRCULAR", "LAYERING"}
    present_typologies = set(laundering_txns["typology"].unique())
    assert expected_typologies.issubset(present_typologies)

    # Check laundering ratio is close to requested ~5%
    ratio = txns["is_laundering"].mean()
    assert 0.03 <= ratio <= 0.08


def test_loader_cache_mechanism(tmp_path):
    """Ensure data loader generates and loads cached parquet files correctly."""
    raw_dir = str(tmp_path / "raw")
    txns1, accs1 = load_dataset(dataset_type="synthetic", raw_dir=raw_dir, num_accounts=200, num_transactions=1000)
    assert len(txns1) >= 1000

    # Load again without regenerating
    txns2, accs2 = load_dataset(dataset_type="synthetic", raw_dir=raw_dir, force_regenerate=False)
    assert len(txns1) == len(txns2)
    assert len(accs1) == len(accs2)
