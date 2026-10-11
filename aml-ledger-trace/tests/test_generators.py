"""
Unit tests for Phase 1 Data Generator and Loader.
"""

import pandas as pd
from backend.data.generator import generate_synthetic_transactions
from backend.data.loader import load_dataset


def test_generator_inr_amounts_and_structuring_threshold():
    txns, accs = generate_synthetic_transactions(num_accounts=500, num_transactions=2000, laundering_ratio=0.04)

    assert len(txns) >= 2000
    assert len(accs) >= 500

    # Verify structuring transactions are just below ₹50,000 (between ₹46,000 and ₹50,000)
    struct_txns = txns[txns["typology"] == "STRUCTURING"]
    assert len(struct_txns) > 0
    assert (struct_txns["amount"] < 50000.0).all()
    assert (struct_txns["amount"] >= 46000.0).all()

    # Verify all 4 typologies exist
    typos = set(txns[txns["is_laundering"] == 1]["typology"].unique())
    assert {"STRUCTURING", "FAN_IN_FAN_OUT", "CIRCULAR", "LAYERING"}.issubset(typos)


def test_loader_cache(tmp_path):
    raw_dir = str(tmp_path / "raw")
    tx1, ac1 = load_dataset(dataset_type="synthetic", raw_dir=raw_dir, num_accounts=100, num_transactions=500)
    assert len(tx1) >= 500
    tx2, ac2 = load_dataset(dataset_type="synthetic", raw_dir=raw_dir)
    assert len(tx1) == len(tx2)
