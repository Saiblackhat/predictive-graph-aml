"""
Unit tests for Phase 2 Preprocessing and Graph Builder.
"""

import pandas as pd
from backend.data.preprocessor import preprocess_transactions
from backend.graph.builder import build_transaction_graph
from backend.graph.statistics import compute_graph_statistics


def test_preprocessing_and_graph_construction(tmp_path):
    txns = pd.DataFrame(
        [
            {"transaction_id": "T1", "sender_id": "A", "receiver_id": "B", "amount": 1000.0, "timestamp": "2026-01-01 10:00:00", "is_laundering": 0},
            {"transaction_id": "T1", "sender_id": "A", "receiver_id": "B", "amount": 1000.0, "timestamp": "2026-01-01 10:00:00", "is_laundering": 0},  # duplicate
            {"transaction_id": "T2", "sender_id": "A", "receiver_id": "B", "amount": 2000.0, "timestamp": "2026-01-01 11:00:00", "is_laundering": 0},
            {"transaction_id": "T3", "sender_id": "B", "receiver_id": "C", "amount": 2500.0, "timestamp": "2026-01-01 12:00:00", "is_laundering": 1},
        ]
    )
    accs = pd.DataFrame([{"account_id": "A", "is_laundering": 0}, {"account_id": "B", "is_laundering": 0}])

    proc_txns, proc_accs = preprocess_transactions(txns, accs, processed_dir=str(tmp_path / "proc"))
    assert len(proc_txns) == 3
    assert len(proc_accs) == 3  # Node C automatically synchronized

    G, multi_G = build_transaction_graph(proc_txns, proc_accs)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G["A"]["B"]["total_amount"] == 3000.0

    stats = compute_graph_statistics(G)
    assert stats["num_nodes"] == 3
    assert stats["total_volume_inr"] == 5500.0
