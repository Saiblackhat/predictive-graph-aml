"""
Unit tests for Graph Builder and Statistics.
"""

import pandas as pd
import networkx as nx
from src.graph.builder import build_transaction_graph
from src.graph.statistics import compute_graph_statistics


def test_build_transaction_graph_and_statistics():
    txns = pd.DataFrame(
        [
            {"transaction_id": "T1", "timestamp": "2026-01-01 10:00:00", "sender_id": "A", "receiver_id": "B", "amount": 100.0, "is_laundering": 0, "txn_timestamp_unix": 1000},
            {"transaction_id": "T2", "timestamp": "2026-01-01 11:00:00", "sender_id": "A", "receiver_id": "B", "amount": 200.0, "is_laundering": 0, "txn_timestamp_unix": 2000},
            {"transaction_id": "T3", "timestamp": "2026-01-01 12:00:00", "sender_id": "B", "receiver_id": "C", "amount": 250.0, "is_laundering": 1, "txn_timestamp_unix": 3000},
        ]
    )
    accs = pd.DataFrame(
        [
            {"account_id": "A", "account_type": "INDIVIDUAL", "is_laundering": 0, "typology": "BENIGN"},
            {"account_id": "B", "account_type": "INDIVIDUAL", "is_laundering": 0, "typology": "BENIGN"},
            {"account_id": "C", "account_type": "INDIVIDUAL", "is_laundering": 1, "typology": "SUSPICIOUS"},
        ]
    )

    G, multi_G = build_transaction_graph(txns, accs)

    # MultiDiGraph should have all 3 transactions
    assert multi_G.number_of_nodes() == 3
    assert multi_G.number_of_edges() == 3

    # Collapsed DiGraph should have 2 edges: (A->B with sum 300, count 2) and (B->C with sum 250, count 1)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G["A"]["B"]["total_amount"] == 300.0
    assert G["A"]["B"]["txn_count"] == 2

    # Verify statistics
    stats = compute_graph_statistics(G)
    assert stats["num_nodes"] == 3
    assert stats["num_edges"] == 2
    assert stats["total_volume"] == 550.0
    assert stats["num_weakly_connected_components"] == 1
