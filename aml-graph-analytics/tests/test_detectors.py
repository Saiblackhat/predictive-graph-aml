"""
Unit tests for Rule-Based Detectors.
"""

import pandas as pd
import networkx as nx
from src.detection.rules import (
    detect_structuring,
    detect_fan_in_fan_out,
    detect_circular_transfers,
    detect_rapid_layering_passthrough,
)
from src.detection.baseline import run_rule_based_detection


def test_structuring_detector():
    txns = pd.DataFrame(
        [
            {"sender_id": "SMURF_1", "receiver_id": "DEST", "amount": 9500.0},
            {"sender_id": "SMURF_1", "receiver_id": "DEST", "amount": 9800.0},
            {"sender_id": "SMURF_1", "receiver_id": "DEST", "amount": 9200.0},
            {"sender_id": "BENIGN_1", "receiver_id": "DEST", "amount": 100.0},
        ]
    )
    flagged = detect_structuring(txns, threshold=10000.0, lower_bound=9000.0, min_txns=3)
    flagged_ids = set(flagged["account_id"])
    assert "SMURF_1" in flagged_ids
    assert "BENIGN_1" not in flagged_ids


def test_fan_in_fan_out_detector():
    G = nx.DiGraph()
    # Create Hub H with 4 incoming and 4 outgoing
    for i in range(4):
        G.add_edge(f"SRC_{i}", "HUB", weight=1000.0)
        G.add_edge("HUB", f"DST_{i}", weight=1000.0)
    # Add regular node
    G.add_edge("X", "Y", weight=50.0)

    flagged = detect_fan_in_fan_out(G, min_in_degree=4, min_out_degree=4)
    assert len(flagged) == 1
    assert flagged.iloc[0]["account_id"] == "HUB"


def test_circular_transfer_detector():
    G = nx.DiGraph()
    # Form cycle A -> B -> C -> A
    G.add_edge("A", "B", weight=500.0)
    G.add_edge("B", "C", weight=500.0)
    G.add_edge("C", "A", weight=500.0)
    # Form linear path X -> Y -> Z
    G.add_edge("X", "Y", weight=100.0)
    G.add_edge("Y", "Z", weight=100.0)

    flagged, cycles = detect_circular_transfers(G, min_cycle_length=3, max_cycle_length=3)
    flagged_ids = set(flagged["account_id"])
    assert {"A", "B", "C"}.issubset(flagged_ids)
    assert "X" not in flagged_ids
    assert len(cycles) >= 1
