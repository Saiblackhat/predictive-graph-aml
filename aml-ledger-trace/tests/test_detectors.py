"""
Unit tests for Phase 3 Configurable Typology Detectors.
"""

import pandas as pd
import networkx as nx
from backend.detection.rules import (
    detect_structuring,
    detect_fan_in_fan_out,
    detect_circular_transfers,
    detect_rapid_layering,
)
from backend.detection.config import DEFAULT_THRESHOLDS


def test_structuring_custom_thresholds():
    txns = pd.DataFrame(
        [
            {"sender_id": "S1", "receiver_id": "DEST", "amount": 48000.0},
            {"sender_id": "S1", "receiver_id": "DEST", "amount": 49000.0},
            {"sender_id": "S1", "receiver_id": "DEST", "amount": 47500.0},
            {"sender_id": "B1", "receiver_id": "DEST", "amount": 1000.0},
        ]
    )
    # Default ₹50,000 threshold
    flagged = detect_structuring(txns)
    assert "S1" in set(flagged["account_id"])
    assert "B1" not in set(flagged["account_id"])

    # Test dynamic threshold update (e.g. lower threshold ₹30,000 -> S1 shouldn't trigger if amounts are 48k)
    custom_cfg = DEFAULT_THRESHOLDS.copy()
    custom_cfg["structuring_limit"] = 30000.0
    custom_cfg["structuring_floor"] = 25000.0
    flagged_custom = detect_structuring(txns, thresholds=custom_cfg)
    assert len(flagged_custom) == 0


def test_cycle_and_hub_detectors():
    G = nx.DiGraph()
    # Cycle A -> B -> C -> A
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")

    # Hub H
    for i in range(4):
        G.add_edge(f"IN_{i}", "H")
        G.add_edge("H", f"OUT_{i}")

    c_df, cycles = detect_circular_transfers(G)
    assert {"A", "B", "C"}.issubset(set(c_df["account_id"]))
    assert len(cycles) >= 1

    h_df = detect_fan_in_fan_out(G)
    assert "H" in set(h_df["account_id"])
