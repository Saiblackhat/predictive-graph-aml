"""
Configurable Rule-Based Typology Detectors for aml-ledger-trace (Phase 3).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
import numpy as np
import pandas as pd
from backend.detection.config import DEFAULT_THRESHOLDS


def detect_structuring(
    transactions_df: pd.DataFrame,
    thresholds: Optional[Dict] = None,
) -> pd.DataFrame:
    cfg = thresholds or DEFAULT_THRESHOLDS
    upper = cfg["structuring_limit"]
    lower = cfg["structuring_floor"]
    min_txns = cfg["structuring_min_txns"]

    suspect = transactions_df[(transactions_df["amount"] >= lower) & (transactions_df["amount"] < upper)]
    s_counts = suspect.groupby("sender_id")["amount"].agg(["count", "sum"]).reset_index()
    r_counts = suspect.groupby("receiver_id")["amount"].agg(["count", "sum"]).reset_index()

    f_senders = s_counts[s_counts["count"] >= min_txns].rename(columns={"sender_id": "account_id"})
    f_receivers = r_counts[r_counts["count"] >= min_txns].rename(columns={"receiver_id": "account_id"})

    combined = pd.concat([f_senders, f_receivers]).drop_duplicates(subset=["account_id"])
    combined["rule_name"] = "RULE_STRUCTURING"
    combined["rule_description"] = f">= {min_txns} transactions between ₹{lower:,.0f} and ₹{upper:,.0f}"
    return combined[["account_id", "rule_name", "rule_description"]]


def detect_fan_in_fan_out(
    G: nx.DiGraph,
    thresholds: Optional[Dict] = None,
) -> pd.DataFrame:
    cfg = thresholds or DEFAULT_THRESHOLDS
    min_in = cfg["fan_in_min_degree"]
    min_out = cfg["fan_out_min_degree"]

    flagged = []
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        if in_deg >= min_in and out_deg >= min_out:
            flagged.append(
                {
                    "account_id": node,
                    "rule_name": "RULE_FAN_IN_FAN_OUT",
                    "rule_description": f"Hub node with in-degree={in_deg} and out-degree={out_deg}",
                }
            )
    return pd.DataFrame(flagged)


def detect_circular_transfers(
    G: nx.DiGraph,
    thresholds: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    cfg = thresholds or DEFAULT_THRESHOLDS
    min_len = cfg["cycle_min_length"]
    max_len = cfg["cycle_max_length"]
    max_search = cfg["cycle_max_search"]

    detected_cycles: List[List[str]] = []
    cycle_nodes: Set[str] = set()

    nodes_ordered = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_ordered)}

    for start_node in nodes_ordered:
        if len(detected_cycles) >= max_search:
            break
        start_idx = node_to_idx[start_node]
        stack = [(start_node, [start_node])]

        while stack and len(detected_cycles) < max_search:
            curr, path = stack.pop()
            curr_len = len(path)

            for neighbor in G.successors(curr):
                if neighbor == start_node:
                    if min_len <= curr_len <= max_len:
                        detected_cycles.append(list(path))
                        cycle_nodes.update(path)
                        if len(detected_cycles) >= max_search:
                            break
                elif curr_len < max_len:
                    if node_to_idx.get(neighbor, -1) > start_idx and neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))

    flagged = [
        {
            "account_id": n,
            "rule_name": "RULE_CIRCULAR_TRANSFER",
            "rule_description": f"Participates in directed closed cycle of length {min_len}-{max_len}",
        }
        for n in cycle_nodes
    ]
    return pd.DataFrame(flagged), detected_cycles


def detect_rapid_layering(
    transactions_df: pd.DataFrame,
    thresholds: Optional[Dict] = None,
) -> pd.DataFrame:
    cfg = thresholds or DEFAULT_THRESHOLDS
    max_retention = cfg["layering_max_retention_ratio"]
    min_txns = cfg["layering_min_txns"]

    in_flows = transactions_df.groupby("receiver_id")["amount"].agg(in_volume="sum", in_count="count")
    out_flows = transactions_df.groupby("sender_id")["amount"].agg(out_volume="sum", out_count="count")

    flow_df = pd.concat([in_flows, out_flows], axis=1).fillna(0)
    active = flow_df[(flow_df["in_count"] >= min_txns) & (flow_df["out_count"] >= min_txns)].copy()

    max_vol = np.maximum(active["in_volume"], active["out_volume"])
    vol_diff = np.abs(active["in_volume"] - active["out_volume"])
    active["retention_ratio"] = np.where(max_vol > 0, vol_diff / max_vol, 1.0)

    layering = active[active["retention_ratio"] <= max_retention].reset_index().rename(columns={"index": "account_id"})
    layering["rule_name"] = "RULE_LAYERING_PASSTHROUGH"
    layering["rule_description"] = "Rapid pass-through with minimal fund retention (< 25% balance difference)"
    return layering[["account_id", "rule_name", "rule_description"]]
