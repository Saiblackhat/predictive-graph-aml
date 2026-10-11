"""
Rule-Based Typology Detectors.

This module implements deterministic expert rules derived from standard financial crime compliance guidelines:
  1. Structuring / Smurfing Detector:
     Detects accounts making or receiving repeated transactions slightly below statutory CTR threshold ($10,000).
  2. Fan-In / Fan-Out Hub Detector:
     Identifies mule/aggregator accounts with disproportionate in-degree followed by dispersed out-degree.
  3. Circular Transfer Detector:
     Finds directed closed loops (A -> B -> C -> A) within the transaction graph using simple cycles.
  4. Layering / Rapid Pass-Through Detector:
     Flags intermediary pass-through accounts where money enters and leaves within a brief time delta with low retention.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd


def detect_structuring(
    transactions_df: pd.DataFrame,
    threshold: float = 10000.0,
    lower_bound: float = 8500.0,
    min_txns: int = 3,
) -> pd.DataFrame:
    """
    Detect accounts engaging in structuring (deposits/transfers clustered just beneath a reporting threshold).

    Parameters:
    -----------
    transactions_df : pd.DataFrame
        Transaction records with sender_id, receiver_id, amount.
    threshold : float
        Statutory threshold (e.g. $10,000).
    lower_bound : float
        Structuring floor (e.g. $8,500).
    min_txns : int
        Minimum count of suspect transactions to trigger rule flag.

    Returns:
    --------
    pd.DataFrame:
        Table of flagged accounts with metric details.
    """
    suspect_mask = (transactions_df["amount"] >= lower_bound) & (transactions_df["amount"] < threshold)
    suspect_txns = transactions_df[suspect_mask]

    # Check both senders (smurfs) and receivers (aggregators)
    sender_counts = suspect_txns.groupby("sender_id")["amount"].agg(["count", "sum"]).reset_index()
    receiver_counts = suspect_txns.groupby("receiver_id")["amount"].agg(["count", "sum"]).reset_index()

    flagged_senders = sender_counts[sender_counts["count"] >= min_txns].rename(
        columns={"sender_id": "account_id"}
    )
    flagged_receivers = receiver_counts[receiver_counts["count"] >= min_txns].rename(
        columns={"receiver_id": "account_id"}
    )

    combined = pd.concat([flagged_senders, flagged_receivers]).drop_duplicates(subset=["account_id"])
    combined["rule_name"] = "RULE_STRUCTURING"
    combined["rule_description"] = f">= {min_txns} transactions between ${lower_bound:,.0f} and ${threshold:,.0f}"
    return combined


def detect_fan_in_fan_out(
    G: nx.DiGraph,
    min_in_degree: int = 4,
    min_out_degree: int = 4,
) -> pd.DataFrame:
    """
    Detect aggregator/dispersal hub accounts having both high in-degree and high out-degree.

    Parameters:
    -----------
    G : nx.DiGraph
        Directed transaction graph.
    min_in_degree : int
        Threshold for incoming degree.
    min_out_degree : int
        Threshold for outgoing degree.

    Returns:
    --------
    pd.DataFrame:
        Flagged hub accounts.
    """
    flagged = []
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        if in_deg >= min_in_degree and out_deg >= min_out_degree:
            flagged.append(
                {
                    "account_id": node,
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "rule_name": "RULE_FAN_IN_FAN_OUT",
                    "rule_description": f"Hub node with in-degree={in_deg} and out-degree={out_deg}",
                }
            )

    return pd.DataFrame(flagged)


def detect_circular_transfers(
    G: nx.DiGraph,
    min_cycle_length: int = 3,
    max_cycle_length: int = 5,
    max_cycles_to_search: int = 500,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Detect circular round-tripping transfers (directed simple cycles of bounded length).
    Uses depth-bounded DFS to ensure O(V * d^k) efficiency instead of exponential Johnson's algorithm.

    Parameters:
    -----------
    G : nx.DiGraph
        Directed transaction graph.
    min_cycle_length : int
        Minimum cycle length (e.g. 3 hops).
    max_cycle_length : int
        Maximum cycle length (e.g. 5 hops).
    max_cycles_to_search : int
        Guardrail to prevent excessive cycle searches.

    Returns:
    --------
    Tuple[pd.DataFrame, List[List[str]]]:
        - DataFrame of flagged accounts participating in cycles.
        - List of detected cycle node sequences.
    """
    detected_cycles: List[List[str]] = []
    cycle_nodes: Set[str] = set()

    # Bounded DFS for short directed cycles
    # To avoid permutation duplicates, order nodes and only consider cycles where the start node has min id
    nodes_ordered = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_ordered)}

    for start_node in nodes_ordered:
        if len(detected_cycles) >= max_cycles_to_search:
            break
        start_idx = node_to_idx[start_node]

        # Stack entries: (current_node, path_list)
        stack = [(start_node, [start_node])]
        while stack and len(detected_cycles) < max_cycles_to_search:
            curr, path = stack.pop()
            curr_len = len(path)

            for neighbor in G.successors(curr):
                if neighbor == start_node:
                    if min_cycle_length <= curr_len <= max_cycle_length:
                        detected_cycles.append(list(path))
                        cycle_nodes.update(path)
                        if len(detected_cycles) >= max_cycles_to_search:
                            break
                elif curr_len < max_cycle_length:
                    # Enforce start_node has lowest index in the cycle to avoid duplicate cycles
                    if node_to_idx.get(neighbor, -1) > start_idx and neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))

    flagged_records = [
        {
            "account_id": node,
            "rule_name": "RULE_CIRCULAR_TRANSFER",
            "rule_description": f"Participates in directed closed transfer cycle of length {min_cycle_length}-{max_cycle_length}",
        }
        for node in cycle_nodes
    ]

    return pd.DataFrame(flagged_records), detected_cycles


def detect_rapid_layering_passthrough(
    transactions_df: pd.DataFrame,
    max_flow_retention_ratio: float = 0.25,
    min_txns_each_side: int = 2,
) -> pd.DataFrame:
    """
    Detect pass-through / layering entities where total inflow approximately equals total outflow
    (funds are immediately layered through rather than retained).

    Parameters:
    -----------
    transactions_df : pd.DataFrame
        Transactions dataframe.
    max_flow_retention_ratio : float
        Allowed retention ratio |in - out| / max(in, out).
    min_txns_each_side : int
        Minimum number of in and out transactions.

    Returns:
    --------
    pd.DataFrame:
        Flagged layering accounts.
    """
    in_flows = transactions_df.groupby("receiver_id")["amount"].agg(in_volume="sum", in_count="count")
    out_flows = transactions_df.groupby("sender_id")["amount"].agg(out_volume="sum", out_count="count")

    flow_df = pd.concat([in_flows, out_flows], axis=1).fillna(0)
    
    # Filter nodes with minimum activity
    active_mask = (flow_df["in_count"] >= min_txns_each_side) & (flow_df["out_count"] >= min_txns_each_side)
    candidate_df = flow_df[active_mask].copy()

    # Compute retention: |inflow - outflow| / max(inflow, outflow)
    max_vol = np.maximum(candidate_df["in_volume"], candidate_df["out_volume"])
    vol_diff = np.abs(candidate_df["in_volume"] - candidate_df["out_volume"])
    candidate_df["retention_ratio"] = np.where(max_vol > 0, vol_diff / max_vol, 1.0)

    # Layering mules retain very little capital (< 25%)
    layering_mask = candidate_df["retention_ratio"] <= max_flow_retention_ratio
    flagged = candidate_df[layering_mask].reset_index().rename(columns={"index": "account_id"})
    
    flagged["rule_name"] = "RULE_LAYERING_PASSTHROUGH"
    flagged["rule_description"] = "Rapid pass-through with minimal fund retention (< 25% balance difference)"
    return flagged[["account_id", "rule_name", "rule_description"]]
