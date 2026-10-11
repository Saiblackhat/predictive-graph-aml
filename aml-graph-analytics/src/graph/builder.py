"""
Financial Transaction Graph Builder.

Constructs directed, weighted graphs using NetworkX:
  - Nodes: Accounts (with account_type, is_laundering ground truth, etc.)
  - Edges: Transactions (aggregated into multi-attribute directed edges:
           total amount, transaction count, average amount, first/last timestamp).

Supports both nx.MultiDiGraph (for fine-grained temporal sequence analysis)
and collapsed nx.DiGraph (for fast topological and centrality analysis).
"""

from __future__ import annotations

from typing import Tuple
import networkx as nx
import pandas as pd


def build_transaction_graph(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, nx.MultiDiGraph]:
    """
    Construct both a simple aggregated DiGraph and a MultiDiGraph from transaction records.

    Parameters:
    -----------
    transactions_df : pd.DataFrame
        Processed transaction records containing sender_id, receiver_id, amount, timestamp, etc.
    accounts_df : pd.DataFrame
        Processed account records containing account_id, account_type, is_laundering, etc.

    Returns:
    --------
    Tuple[nx.DiGraph, nx.MultiDiGraph]:
        - G: Collapsed directed weighted graph (one edge per pair with aggregated metrics).
        - multi_G: MultiDiGraph preserving every individual transaction as an edge.
    """
    multi_G = nx.MultiDiGraph()
    G = nx.DiGraph()

    # 1. Add Nodes with Metadata
    account_attr_dict = {}
    for _, row in accounts_df.iterrows():
        acc_id = str(row["account_id"])
        attrs = {
            "account_id": acc_id,
            "account_type": row.get("account_type", "UNKNOWN"),
            "is_laundering": int(row.get("is_laundering", 0)),
            "typology": row.get("typology", "BENIGN"),
        }
        account_attr_dict[acc_id] = attrs

    multi_G.add_nodes_from((acc, attrs) for acc, attrs in account_attr_dict.items())
    G.add_nodes_from((acc, attrs) for acc, attrs in account_attr_dict.items())

    # Ensure any endpoints not in accounts_df are initialized with default metadata
    for _, row in transactions_df.iterrows():
        u = str(row["sender_id"])
        v = str(row["receiver_id"])
        for node in (u, v):
            if not G.has_node(node):
                default_meta = {
                    "account_id": node,
                    "account_type": "UNKNOWN",
                    "is_laundering": 0,
                    "typology": "BENIGN",
                }
                G.add_node(node, **default_meta)
                multi_G.add_node(node, **default_meta)

        # 2. Add individual transactions to MultiDiGraph
        multi_G.add_edge(
            u,
            v,
            transaction_id=row.get("transaction_id", ""),
            amount=float(row["amount"]),
            timestamp=str(row["timestamp"]),
            timestamp_unix=int(row.get("txn_timestamp_unix", 0)),
            transaction_type=row.get("transaction_type", "TRANSFER"),
            is_laundering=int(row.get("is_laundering", 0)),
            typology=row.get("typology", "BENIGN"),
        )

    # 3. Aggregate pairwise interactions for collapsed DiGraph
    agg_df = (
        transactions_df.groupby(["sender_id", "receiver_id"])
        .agg(
            total_amount=("amount", "sum"),
            txn_count=("amount", "count"),
            avg_amount=("amount", "mean"),
            min_amount=("amount", "min"),
            max_amount=("amount", "max"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            is_laundering_edge=("is_laundering", "max"),
        )
        .reset_index()
    )

    for _, row in agg_df.iterrows():
        u = str(row["sender_id"])
        v = str(row["receiver_id"])
        G.add_edge(
            u,
            v,
            weight=float(row["total_amount"]),
            total_amount=float(row["total_amount"]),
            txn_count=int(row["txn_count"]),
            avg_amount=float(row["avg_amount"]),
            min_amount=float(row["min_amount"]),
            max_amount=float(row["max_amount"]),
            first_timestamp=str(row["first_timestamp"]),
            last_timestamp=str(row["last_timestamp"]),
            is_laundering=int(row["is_laundering_edge"]),
        )

    return G, multi_G
