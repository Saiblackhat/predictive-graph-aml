"""
Directed Weighted Transaction Graph Builder for aml-ledger-trace (Phase 2).
"""

from __future__ import annotations

from typing import Tuple
import networkx as nx
import pandas as pd


def build_transaction_graph(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, nx.MultiDiGraph]:
    multi_G = nx.MultiDiGraph()
    G = nx.DiGraph()

    for _, row in accounts_df.iterrows():
        node_id = str(row["account_id"])
        meta = {
            "account_id": node_id,
            "account_type": row.get("account_type", "INDIVIDUAL"),
            "is_laundering": int(row.get("is_laundering", 0)),
            "typology": row.get("typology", "BENIGN"),
        }
        G.add_node(node_id, **meta)
        multi_G.add_node(node_id, **meta)

    # MultiDiGraph edges
    for _, row in transactions_df.iterrows():
        u = str(row["sender_id"])
        v = str(row["receiver_id"])
        for n in (u, v):
            if not G.has_node(n):
                d = {"account_id": n, "account_type": "UNKNOWN", "is_laundering": 0, "typology": "BENIGN"}
                G.add_node(n, **d)
                multi_G.add_node(n, **d)

        multi_G.add_edge(
            u,
            v,
            transaction_id=row.get("transaction_id", ""),
            amount=float(row["amount"]),
            timestamp=str(row["timestamp"]),
            timestamp_unix=int(row.get("txn_timestamp_unix", 0)),
            transaction_type=row.get("transaction_type", "UPI"),
            is_laundering=int(row.get("is_laundering", 0)),
            typology=row.get("typology", "BENIGN"),
        )

    # Collapsed DiGraph edges
    agg_df = (
        transactions_df.groupby(["sender_id", "receiver_id"])
        .agg(
            total_amount=("amount", "sum"),
            txn_count=("amount", "count"),
            avg_amount=("amount", "mean"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            is_laundering=("is_laundering", "max"),
        )
        .reset_index()
    )

    for _, row in agg_df.iterrows():
        G.add_edge(
            str(row["sender_id"]),
            str(row["receiver_id"]),
            weight=float(row["total_amount"]),
            total_amount=float(row["total_amount"]),
            txn_count=int(row["txn_count"]),
            avg_amount=float(row["avg_amount"]),
            first_timestamp=str(row["first_timestamp"]),
            last_timestamp=str(row["last_timestamp"]),
            is_laundering=int(row["is_laundering"]),
        )

    return G, multi_G
