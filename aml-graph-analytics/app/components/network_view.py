"""
PyVis Interactive Network Visualizer Component.

Renders interactive HTML graphs for ego-networks around flagged accounts:
  - Node color represents risk tier (Critical: Red, High: Orange, Medium: Yellow, Low: Blue).
  - Node size proportional to total transacted volume or degree.
  - Edges labeled with transaction amounts and transfer counts.
  - Physics-enabled layout for intuitive graph exploration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import networkx as nx
from pyvis.network import Network
import pandas as pd


def generate_ego_network_html(
    G: nx.DiGraph,
    center_account_id: str,
    radius: int = 1,
    max_nodes: int = 40,
    scored_accounts_df: Optional[pd.DataFrame] = None,
    output_html_path: str = "reports/network_view.html",
) -> str:
    """
    Generate an interactive PyVis HTML visualization for a target account's neighborhood.

    Parameters:
    -----------
    G : nx.DiGraph
        Transaction network.
    center_account_id : str
        The account ID to inspect.
    radius : int
        Ego network radius (1 for direct neighbors, 2 for extended neighborhood).
    max_nodes : int
        Max nodes threshold to ensure fluid browser responsiveness.
    scored_accounts_df : Optional[pd.DataFrame]
        Table mapping account_id to risk scores and tiers.
    output_html_path : str
        Target filepath for generated standalone HTML.

    Returns:
    --------
    str:
        HTML string representation of the interactive graph.
    """
    if center_account_id not in G:
        return "<p style='color:red;'>Selected account not found in transaction network.</p>"

    # Extract ego subgraph
    ego_nodes = nx.single_source_shortest_path_length(G.to_undirected(), center_account_id, cutoff=radius).keys()
    if len(ego_nodes) > max_nodes:
        # Sample neighbors based on edge weights
        sub_nodes = [center_account_id]
        neighbors = sorted(
            [n for n in ego_nodes if n != center_account_id],
            key=lambda n: G.get_edge_data(center_account_id, n, {}).get("weight", 0)
            + G.get_edge_data(n, center_account_id, {}).get("weight", 0),
            reverse=True,
        )
        ego_nodes = sub_nodes + neighbors[: max_nodes - 1]

    subgraph = G.subgraph(ego_nodes)

    # Initialize PyVis network
    net = Network(height="550px", width="100%", directed=True, bgcolor="#111827", font_color="#F3F4F6")
    net.force_atlas_2based()

    # Create score lookup
    score_lookup = {}
    if scored_accounts_df is not None and not scored_accounts_df.empty:
        for _, r in scored_accounts_df.iterrows():
            score_lookup[str(r["account_id"])] = {
                "score": r.get("aml_risk_score", 0.0),
                "tier": r.get("risk_tier", "LOW"),
                "rules": r.get("triggered_rules", []),
            }

    # Node coloring palette
    color_map = {
        "CRITICAL": "#EF4444",  # Red
        "HIGH": "#F97316",      # Orange
        "MEDIUM": "#FBBF24",    # Yellow
        "LOW": "#3B82F6",       # Blue
    }

    # Add nodes
    for node in subgraph.nodes():
        node_str = str(node)
        meta = score_lookup.get(node_str, {"score": 0.0, "tier": "LOW", "rules": []})
        tier = meta["tier"]
        score = meta["score"]
        rules = meta["rules"]

        is_center = node_str == center_account_id
        color = "#DC2626" if is_center and tier == "CRITICAL" else color_map.get(tier, "#9CA3AF")
        size = 32 if is_center else 18

        hover_text = (
            f"<b>Account:</b> {node_str}<br>"
            f"<b>Risk Score:</b> {score} / 100 ({tier})<br>"
            f"<b>In-Degree:</b> {G.in_degree(node)}<br>"
            f"<b>Out-Degree:</b> {G.out_degree(node)}<br>"
            f"<b>Rules:</b> {', '.join(rules) if rules else 'None'}"
        )

        net.add_node(
            node_str,
            label=f"{node_str}{' (FOCUS)' if is_center else ''}",
            title=hover_text,
            color=color,
            size=size,
            borderWidth=3 if is_center else 1,
        )

    # Add edges
    for u, v, data in subgraph.edges(data=True):
        amt = data.get("total_amount", 0.0)
        cnt = data.get("txn_count", 1)
        net.add_edge(
            str(u),
            str(v),
            value=np.log1p(amt),
            title=f"Total: ${amt:,.2f} ({cnt} transfers)",
            label=f"${amt:,.0f}" if amt > 1000 else "",
            color="#6B7280",
            arrows="to",
        )

    # Render HTML
    out_p = Path(output_html_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out_p))

    with open(out_p, "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content
