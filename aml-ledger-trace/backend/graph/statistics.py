"""
Graph Diagnostics and Statistics Module for aml-ledger-trace (Phase 2).
"""

from __future__ import annotations

from typing import Dict, Any
import networkx as nx
import numpy as np


def compute_graph_statistics(G: nx.DiGraph) -> Dict[str, Any]:
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {"num_nodes": 0, "num_edges": 0}

    density = nx.density(G)
    wcc = nx.number_weakly_connected_components(G)
    scc = nx.number_strongly_connected_components(G)

    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_vol = sum(d.get("weight", 0.0) for _, _, d in G.edges(data=True))

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": float(density),
        "total_volume_inr": float(total_vol),
        "wcc_count": wcc,
        "scc_count": scc,
        "avg_in_degree": float(np.mean(in_degrees)) if in_degrees else 0.0,
        "avg_out_degree": float(np.mean(out_degrees)) if out_degrees else 0.0,
        "max_in_degree": int(np.max(in_degrees)) if in_degrees else 0,
        "max_out_degree": int(np.max(out_degrees)) if out_degrees else 0,
    }
