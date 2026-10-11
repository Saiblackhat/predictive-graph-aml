"""
Graph Diagnostics and Topological Statistics Module.

Computes essential network metrics for AML graph surveillance:
  - Node & Edge counts
  - Graph density
  - Weakly and Strongly Connected Components (WCC, SCC)
  - Degree distribution summary (in, out, total)
  - Reciprocity and clustering coefficient statistics
"""

from __future__ import annotations

from typing import Dict, Any
import networkx as nx
import numpy as np


def compute_graph_statistics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute comprehensive topological summary metrics for a directed transaction network.

    Parameters:
    -----------
    G : nx.DiGraph
        The directed transaction graph.

    Returns:
    --------
    Dict[str, Any]:
        Dictionary of network statistics.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {"num_nodes": 0, "num_edges": 0}

    density = nx.density(G)
    wcc_count = nx.number_weakly_connected_components(G)
    scc_count = nx.number_strongly_connected_components(G)

    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    
    total_volume = sum(d.get("weight", 0.0) for _, _, d in G.edges(data=True))

    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": float(density),
        "total_volume": float(total_volume),
        "num_weakly_connected_components": wcc_count,
        "num_strongly_connected_components": scc_count,
        "max_in_degree": int(np.max(in_degrees)) if in_degrees else 0,
        "avg_in_degree": float(np.mean(in_degrees)) if in_degrees else 0.0,
        "max_out_degree": int(np.max(out_degrees)) if out_degrees else 0,
        "avg_out_degree": float(np.mean(out_degrees)) if out_degrees else 0.0,
        "reciprocity": float(nx.reciprocity(G)) if num_edges > 0 else 0.0,
    }

    return stats


def print_graph_summary(stats: Dict[str, Any]) -> None:
    """Pretty-print graph summary statistics to console."""
    print("=" * 60)
    print("FINANCIAL TRANSACTION GRAPH TOPOLOGICAL SUMMARY")
    print("=" * 60)
    print(f"Total Accounts (Nodes)           : {stats.get('num_nodes', 0):,}")
    print(f"Total Directed Edges             : {stats.get('num_edges', 0):,}")
    print(f"Total Transaction Volume         : ${stats.get('total_volume', 0.0):,.2f}")
    print(f"Graph Density                    : {stats.get('density', 0.0):.6f}")
    print(f"Weakly Connected Components      : {stats.get('num_weakly_connected_components', 0):,}")
    print(f"Strongly Connected Components    : {stats.get('num_strongly_connected_components', 0):,}")
    print(f"Average / Max In-Degree          : {stats.get('avg_in_degree', 0.0):.2f} / {stats.get('max_in_degree', 0)}")
    print(f"Average / Max Out-Degree         : {stats.get('avg_out_degree', 0.0):.2f} / {stats.get('max_out_degree', 0)}")
    print(f"Network Reciprocity              : {stats.get('reciprocity', 0.0):.4f}")
    print("=" * 60)
