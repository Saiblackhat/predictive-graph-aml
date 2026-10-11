"""
Topological Graph Feature Engineering.

Extracts rich graph structural features per account node:
  - In-Degree, Out-Degree, Total Degree, Degree Ratio (In / (Out + 1))
  - PageRank (centrality and authority)
  - Betweenness Centrality (intermediary broker score, k-sampled approximation for scale)
  - Local Clustering Coefficient (neighborhood cohesion)
  - Louvain Community Membership (partition / modularity cluster)
  - Cycle Participation Count
"""

from __future__ import annotations

from typing import Dict, List, Optional
import networkx as nx
import pandas as pd
import numpy as np


def extract_graph_features(
    G: nx.DiGraph,
    approximate_betweenness_k: Optional[int] = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute graph topological features for all nodes in the directed graph.

    Parameters:
    -----------
    G : nx.DiGraph
        Directed transaction graph.
    approximate_betweenness_k : Optional[int]
        Number of pivot samples for betweenness centrality calculation.
        Using k-sampling ensures linear runtime O(k * (|V| + |E|)) rather than cubic O(|V| * |E|).
    random_seed : int
        Seed for stochastic centrality approximation.

    Returns:
    --------
    pd.DataFrame:
        DataFrame indexed by account_id containing topological features.
    """
    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame()

    # 1. Degree Metrics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # 2. PageRank
    try:
        pageranks = nx.pagerank(G, alpha=0.85, max_iter=200, weight="weight")
    except Exception:
        pageranks = {node: 1.0 / len(nodes) for node in nodes}

    # 3. Betweenness Centrality (Approximated if graph is large)
    num_nodes = len(nodes)
    k_sample = None
    if approximate_betweenness_k and num_nodes > approximate_betweenness_k:
        k_sample = approximate_betweenness_k

    try:
        betweenness = nx.betweenness_centrality(
            G,
            k=k_sample,
            normalized=True,
            weight="weight",
            seed=random_seed,
        )
    except Exception:
        betweenness = {node: 0.0 for node in nodes}

    # 4. Clustering Coefficient (convert to undirected view for standard local clustering)
    undirected_G = G.to_undirected(as_view=True)
    try:
        clustering_coeff = nx.clustering(undirected_G)
    except Exception:
        clustering_coeff = {node: 0.0 for node in nodes}

    # 5. Louvain Community Detection
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(undirected_G, random_state=random_seed)
    except Exception:
        try:
            communities = nx.algorithms.community.louvain_communities(undirected_G, seed=random_seed)
            partition = {}
            for cid, cset in enumerate(communities):
                for node in cset:
                    partition[node] = cid
        except Exception:
            partition = {node: 0 for node in nodes}

    # 6. Cycle Participation Detection (fast bounded DFS)
    cycle_counts = {node: 0 for node in nodes}
    nodes_ordered = list(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes_ordered)}
    found_cycles = 0

    for start_node in nodes_ordered:
        if found_cycles >= 500:
            break
        start_idx = node_to_idx[start_node]
        stack = [(start_node, [start_node])]
        while stack and found_cycles < 500:
            curr, path = stack.pop()
            curr_len = len(path)
            for neighbor in G.successors(curr):
                if neighbor == start_node:
                    if 3 <= curr_len <= 5:
                        found_cycles += 1
                        for n in path:
                            cycle_counts[n] = cycle_counts.get(n, 0) + 1
                        if found_cycles >= 500:
                            break
                elif curr_len < 5:
                    if node_to_idx.get(neighbor, -1) > start_idx and neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))

    records = []
    for node in nodes:
        in_d = in_degrees.get(node, 0)
        out_d = out_degrees.get(node, 0)
        tot_d = in_d + out_d
        deg_ratio = in_d / (out_d + 1.0)

        records.append(
            {
                "account_id": node,
                "in_degree": in_d,
                "out_degree": out_d,
                "total_degree": tot_d,
                "degree_ratio": deg_ratio,
                "pagerank": pageranks.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "clustering_coefficient": clustering_coeff.get(node, 0.0),
                "louvain_community_id": partition.get(node, 0),
                "cycle_participation_count": cycle_counts.get(node, 0),
            }
        )

    df_feats = pd.DataFrame(records)
    return df_feats
