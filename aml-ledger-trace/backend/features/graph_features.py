"""
Topological Graph Feature Extraction Module for aml-ledger-trace (Phase 4).
"""

from __future__ import annotations

from typing import Optional
import networkx as nx
import pandas as pd
import numpy as np


def extract_graph_features(
    G: nx.DiGraph,
    approximate_betweenness_k: Optional[int] = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame()

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # PageRank
    try:
        pageranks = nx.pagerank(G, alpha=0.85, max_iter=200, weight="weight")
    except Exception:
        pageranks = {n: 1.0 / len(nodes) for n in nodes}

    # Betweenness Centrality
    num_nodes = len(nodes)
    k_sample = approximate_betweenness_k if (approximate_betweenness_k and num_nodes > approximate_betweenness_k) else None
    try:
        betweenness = nx.betweenness_centrality(G, k=k_sample, normalized=True, weight="weight", seed=random_seed)
    except Exception:
        betweenness = {n: 0.0 for n in nodes}

    # Clustering Coefficient
    undir_G = G.to_undirected(as_view=True)
    try:
        clustering_coeff = nx.clustering(undir_G)
    except Exception:
        clustering_coeff = {n: 0.0 for n in nodes}

    # Louvain Community Detection
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(undir_G, random_state=random_seed)
    except Exception:
        try:
            communities = nx.algorithms.community.louvain_communities(undir_G, seed=random_seed)
            partition = {}
            for cid, cset in enumerate(communities):
                for node in cset:
                    partition[node] = cid
        except Exception:
            partition = {n: 0 for n in nodes}

    # Cycle Participation Count (bounded DFS)
    cycle_counts = {n: 0 for n in nodes}
    nodes_ordered = list(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes_ordered)}
    found = 0
    for s_node in nodes_ordered:
        if found >= 500:
            break
        s_idx = node_to_idx[s_node]
        stack = [(s_node, [s_node])]
        while stack and found < 500:
            curr, path = stack.pop()
            curr_len = len(path)
            for nbr in G.successors(curr):
                if nbr == s_node:
                    if 3 <= curr_len <= 5:
                        found += 1
                        for p in path:
                            cycle_counts[p] = cycle_counts.get(p, 0) + 1
                        if found >= 500:
                            break
                elif curr_len < 5:
                    if node_to_idx.get(nbr, -1) > s_idx and nbr not in path:
                        stack.append((nbr, path + [nbr]))

    records = []
    for node in nodes:
        in_d = in_degrees.get(node, 0)
        out_d = out_degrees.get(node, 0)
        records.append(
            {
                "account_id": node,
                "in_degree": in_d,
                "out_degree": out_d,
                "total_degree": in_d + out_d,
                "degree_ratio": in_d / (out_d + 1.0),
                "pagerank": pageranks.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "clustering_coefficient": clustering_coeff.get(node, 0.0),
                "louvain_community_id": partition.get(node, 0),
                "cycle_participation_count": cycle_counts.get(node, 0),
            }
        )

    return pd.DataFrame(records)
