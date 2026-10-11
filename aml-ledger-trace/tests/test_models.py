"""
Unit tests for Phase 4 Feature Engineering, Models, and SHAP.
"""

import pandas as pd
import networkx as nx
from backend.features.graph_features import extract_graph_features
from backend.features.transaction_features import extract_transaction_features
from backend.features.temporal_features import extract_temporal_features
from backend.models.train import prepare_feature_matrix, train_and_evaluate_all_models
from backend.models.explainability import LedgerTraceExplainer


def test_models_pipeline_and_shap():
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=50000.0)
    G.add_edge("B", "C", weight=48000.0)
    G.add_edge("C", "A", weight=47000.0)
    for i in range(12):
        G.add_edge(f"U_{i}", f"V_{i}", weight=1000.0)

    nodes = list(G.nodes())
    accs = pd.DataFrame([{"account_id": n, "is_laundering": 1 if n in ["A", "B", "C"] else 0} for n in nodes])
    txns = pd.DataFrame([{"sender_id": u, "receiver_id": v, "amount": float(d["weight"]), "timestamp": "2026-01-01 10:00:00", "txn_hour": 10} for u, v, d in G.edges(data=True)])

    gf = extract_graph_features(G)
    tf = extract_transaction_features(txns)
    temf = extract_temporal_features(txns)

    X, y, f_names = prepare_feature_matrix(gf, tf, temf, accs)
    rule_m = {"precision": 0.5, "recall": 0.5, "f1_score": 0.5}

    results = train_and_evaluate_all_models(G, X, y, rule_m)
    assert len(results["comparison_table"]) == 5  # Rules, RF, XGBoost, GNN, Isolation Forest

    explainer = LedgerTraceExplainer(results["xgb"], f_names)
    exp = explainer.explain("A", X.loc["A"], top_k=3)
    assert len(exp["top_reasons"]) <= 3
    assert "plain_language" in exp["top_reasons"][0]
