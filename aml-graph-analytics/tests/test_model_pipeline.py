"""
Unit tests for Feature Extraction and Model Pipeline.
"""

import pandas as pd
import networkx as nx
from src.features.graph_features import extract_graph_features
from src.features.transaction_features import extract_transaction_features
from src.models.train import prepare_feature_matrix, train_aml_models
from src.models.scorer import compute_account_risk_scores


def test_feature_extraction_and_training():
    # Build mini network
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=100.0)
    G.add_edge("B", "C", weight=200.0)
    G.add_edge("C", "A", weight=300.0)
    for i in range(10):
        G.add_edge(f"U_{i}", f"V_{i}", weight=50.0)

    # Graph features
    g_feats = extract_graph_features(G, approximate_betweenness_k=50)
    assert len(g_feats) == len(G.nodes())
    assert "pagerank" in g_feats.columns
    assert "betweenness_centrality" in g_feats.columns

    # Txn features
    txns = pd.DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 100.0, "transaction_type": "TRANSFER"},
            {"sender_id": "B", "receiver_id": "C", "amount": 200.0, "transaction_type": "TRANSFER"},
            {"sender_id": "C", "receiver_id": "A", "amount": 300.0, "transaction_type": "TRANSFER"},
        ]
        + [{"sender_id": f"U_{i}", "receiver_id": f"V_{i}", "amount": 50.0, "transaction_type": "TRANSFER"} for i in range(10)]
    )
    t_feats = extract_transaction_features(txns)
    assert "total_transacted_amount" in t_feats.columns

    # Prepare feature matrix
    accs = pd.DataFrame(
        [{"account_id": node, "is_laundering": 1 if node in ["A", "B", "C"] else 0} for node in G.nodes()]
    )
    X, y, cols = prepare_feature_matrix(g_feats, t_feats, accs)
    assert len(X) == len(G.nodes())
    assert len(y) == len(G.nodes())

    # Train model
    artifacts = train_aml_models(X, y, test_size=0.3, random_seed=42)
    assert artifacts["rf_model"] is not None
    assert artifacts["xgb_model"] is not None

    # Test risk scorer
    rules_eval = pd.DataFrame(
        [{"account_id": n, "triggered_rules": [], "rule_count": 0, "rule_flagged": 0} for n in G.nodes()]
    )
    scores_df = compute_account_risk_scores(artifacts["xgb_model"], X, accs, rules_eval)
    assert "aml_risk_score" in scores_df.columns
    assert "risk_tier" in scores_df.columns
    assert scores_df["aml_risk_score"].min() >= 0.0
    assert scores_df["aml_risk_score"].max() <= 100.0
