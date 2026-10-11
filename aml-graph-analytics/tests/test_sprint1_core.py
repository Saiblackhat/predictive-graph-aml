"""
Unit tests for Sprint 1: Explainability, Anomaly Detection, Temporal Features, and GNN.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pytest
from src.features.temporal_features import extract_temporal_features
from src.models.anomaly import train_anomaly_detector, compute_anomaly_flags
from src.models.explainability import AMLExplainabilityEngine
from src.models.gnn import train_gnn_model
from xgboost import XGBClassifier


def test_temporal_feature_extraction():
    txns = pd.DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 1000.0, "timestamp": "2026-01-01 10:00:00", "txn_hour": 10},
            {"sender_id": "A", "receiver_id": "C", "amount": 2000.0, "timestamp": "2026-01-01 11:00:00", "txn_hour": 11},
            {"sender_id": "A", "receiver_id": "D", "amount": 500.0, "timestamp": "2026-01-01 23:00:00", "txn_hour": 23},  # off-hours
            {"sender_id": "B", "receiver_id": "C", "amount": 50.0, "timestamp": "2026-01-02 14:00:00", "txn_hour": 14},
        ]
    )
    temp_feats = extract_temporal_features(txns)
    assert "transaction_velocity" in temp_feats.columns
    assert "burst_score" in temp_feats.columns
    assert "off_hours_ratio" in temp_feats.columns
    assert "dormant_then_active" in temp_feats.columns

    a_row = temp_feats[temp_feats["account_id"] == "A"].iloc[0]
    assert a_row["off_hours_ratio"] > 0.0


def test_anomaly_detection():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, size=(100, 5)), columns=[f"f_{i}" for i in range(5)])
    X.index = [f"ACC_{i}" for i in range(100)]
    # Inject obvious outlier
    X.iloc[0] = 50.0

    scored_accounts = pd.DataFrame(
        {"account_id": X.index, "aml_risk_score": 10.0, "typology": "BENIGN"}
    )

    iso = train_anomaly_detector(X, contamination=0.05, random_seed=42)
    flagged = compute_anomaly_flags(iso, X, scored_accounts, supervised_score_threshold=35.0)

    assert "anomaly_score" in flagged.columns
    assert "is_unknown_anomaly" in flagged.columns
    # Outlier should have been caught as unknown anomaly
    assert flagged[flagged["account_id"] == "ACC_0"]["is_unknown_anomaly"].iloc[0] == 1


def test_shap_explainability():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, size=(50, 4)), columns=["cycle_participation_count", "total_structuring_count", "pagerank", "burst_score"])
    y = pd.Series((X["cycle_participation_count"] > 0).astype(int))

    xgb = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    xgb.fit(X, y)

    engine = AMLExplainabilityEngine(xgb, feature_names=list(X.columns))
    explanation = engine.explain_account("ACC_TEST", X.iloc[0], top_k=3)

    assert explanation["account_id"] == "ACC_TEST"
    assert len(explanation["top_contributions"]) <= 3
    assert "plain_language" in explanation["top_contributions"][0]


def test_gnn_model_training():
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")
    for i in range(20):
        G.add_edge(f"U_{i}", f"V_{i}")

    nodes = list(G.nodes())
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, size=(len(nodes), 4)), index=nodes, columns=[f"f_{i}" for i in range(4)])
    y = pd.Series([1 if n in ["A", "B", "C"] else 0 for n in nodes], index=nodes)

    train_idx = list(range(15))
    test_idx = list(range(15, len(nodes)))

    results = train_gnn_model(G, X, y, train_idx, test_idx, epochs=20, random_seed=42)
    assert results["model"] is not None
    assert "f1_score" in results
    assert len(results["all_probs"]) == len(nodes)
