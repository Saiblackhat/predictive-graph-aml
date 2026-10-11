"""
Model Training, GNN, Isolation Forest, and Risk Scoring Module for aml-ledger-trace (Phase 4).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
import networkx as nx


# 1. Feature Matrix Assembler
def prepare_feature_matrix(
    graph_df: pd.DataFrame,
    txn_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    merged = pd.merge(accounts_df[["account_id", "is_laundering"]], graph_df, on="account_id", how="left")
    merged = pd.merge(merged, txn_df, on="account_id", how="left")
    merged = pd.merge(merged, temp_df, on="account_id", how="left").fillna(0.0)

    feature_cols = [c for c in merged.columns if c not in ["account_id", "is_laundering", "account_type", "typology"]]
    X = merged[feature_cols].copy()
    y = merged["is_laundering"].astype(int)
    X.index = merged["account_id"]
    y.index = merged["account_id"]

    return X, y, feature_cols


# 2. GNN Architecture
class GNNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) + self.bias
        return output


class GNNNodeClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = GNNLayer(in_features, hidden_dim)
        self.conv2 = GNNLayer(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, adj))
        logits = self.conv2(h, adj)
        return logits.squeeze(-1)


# 3. Comprehensive Multi-Model Trainer
def train_and_evaluate_all_models(
    G: nx.DiGraph,
    X: pd.DataFrame,
    y: pd.Series,
    rule_metrics: Dict[str, Any],
    model_save_dir: str = "reports/models",
    random_seed: int = 42,
) -> Dict[str, Any]:
    save_path = Path(model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_seed
    )

    pos_scale = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=random_seed)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    # Model 2: XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=pos_scale, random_state=random_seed, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    # Model 3: Isolation Forest
    iso = IsolationForest(contamination=0.03, random_state=random_seed)
    iso.fit(X_train)
    iso_preds = (iso.predict(X_test) == -1).astype(int)
    iso_scores = -iso.decision_function(X_test)

    # Model 4: PyTorch GNN
    node_list = list(X.index)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    adj = np.eye(n, dtype=np.float32)
    undir = G.to_undirected()
    for u, v in undir.edges():
        if u in node_to_idx and v in node_to_idx:
            adj[node_to_idx[u], node_to_idx[v]] = 1.0
            adj[node_to_idx[v], node_to_idx[u]] = 1.0
    deg = np.sum(adj, axis=1)
    deg_inv = np.zeros_like(deg)
    deg_inv[deg > 0] = np.power(deg[deg > 0], -0.5)
    norm_adj = np.diag(deg_inv) @ adj @ np.diag(deg_inv)
    adj_tensor = torch.FloatTensor(norm_adj)

    feat_norm = (X.values - X.values.mean(axis=0)) / (X.values.std(axis=0) + 1e-7)
    x_tensor = torch.FloatTensor(feat_norm)
    y_tensor = torch.FloatTensor(y.values)

    train_idx = [node_to_idx[n] for n in X_train.index]
    test_idx = [node_to_idx[n] for n in X_test.index]

    gnn = GNNNodeClassifier(in_features=X.shape[1], hidden_dim=32)
    opt = torch.optim.Adam(gnn.parameters(), lr=0.01)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_scale]))

    gnn.train()
    for _ in range(60):
        opt.zero_grad()
        out = gnn(x_tensor, adj_tensor)
        loss = crit(out[torch.LongTensor(train_idx)], y_tensor[torch.LongTensor(train_idx)])
        loss.backward()
        opt.step()

    gnn.eval()
    with torch.no_grad():
        logits = gnn(x_tensor, adj_tensor)
        gnn_all_probs = torch.sigmoid(logits).numpy()

    gnn_test_probs = gnn_all_probs[test_idx]
    gnn_test_preds = (gnn_test_probs >= 0.5).astype(int)

    # Save models
    joblib.dump(rf, save_path / "random_forest_model.joblib")
    joblib.dump(xgb, save_path / "xgboost_model.joblib")
    joblib.dump(iso, save_path / "isolation_forest.joblib")
    torch.save(gnn.state_dict(), save_path / "gnn_model.pth")

    def _m(y_true, y_p, y_pr):
        return {
            "precision": float(precision_score(y_true, y_p, zero_division=0)),
            "recall": float(recall_score(y_true, y_p, zero_division=0)),
            "f1": float(f1_score(y_true, y_p, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, y_pr)),
        }

    rf_metrics = _m(y_test, rf_pred, rf_prob)
    xgb_metrics = _m(y_test, xgb_pred, xgb_prob)
    gnn_metrics = _m(y_test, gnn_test_preds, gnn_test_probs)
    iso_metrics = _m(y_test, iso_preds, iso_scores)

    comparison_table = [
        {"model": "Rule-Based Baseline", "precision": rule_metrics["precision"], "recall": rule_metrics["recall"], "f1": rule_metrics["f1_score"], "pr_auc": None},
        {"model": "Random Forest", "precision": rf_metrics["precision"], "recall": rf_metrics["recall"], "f1": rf_metrics["f1"], "pr_auc": rf_metrics["pr_auc"]},
        {"model": "XGBoost", "precision": xgb_metrics["precision"], "recall": xgb_metrics["recall"], "f1": xgb_metrics["f1"], "pr_auc": xgb_metrics["pr_auc"]},
        {"model": "GraphSAGE / GCN", "precision": gnn_metrics["precision"], "recall": gnn_metrics["recall"], "f1": gnn_metrics["f1"], "pr_auc": gnn_metrics["pr_auc"]},
        {"model": "Isolation Forest", "precision": iso_metrics["precision"], "recall": iso_metrics["recall"], "f1": iso_metrics["f1"], "pr_auc": iso_metrics["pr_auc"]},
    ]

    return {
        "rf": rf,
        "xgb": xgb,
        "iso": iso,
        "gnn": gnn,
        "comparison_table": comparison_table,
        "all_xgb_probs": xgb.predict_proba(X)[:, 1],
    }
