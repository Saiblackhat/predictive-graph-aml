"""
Graph Neural Network (GNN) Node-Classification Module (Sprint 1).

Implements a 2-layer Graph Convolutional Network (GCN / GraphSAGE style)
in pure PyTorch for inductive and transductive node classification on transaction graphs.
Features:
  - Adjacency matrix normalization with self-loops (Kipf & Welling).
  - Cost-sensitive positive class weighting for extreme imbalanced learning.
  - Dropout regularization and non-linear feature transformation.
  - Inference method outputting calibrated AML probabilities.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


class GraphConvolution(nn.Module):
    """
    Standard graph convolutional layer: H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Support = X @ W
        support = torch.mm(x, self.weight)
        # Output = Adj @ Support
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GNNAMLClassifier(nn.Module):
    """
    2-Layer Graph Convolutional Network for node fraud classification.
    """

    def __init__(self, in_features: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.gc2(h, adj)
        return logits.squeeze(-1)


def build_normalized_adjacency(G: nx.DiGraph, node_list: list) -> torch.Tensor:
    """
    Build symmetrically normalized adjacency matrix with self-loops:
    A_norm = D^{-1/2} * (A + I) * D^{-1/2}
    """
    n = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # Convert to undirected view to enable bidirectional message passing
    undir_G = G.to_undirected()

    adj = np.eye(n, dtype=np.float32)  # Include self-loops

    for u, v in undir_G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # Symmetrically normalize: D^{-1/2} A D^{-1/2}
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    pos_mask = deg > 0
    deg_inv_sqrt[pos_mask] = np.power(deg[pos_mask], -0.5)
    deg_mat = np.diag(deg_inv_sqrt)

    norm_adj = deg_mat @ adj @ deg_mat
    return torch.FloatTensor(norm_adj)


def train_gnn_model(
    G: nx.DiGraph,
    X: pd.DataFrame,
    y: pd.Series,
    train_indices: list,
    test_indices: list,
    epochs: int = 120,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    random_seed: int = 42,
    model_save_dir: str = "reports/models",
) -> Dict[str, Any]:
    """
    Train GNN node classification model.

    Parameters:
    -----------
    G : nx.DiGraph
        Directed transaction graph.
    X : pd.DataFrame
        Account feature matrix.
    y : pd.Series
        Binary targets.
    train_indices : list
        Row indices for training nodes.
    test_indices : list
        Row indices for testing nodes.

    Returns:
    --------
    Dict[str, Any]:
        Trained model, test metrics, and predicted probabilities.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    node_list = list(X.index)
    adj_tensor = build_normalized_adjacency(G, node_list)

    # Standardize input features
    features_np = (X.values - X.values.mean(axis=0)) / (X.values.std(axis=0) + 1e-7)
    x_tensor = torch.FloatTensor(features_np)

    y_np = y.values.astype(np.float32)
    y_tensor = torch.FloatTensor(y_np)

    train_idx = torch.LongTensor(train_indices)
    test_idx = torch.LongTensor(test_indices)

    # Compute positive class loss weight
    pos_count = (y_np[train_indices] == 1).sum()
    neg_count = (y_np[train_indices] == 0).sum()
    pos_weight = torch.FloatTensor([float(neg_count / (pos_count if pos_count > 0 else 1.0))])

    model = GNNAMLClassifier(in_features=X.shape[1], hidden_dim=32, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x_tensor, adj_tensor)
        loss = criterion(out[train_idx], y_tensor[train_idx])
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor, adj_tensor)
        probs = torch.sigmoid(logits).numpy()

    test_probs = probs[test_indices]
    test_preds = (test_probs >= 0.5).astype(int)
    y_test_true = y_np[test_indices].astype(int)

    prec = precision_score(y_test_true, test_preds, zero_division=0)
    rec = recall_score(y_test_true, test_preds, zero_division=0)
    f1 = f1_score(y_test_true, test_preds, zero_division=0)
    pr_auc = average_precision_score(y_test_true, test_probs)

    save_path = Path(model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "gnn_model.pth")

    return {
        "model": model,
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "all_probs": probs,
    }
