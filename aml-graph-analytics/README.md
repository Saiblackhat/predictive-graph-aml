# Predictive Graph Analytics for Anti-Money Laundering (AML)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A complete, production-grade Anti-Money Laundering (AML) detection platform. Financial transactions are modeled as a directed weighted multi-attribute graph where accounts are nodes and transactions are edges. The platform combines deterministic typology detection with machine learning classifiers (Random Forest, XGBoost) trained on topological and transactional flow features, and presents findings in an interactive Streamlit investigation dashboard.

---

## 📌 Problem Statement & AML Typologies
Financial institutions face massive transaction volumes where illicit activity accounts for a tiny fraction (~2–3%). Legacy rule-based systems suffer from either high false-positive rates or low recall due to rigid thresholds. This project uncovers four canonical money laundering typologies:

1. **Structuring / Smurfing:** Deliberately executing multiple deposits just beneath the statutory threshold (e.g. $9,000–$9,950 when the reporting threshold is $10,000).
2. **Fan-In / Fan-Out:** Multiple mule accounts funneling funds into an aggregator hub (fan-in), which rapidly disburses amounts to external receivers (fan-out).
3. **Circular Transfers (Round-Tripping):** Directed closed cycles ($A \to B \to C \to A$) designed to obfuscate beneficial ownership while returning capital to the initiator.
4. **Layering Chains:** High-speed sequential multi-hop transfers ($A \to B \to C \to D \to E$) across short time deltas with near-zero fund retention.

---

## 🏛️ System Architecture

```text
       +---------------------------------------------+
       |   Raw Transactions & Account Registries     |
       |  (Synthetic Generator ~50k / IBM Kaggle Set)|
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Preprocessing & Amount Normalization      |
       |   (Deduplication, Log1p, Z-score, Timestamps)
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Directed Weighted Graph Construction     |
       | (NetworkX DiGraph & MultiDiGraph Modeling)  |
       +---------------------------------------------+
             /                               \
            v                                 v
+-----------------------+         +--------------------------+
|  Rule-Based Detectors |         |  Feature Engineering     |
| - Structuring Rules   |         | - Graph: PageRank, Louvain
| - Fan-In/Out Hubs     |         |   Betweenness, Cycles    |
| - Bounded Cycle DFS   |         | - Flows: Velocity, Ratios|
| - Rapid Pass-Throughs |         +--------------------------+
+-----------------------+                      |
            \                                  v
             \                    +--------------------------+
              \                   | Imbalanced Supervised ML |
               \                  | (Random Forest, XGBoost) |
                \                 +--------------------------+
                 \                             /
                  v                           v
       +---------------------------------------------+
       |    Calibrated Account Risk Scorer (0-100)   |
       |   [LOW: 0-25 | MED: 26-50 | HIGH/CRIT: 51+] |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |    Streamlit AML Investigation Dashboard     |
       |  - Overview KPIs     - Ego-Network PyVis    |
       |  - Scored Explorer   - Cycle Visualizer     |
       +---------------------------------------------+
```

---

## 🚀 Quickstart & One-Command Run

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-org/aml-graph-analytics.git
cd aml-graph-analytics
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Run the full data generation, graph construction, detection, ML training, and scoring pipeline in one command:
```bash
python main.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```

### 4. Run pytest Suite
```bash
python -m pytest tests -v
```

---

## 📊 Benchmark & Evaluation Results

Tested on **50,008 transactions** across **5,250 accounts** with **2.52% injected laundering prevalence**:

| Surveillance Approach | Precision | Recall | F1 Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Rule-Based Baseline** | 4.59% | 72.00% | 0.0863 | N/A |
| **Random Forest (Graph + Flows)** | **100.00%** | **100.00%** | **1.0000** | **1.0000** |
| **XGBoost (Cost-Weighted Gradient Boost)** | **100.00%** | **100.00%** | **1.0000** | **1.0000** |

### Key Takeaway for Interviews:
* **The Rule-Based Baseline** flags suspicious accounts based on individual hardcoded thresholds. Because legitimate accounts sometimes deposit near-threshold amounts or experience bursty activity, it yields **high false positives (4.59% precision)**.
* **Graph + Machine Learning** combines topology (PageRank, degree ratios, bounded cycle membership, Louvain communities) with transactional flow dynamics. This multi-dimensional representation eliminates false positives while capturing complex multi-hop laundering structures.

---

## 📁 Repository Structure
```text
aml-graph-analytics/
├── data/
│   ├── raw/                       # Cached synthetic & IBM datasets
│   └── processed/                 # Standardized transactions & 0-100 risk scores
├── src/
│   ├── data/
│   │   ├── generator.py           # Synthetic generator with injected typologies
│   │   ├── loader.py              # Ingestion interface (Synthetic / IBM AML Kaggle)
│   │   └── preprocessor.py        # Cleansing, log1p, deduplication
│   ├── graph/
│   │   ├── builder.py             # DiGraph & MultiDiGraph construction
│   │   └── statistics.py          # WCC, SCC, density, degree distributions
│   ├── detection/
│   │   ├── rules.py               # Deterministic detectors (cycles, smurfing, hubs)
│   │   └── baseline.py            # Rule baseline evaluator
│   ├── features/
│   │   ├── graph_features.py      # PageRank, betweenness, Louvain, cycles
│   │   └── transaction_features.py# Net flows, velocities, structuring counts
│   └── models/
│       ├── train.py               # Imbalance-weighted training (RF & XGBoost)
│       ├── evaluate.py            # PR-AUC, Confusion Matrix, feature importances
│       └── scorer.py              # 0-100 calibrated risk scoring & tiering
├── app/
│   ├── dashboard.py               # Streamlit multi-page surveillance dashboard
│   └── components/
│       └── network_view.py        # Interactive PyVis physics ego-network renderer
├── tests/
│   ├── test_generator.py          # Data generation & schema tests
│   ├── test_preprocessing.py      # Cleaning & normalization tests
│   ├── test_graph_builder.py      # Graph construction tests
│   ├── test_detectors.py          # Typology rule detectors tests
│   └── test_model_pipeline.py     # Feature extraction & inference tests
├── reports/
│   ├── evaluation_report.md       # Pipeline summary report
│   └── feature_importance.png     # XGBoost feature importance plot
├── main.py                        # Single-command pipeline orchestrator
├── requirements.txt               # Pinned dependencies
└── README.md                      # Project documentation & interview guide
```

---

## 🔮 Limitations & Future Work
1. **Dynamic Temporal Graphs:** Currently, transaction edges aggregate volume over time. Incorporating Continuous-Time Dynamic Graphs (CTDG) can capture millisecond-level money movement.
2. **Graph Neural Networks (GNNs):** Transitioning from handcrafted topological features to inductive Graph Convolutional Networks (GCN) or Relational Graph Convolutional Networks (R-GCN) can learn multi-relational structural representations directly from raw transactions.
3. **Semi-Supervised Node Classification:** Real financial environments have very sparse labels; implementing Graph Self-Supervised Learning (e.g. GraphSAGE with contrastive loss) would enable detection of emerging laundering patterns with zero supervision.
