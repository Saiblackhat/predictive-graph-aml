# Ledger Trace — Predictive Graph Analytics for Anti-Money Laundering (AML)

Ledger Trace is an end-to-end AML graph analytics and compliance investigation platform built in Python and modern web standards. It analyzes financial transaction flows as directed weighted graphs, detects complex money laundering typologies (circular round-tripping, smurfing/structuring below ₹50,000, fan-in/fan-out dispersal, and multi-hop layering), scores account risk with machine learning models (XGBoost, Random Forest, PyTorch GNN, Isolation Forest), explains risk drivers using SHAP, and allows compliance analysts to manage cases and generate FinCEN-style Suspicious Activity Reports (SAR) in PDF format.

---

## 🏛️ System Architecture

- **Backend**: FastAPI (Python 3.11+ / 3.14 compatible), NetworkX, PyTorch Geometric, XGBoost, Scikit-learn, SHAP, SQLAlchemy (SQLite), ReportLab.
- **Frontend**: Standalone interactive SVG canvas styled with exact Ledger Trace design tokens (IBM Plex fonts, 4px radius, 1px hairline borders, dark/light theme support, INR ₹ currency formatting).
- **Compliance & Reporting**: FinCEN-standard Suspicious Activity Report (SAR) PDF generator with synthetic data disclaimers, audit logging, and analyst labeling feedback.

---

## 🚀 Quickstart

### 1. Run the Application

From `aml-ledger-trace/`:

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at:
- **Interactive Web App**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2. Run Automated Test Suite

```powershell
python -m pytest tests -v
```

All 11 unit and integration test suites cover:
- Synthetic transaction and ground-truth laundering generation.
- Data cleaning, graph construction, and bounded cycle detection.
- Rule-based detection benchmarks.
- ML models (Random Forest, XGBoost, GNN, Isolation Forest) and SHAP explainability.
- Case management workflows and ReportLab SAR PDF compilation.

---

## 🎯 Key Features & Modules

### 1. Graph Detection & Topology Rules
- **Structuring (Smurfing)**: Detects deposits clustered between ₹45,000 and ₹49,999 (avoiding the ₹50,000 mandatory reporting threshold).
- **Circular Round-tripping**: Bounded DFS cycle detector (depth 3 to 5) identifying closed loops returning funds to originators.
- **Fan-in / Fan-out Hubs**: Identifies money mule aggregation nodes and rapid dispersion hubs.
- **Layering Chains**: Tracks rapid pass-through accounts moving >80% of funds within 12 hours.

### 2. Explainable Machine Learning (SHAP)
- Trained models achieve high Recall and Precision on highly imbalanced transaction graphs.
- Every flagged account includes top plain-language drivers (e.g. *"Participates in 4-account directed closed loop: raised risk (+0.54)"*).

### 3. Investigation & Case Management
- **1-Hop and 2-Hop Ego-Network**: Focus dynamically on a specific suspect's immediate counterparties.
- **Case Audit Trail**: Every status transition, note addition, and analyst label is persisted and timestamped.
- **SAR PDF Generator**: 1-click download of compliance-ready SAR documentation (`POST /api/cases/{id}/sar`).
- **Data Export**: Export flagged suspect registries to CSV and Excel format (`/api/export?format=xlsx`).
