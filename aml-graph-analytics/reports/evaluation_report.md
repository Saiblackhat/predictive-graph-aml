# Anti-Money Laundering (AML) Predictive Graph Analytics Report

## Executive Summary
This report summarizes the results of the AML graph analytics surveillance pipeline executed on a universe of **5,250 accounts** and **50,008 financial transactions**.

## Graph Topological Diagnostics
- **Total Accounts (Nodes):** 5,250
- **Directed Transaction Edges:** 48,984
- **Total Transacted Volume:** $39,762,433.39
- **Graph Density:** 0.001778
- **Weakly Connected Components:** 2
- **Strongly Connected Components:** 43

## Comparative Surveillance Benchmark
| Model / Detection Approach | Precision | Recall | F1 Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Rule-Based Heuristic Baseline** | 0.0459 | 0.7200 | 0.0863 | N/A |
| **Random Forest (Topological + Flow)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost (Cost-Weighted Gradient Boosting)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Key Insights
1. **Recall Superiority:** The Machine Learning models (specifically XGBoost) achieve dramatically higher recall (100.00%) than the rule baseline (72.00%) while sustaining comparable precision.
2. **Top Discriminative Features:** Feature importance analysis reveals graph topological metrics (PageRank, cycle participation, in/out degree ratios) alongside structuring transaction frequency are the strongest signals for identifying money-laundering entities.
3. **Investigation Efficiency:** Analysts can prioritize entities in the `CRITICAL` risk tier (Score >= 75) in the interactive Streamlit dashboard.
