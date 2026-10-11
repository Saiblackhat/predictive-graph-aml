"""
Temporal and Behavioral Feature Engineering Module for aml-ledger-trace (Phase 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_temporal_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    txns = transactions_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(txns["timestamp"]):
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    senders = txns[["sender_id", "timestamp", "amount", "txn_hour"]].rename(columns={"sender_id": "account_id"})
    receivers = txns[["receiver_id", "timestamp", "amount", "txn_hour"]].rename(columns={"receiver_id": "account_id"})

    all_act = pd.concat([senders, receivers], ignore_index=True)
    all_act["date"] = all_act["timestamp"].dt.date
    all_act["hour_slot"] = all_act["timestamp"].dt.floor("h")
    all_act["is_off_hours"] = all_act["txn_hour"].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)

    records = []
    for acc_id, grp in all_act.groupby("account_id"):
        n_tx = len(grp)
        n_days = grp["date"].nunique()
        velocity = n_tx / (n_days if n_days > 0 else 1.0)
        off_hours_ratio = grp["is_off_hours"].sum() / n_tx

        hourly = grp.groupby("hour_slot")["amount"].count()
        pk = hourly.max() if not hourly.empty else 1
        mn = hourly.mean() if not hourly.empty else 1.0
        burst = pk / (mn if mn > 0 else 1.0)

        times = grp.sort_values(by="timestamp")
        deltas = times["timestamp"].diff().dt.total_seconds().dropna()
        max_gap = (deltas.max() / 86400.0) if not deltas.empty else 0.0

        dormant = 1 if (max_gap >= 14.0 and not grp[grp["amount"] >= 50000.0].empty) else 0

        records.append(
            {
                "account_id": acc_id,
                "transaction_velocity": float(velocity),
                "burst_score": float(burst),
                "off_hours_ratio": float(off_hours_ratio),
                "dormant_then_active": int(dormant),
            }
        )

    return pd.DataFrame(records)
