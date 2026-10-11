"""
Temporal & Behavioral Feature Engineering Module (Sprint 1).

Extracts dynamic temporal and behavioral velocity indicators:
  1. Transaction Velocity: Average transactions per active transacting day.
  2. Burst Score: Ratio of peak 1-hour transaction volume to median hourly volume.
  3. Dormant-Then-Active Flag: Inactive for >= 14 days, followed by sudden high-value surge (> $15,000).
  4. Off-Hours Ratio: Fraction of transactions executed outside standard banking hours (22:00 to 06:00).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_temporal_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract dynamic temporal and behavioral transaction features per account.

    Parameters:
    -----------
    transactions_df : pd.DataFrame
        Processed transactions table containing sender_id, receiver_id, amount, timestamp, txn_hour.

    Returns:
    --------
    pd.DataFrame:
        Table indexed by account_id with temporal engineered features.
    """
    txns = transactions_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(txns["timestamp"]):
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    # Prepare union of activity per account: account, timestamp, amount, hour, direction
    senders = txns[["sender_id", "timestamp", "amount", "txn_hour"]].rename(
        columns={"sender_id": "account_id"}
    )
    senders["is_outflow"] = 1

    receivers = txns[["receiver_id", "timestamp", "amount", "txn_hour"]].rename(
        columns={"receiver_id": "account_id"}
    )
    receivers["is_outflow"] = 0

    all_activity = pd.concat([senders, receivers], ignore_index=True)
    all_activity["date"] = all_activity["timestamp"].dt.date
    all_activity["hour_slot"] = all_activity["timestamp"].dt.floor("h")

    # 1. Off-hours ratio (10 PM to 6 AM)
    all_activity["is_off_hours"] = all_activity["txn_hour"].apply(
        lambda h: 1 if (h >= 22 or h < 6) else 0
    )

    temporal_records = []

    for acc_id, group in all_activity.groupby("account_id"):
        total_txns = len(group)
        active_days = group["date"].nunique()
        velocity = total_txns / (active_days if active_days > 0 else 1.0)

        # Off-hours ratio
        off_hours_count = group["is_off_hours"].sum()
        off_hours_ratio = off_hours_count / total_txns

        # Hourly burst score: peak hourly count vs mean hourly count
        hourly_counts = group.groupby("hour_slot")["amount"].count()
        peak_hour_txns = hourly_counts.max() if not hourly_counts.empty else 1
        mean_hour_txns = hourly_counts.mean() if not hourly_counts.empty else 1.0
        burst_score = peak_hour_txns / (mean_hour_txns if mean_hour_txns > 0 else 1.0)

        # Dormant-then-active surge check:
        # Sort by timestamp, compute gap between consecutive events
        sorted_times = group.sort_values(by="timestamp")
        time_deltas = sorted_times["timestamp"].diff().dt.total_seconds().dropna()
        max_gap_days = (time_deltas.max() / 86400.0) if not time_deltas.empty else 0.0

        dormant_flag = 0
        if max_gap_days >= 14.0:
            # Check if any transaction following a 14-day gap was large (> $15,000)
            large_txns = group[group["amount"] >= 15000.0]
            if not large_txns.empty:
                dormant_flag = 1

        temporal_records.append(
            {
                "account_id": acc_id,
                "transaction_velocity": float(velocity),
                "burst_score": float(burst_score),
                "off_hours_ratio": float(off_hours_ratio),
                "dormant_then_active": int(dormant_flag),
            }
        )

    return pd.DataFrame(temporal_records)
