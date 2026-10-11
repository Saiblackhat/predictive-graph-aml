"""
Synthetic Transaction and Laundering Pattern Generator.

This module simulates realistic retail and commercial banking transaction streams
and injects recognizable Anti-Money Laundering (AML) typologies:
  1. Structuring / Smurfing: Keeping deposits just under a regulatory threshold ($10,000).
  2. Fan-In / Fan-Out: Multiple source accounts sending into a mule aggregator,
     which subsequently fans funds out to multiple destinations.
  3. Circular Transfers: Round-tripping funds (A -> B -> C -> A) to disguise beneficial ownership.
  4. Layering Chains: Rapid sequential multi-hop transfers (A -> B -> C -> D) over short intervals.

All generated records include explicit ground-truth flags ('is_laundering', 'typology')
for scientific model training and evaluation.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def generate_synthetic_data(
    num_accounts: int = 5000,
    num_transactions: int = 50000,
    laundering_ratio: float = 0.025,
    random_seed: int = 42,
    start_date: str = "2026-01-01 00:00:00",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic transactions with realistic background activity and injected AML typologies.

    Parameters:
    -----------
    num_accounts : int
        Number of distinct legitimate accounts to create (e.g., 5,000).
    num_transactions : int
        Approximate total transactions to generate (~50,000).
    laundering_ratio : float
        Fraction of transactions that should be part of injected laundering typologies (~2-3%).
    random_seed : int
        Seed for reproducibility.
    start_date : str
        Base start timestamp for transaction timestamps.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - transactions_df: DataFrame of individual transactions with sender, receiver,
          timestamp, amount, transaction_type, is_laundering, typology, etc.
        - accounts_df: Registry of accounts with account_id, true_label (is_laundering),
          account_type, and assigned typology.
    """
    rng = np.random.default_rng(random_seed)
    base_time = datetime.datetime.fromisoformat(start_date)

    # 1. Generate Account Registry
    account_ids = [f"ACC_{i:05d}" for i in range(num_accounts)]
    account_types = rng.choice(
        ["INDIVIDUAL", "BUSINESS", "MERCHANT", "SHELL_COMPANY"],
        size=num_accounts,
        p=[0.75, 0.20, 0.04, 0.01],
    )
    
    # Store account-level metadata
    account_meta: Dict[str, Dict] = {
        acc_id: {
            "account_id": acc_id,
            "account_type": acc_type,
            "is_laundering": 0,
            "typology": "BENIGN",
        }
        for acc_id, acc_type in zip(account_ids, account_types)
    }

    target_laundering_txns = int(num_transactions * laundering_ratio)
    target_benign_txns = num_transactions - target_laundering_txns

    # -------------------------------------------------------------
    # 2. Generate Benign Background Transactions
    # -------------------------------------------------------------
    # Realistic amounts follow a lognormal distribution: median ~$120, heavy tail
    benign_amounts = np.round(rng.lognormal(mean=4.8, sigma=1.2, size=target_benign_txns), 2)
    # Filter negative/zero/extreme unrealistic amounts
    benign_amounts = np.clip(benign_amounts, 5.0, 50000.0)

    # Sender and Receiver choices: Prefer popular merchants (Power law behavior)
    weights = 1.0 / (np.arange(1, num_accounts + 1) ** 0.6)
    weights /= weights.sum()

    senders = rng.choice(account_ids, size=target_benign_txns, p=weights)
    receivers = rng.choice(account_ids, size=target_benign_txns, p=weights)

    # Prevent self-transfers for benign traffic
    mask_self = senders == receivers
    while np.any(mask_self):
        receivers[mask_self] = rng.choice(account_ids, size=np.sum(mask_self), p=weights)
        mask_self = senders == receivers

    # Timestamps spread over 90 days with Poisson-like arrivals
    random_seconds = np.sort(rng.uniform(0, 90 * 86400, size=target_benign_txns))
    benign_txns = []
    txn_types = ["TRANSFER", "WIRE", "PAYMENT", "CASH_DEPOSIT"]
    txn_type_probs = [0.45, 0.25, 0.25, 0.05]

    for i in range(target_benign_txns):
        txn_time = base_time + datetime.timedelta(seconds=float(random_seconds[i]))
        txn_type = rng.choice(txn_types, p=txn_type_probs)
        benign_txns.append(
            {
                "transaction_id": f"TXN_{i:07d}",
                "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id": senders[i],
                "receiver_id": receivers[i],
                "amount": float(benign_amounts[i]),
                "transaction_type": txn_type,
                "is_laundering": 0,
                "typology": "BENIGN",
            }
        )

    # -------------------------------------------------------------
    # 3. Inject Money Laundering Typologies
    # -------------------------------------------------------------
    laundering_txns: List[Dict] = []
    current_txn_idx = target_benign_txns

    # Allocate proportions to 4 typologies:
    # 25% Structuring, 25% Fan-in/out, 25% Cycles, 25% Layering
    quota_per_pattern = target_laundering_txns // 4

    # Available pool of suspicious accounts (allocate new or dedicated accounts)
    suspicious_pool_size = 250
    suspicious_accounts = [f"ACC_LAUND_{i:04d}" for i in range(suspicious_pool_size)]
    for s_acc in suspicious_accounts:
        account_meta[s_acc] = {
            "account_id": s_acc,
            "account_type": "INDIVIDUAL",
            "is_laundering": 1,
            "typology": "SUSPICIOUS",
        }

    pool_ptr = 0

    # TYPOLOGY 1: Structuring / Smurfing
    # Multiple deposits just below CTR (Currency Transaction Report) threshold of $10,000
    # e.g., amounts between $9,000 and $9,950 within hours or few days
    smurf_count = 0
    while smurf_count < quota_per_pattern:
        aggregator = suspicious_accounts[pool_ptr % suspicious_pool_size]
        pool_ptr += 1
        account_meta[aggregator]["typology"] = "STRUCTURING"
        account_meta[aggregator]["is_laundering"] = 1

        # 4 to 8 smurf accounts sending to the aggregator
        num_smurfs = rng.integers(4, 9)
        cluster_base_seconds = rng.uniform(0, 85 * 86400)
        
        for k in range(num_smurfs):
            smurf_acc = suspicious_accounts[pool_ptr % suspicious_pool_size]
            pool_ptr += 1
            account_meta[smurf_acc]["typology"] = "STRUCTURING"
            account_meta[smurf_acc]["is_laundering"] = 1

            amount = float(np.round(rng.uniform(9100.0, 9950.0), 2))
            txn_time = base_time + datetime.timedelta(
                seconds=float(cluster_base_seconds + rng.uniform(0, 3600 * 24))
            )
            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": smurf_acc,
                    "receiver_id": aggregator,
                    "amount": amount,
                    "transaction_type": "CASH_DEPOSIT" if rng.random() > 0.4 else "TRANSFER",
                    "is_laundering": 1,
                    "typology": "STRUCTURING",
                }
            )
            current_txn_idx += 1
            smurf_count += 1
            if smurf_count >= quota_per_pattern:
                break

    # TYPOLOGY 2: Fan-In / Fan-Out Hubs
    # Inflow from many senders to hub account, followed quickly by outflow to many receivers
    fan_count = 0
    while fan_count < quota_per_pattern:
        hub_acc = suspicious_accounts[pool_ptr % suspicious_pool_size]
        pool_ptr += 1
        account_meta[hub_acc]["typology"] = "FAN_IN_FAN_OUT"
        account_meta[hub_acc]["is_laundering"] = 1

        in_sources = rng.integers(5, 10)
        out_destinations = rng.integers(5, 10)
        hub_time_start = rng.uniform(0, 85 * 86400)

        # Inflow: Many -> Hub
        inflow_total = 0.0
        for _ in range(in_sources):
            src_acc = suspicious_accounts[pool_ptr % suspicious_pool_size]
            pool_ptr += 1
            account_meta[src_acc]["typology"] = "FAN_IN_FAN_OUT"
            account_meta[src_acc]["is_laundering"] = 1

            amt = float(np.round(rng.uniform(3000.0, 8000.0), 2))
            inflow_total += amt
            t = base_time + datetime.timedelta(
                seconds=float(hub_time_start + rng.uniform(0, 3600 * 12))
            )
            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": src_acc,
                    "receiver_id": hub_acc,
                    "amount": amt,
                    "transaction_type": "WIRE",
                    "is_laundering": 1,
                    "typology": "FAN_IN_FAN_OUT",
                }
            )
            current_txn_idx += 1
            fan_count += 1

        # Outflow: Hub -> Many (Dispersal minus commission/friction)
        outflow_amt = np.round((inflow_total * 0.96) / out_destinations, 2)
        for _ in range(out_destinations):
            dst_acc = suspicious_accounts[pool_ptr % suspicious_pool_size]
            pool_ptr += 1
            account_meta[dst_acc]["typology"] = "FAN_IN_FAN_OUT"
            account_meta[dst_acc]["is_laundering"] = 1

            t = base_time + datetime.timedelta(
                seconds=float(hub_time_start + 3600 * 13 + rng.uniform(0, 3600 * 12))
            )
            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": hub_acc,
                    "receiver_id": dst_acc,
                    "amount": float(outflow_amt),
                    "transaction_type": "TRANSFER",
                    "is_laundering": 1,
                    "typology": "FAN_IN_FAN_OUT",
                }
            )
            current_txn_idx += 1
            fan_count += 1

    # TYPOLOGY 3: Circular Transfers (Round-Tripping: A -> B -> C -> A)
    cycle_count = 0
    while cycle_count < quota_per_pattern:
        cycle_len = rng.choice([3, 4, 5])
        cycle_nodes = [
            suspicious_accounts[(pool_ptr + c) % suspicious_pool_size] for c in range(cycle_len)
        ]
        pool_ptr += cycle_len

        for node in cycle_nodes:
            account_meta[node]["typology"] = "CIRCULAR"
            account_meta[node]["is_laundering"] = 1

        cycle_start_seconds = rng.uniform(0, 85 * 86400)
        curr_time = cycle_start_seconds
        base_cycle_amt = rng.uniform(15000.0, 50000.0)

        for step in range(cycle_len):
            u = cycle_nodes[step]
            v = cycle_nodes[(step + 1) % cycle_len]
            curr_time += rng.uniform(600, 3600 * 6)  # 10 min to 6 hours between hops
            amt = float(np.round(base_cycle_amt * (0.99 ** step), 2))
            txn_time = base_time + datetime.timedelta(seconds=float(curr_time))

            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": u,
                    "receiver_id": v,
                    "amount": amt,
                    "transaction_type": "WIRE",
                    "is_laundering": 1,
                    "typology": "CIRCULAR",
                }
            )
            current_txn_idx += 1
            cycle_count += 1

    # TYPOLOGY 4: Layering Chains (Multi-Hop Rapid Pass-Through)
    chain_count = 0
    while chain_count < quota_per_pattern:
        chain_len = rng.choice([4, 5, 6])
        chain_nodes = [
            suspicious_accounts[(pool_ptr + c) % suspicious_pool_size] for c in range(chain_len)
        ]
        pool_ptr += chain_len

        for node in chain_nodes:
            account_meta[node]["typology"] = "LAYERING"
            account_meta[node]["is_laundering"] = 1

        chain_start_seconds = rng.uniform(0, 85 * 86400)
        curr_time = chain_start_seconds
        chain_amt = rng.uniform(20000.0, 60000.0)

        for step in range(chain_len - 1):
            u = chain_nodes[step]
            v = chain_nodes[step + 1]
            curr_time += rng.uniform(300, 3600 * 3)  # Rapid hops within 5 min to 3 hrs
            amt = float(np.round(chain_amt * (0.985 ** step), 2))
            txn_time = base_time + datetime.timedelta(seconds=float(curr_time))

            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": u,
                    "receiver_id": v,
                    "amount": amt,
                    "transaction_type": "TRANSFER",
                    "is_laundering": 1,
                    "typology": "LAYERING",
                }
            )
            current_txn_idx += 1
            chain_count += 1

    # Combine benign and laundering transactions and sort chronologically
    all_txns = benign_txns + laundering_txns
    transactions_df = pd.DataFrame(all_txns)
    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
    transactions_df = transactions_df.sort_values(by="timestamp").reset_index(drop=True)
    # Refresh transaction IDs in chronological sequence
    transactions_df["transaction_id"] = [f"TXN_{i:07d}" for i in range(len(transactions_df))]

    # Build account DataFrame
    accounts_df = pd.DataFrame(list(account_meta.values()))

    return transactions_df, accounts_df


if __name__ == "__main__":
    txns, accs = generate_synthetic_data(num_accounts=5000, num_transactions=50000)
    print(f"Generated {len(txns)} transactions across {len(accs)} accounts.")
    print(f"Laundering transactions: {txns['is_laundering'].sum()} ({txns['is_laundering'].mean():.2%})")
    print(f"Laundering accounts: {accs['is_laundering'].sum()} ({accs['is_laundering'].mean():.2%})")
    print("Typology breakdown:")
    print(txns["typology"].value_counts())
