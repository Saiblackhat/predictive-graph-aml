"""
Synthetic Transaction and Laundering Pattern Generator for Ledger Trace (Phase 1).

Generates realistic background financial traffic formatted in Indian Rupees (INR - ₹)
with injected money-laundering typologies:
  1. Structuring / Smurfing: Clustered deposits deliberately kept just below the statutory threshold of ₹50,000 (e.g. ₹46,000 to ₹49,800).
  2. Fan-In / Fan-Out: Multiple mule sources depositing into an aggregator account, which rapidly disburses funds outward.
  3. Circular Transfers: Round-tripping loops (A -> B -> C -> A) designed to obfuscate beneficial ownership.
  4. Layering Chains: Rapid sequential multi-hop transfers (A -> B -> C -> D -> E) with low fund retention.

All injected entities are labeled with ground truth ('is_laundering', 'typology') for objective model evaluation.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def generate_synthetic_transactions(
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
        Number of accounts in the baseline registry.
    num_transactions : int
        Total transactions to generate (~50,000).
    laundering_ratio : float
        Fraction of transactions that represent money laundering (~2-3%).
    random_seed : int
        Seed for deterministic reproducibility.
    start_date : str
        Start timestamp.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - transactions_df: All individual transactions with timestamps, amounts (INR), senders, receivers, labels.
        - accounts_df: Registry of accounts with true labels and typologies.
    """
    rng = np.random.default_rng(random_seed)
    base_time = datetime.datetime.fromisoformat(start_date)

    account_ids = [f"ACC_{i:05d}" for i in range(num_accounts)]
    account_types = rng.choice(
        ["INDIVIDUAL", "BUSINESS", "MERCHANT", "SHELL_COMPANY"],
        size=num_accounts,
        p=[0.75, 0.20, 0.04, 0.01],
    )

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

    # 1. Benign Background Transactions (INR scale, lognormal distribution, median ~₹1,200)
    benign_amounts = np.round(rng.lognormal(mean=7.1, sigma=1.2, size=target_benign_txns), 2)
    benign_amounts = np.clip(benign_amounts, 50.0, 300000.0)

    weights = 1.0 / (np.arange(1, num_accounts + 1) ** 0.6)
    weights /= weights.sum()

    senders = rng.choice(account_ids, size=target_benign_txns, p=weights)
    receivers = rng.choice(account_ids, size=target_benign_txns, p=weights)

    mask_self = senders == receivers
    while np.any(mask_self):
        receivers[mask_self] = rng.choice(account_ids, size=np.sum(mask_self), p=weights)
        mask_self = senders == receivers

    random_seconds = np.sort(rng.uniform(0, 90 * 86400, size=target_benign_txns))
    benign_txns = []
    txn_types = ["UPI", "NEFT", "IMPS", "RTGS", "CASH_DEPOSIT"]
    txn_type_probs = [0.45, 0.25, 0.20, 0.05, 0.05]

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

    # 2. Inject Money Laundering Typologies
    laundering_txns: List[Dict] = []
    current_txn_idx = target_benign_txns
    quota_per_pattern = target_laundering_txns // 4

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

    # Typology A: Structuring / Smurfing (just below statutory threshold of ₹50,000)
    # Amounts between ₹46,000 and ₹49,850
    smurf_count = 0
    while smurf_count < quota_per_pattern:
        aggregator = suspicious_accounts[pool_ptr % suspicious_pool_size]
        pool_ptr += 1
        account_meta[aggregator]["typology"] = "STRUCTURING"
        account_meta[aggregator]["is_laundering"] = 1

        num_smurfs = rng.integers(4, 8)
        cluster_base_seconds = rng.uniform(0, 85 * 86400)

        for _ in range(num_smurfs):
            smurf_acc = suspicious_accounts[pool_ptr % suspicious_pool_size]
            pool_ptr += 1
            account_meta[smurf_acc]["typology"] = "STRUCTURING"
            account_meta[smurf_acc]["is_laundering"] = 1

            amt = float(np.round(rng.uniform(46500.0, 49850.0), 2))
            txn_time = base_time + datetime.timedelta(
                seconds=float(cluster_base_seconds + rng.uniform(0, 3600 * 24))
            )
            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": smurf_acc,
                    "receiver_id": aggregator,
                    "amount": amt,
                    "transaction_type": "CASH_DEPOSIT" if rng.random() > 0.4 else "IMPS",
                    "is_laundering": 1,
                    "typology": "STRUCTURING",
                }
            )
            current_txn_idx += 1
            smurf_count += 1
            if smurf_count >= quota_per_pattern:
                break

    # Typology B: Fan-In / Fan-Out
    fan_count = 0
    while fan_count < quota_per_pattern:
        hub_acc = suspicious_accounts[pool_ptr % suspicious_pool_size]
        pool_ptr += 1
        account_meta[hub_acc]["typology"] = "FAN_IN_FAN_OUT"
        account_meta[hub_acc]["is_laundering"] = 1

        in_sources = rng.integers(5, 9)
        out_destinations = rng.integers(5, 9)
        hub_start_seconds = rng.uniform(0, 85 * 86400)

        inflow_total = 0.0
        for _ in range(in_sources):
            src_acc = suspicious_accounts[pool_ptr % suspicious_pool_size]
            pool_ptr += 1
            account_meta[src_acc]["typology"] = "FAN_IN_FAN_OUT"
            account_meta[src_acc]["is_laundering"] = 1

            amt = float(np.round(rng.uniform(15000.0, 45000.0), 2))
            inflow_total += amt
            t = base_time + datetime.timedelta(seconds=float(hub_start_seconds + rng.uniform(0, 3600 * 12)))
            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": src_acc,
                    "receiver_id": hub_acc,
                    "amount": amt,
                    "transaction_type": "NEFT",
                    "is_laundering": 1,
                    "typology": "FAN_IN_FAN_OUT",
                }
            )
            current_txn_idx += 1
            fan_count += 1

        outflow_amt = np.round((inflow_total * 0.96) / out_destinations, 2)
        for _ in range(out_destinations):
            dst_acc = suspicious_accounts[pool_ptr % suspicious_pool_size]
            pool_ptr += 1
            account_meta[dst_acc]["typology"] = "FAN_IN_FAN_OUT"
            account_meta[dst_acc]["is_laundering"] = 1

            t = base_time + datetime.timedelta(seconds=float(hub_start_seconds + 3600 * 13 + rng.uniform(0, 3600 * 12)))
            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": hub_acc,
                    "receiver_id": dst_acc,
                    "amount": float(outflow_amt),
                    "transaction_type": "UPI",
                    "is_laundering": 1,
                    "typology": "FAN_IN_FAN_OUT",
                }
            )
            current_txn_idx += 1
            fan_count += 1

    # Typology C: Circular Transfers (Round-Tripping: A -> B -> C -> A)
    cycle_count = 0
    while cycle_count < quota_per_pattern:
        cycle_len = rng.choice([3, 4, 5])
        cycle_nodes = [suspicious_accounts[(pool_ptr + c) % suspicious_pool_size] for c in range(cycle_len)]
        pool_ptr += cycle_len

        for node in cycle_nodes:
            account_meta[node]["typology"] = "CIRCULAR"
            account_meta[node]["is_laundering"] = 1

        cycle_start = rng.uniform(0, 85 * 86400)
        curr_time = cycle_start
        base_amt = rng.uniform(75000.0, 250000.0)

        for step in range(cycle_len):
            u = cycle_nodes[step]
            v = cycle_nodes[(step + 1) % cycle_len]
            curr_time += rng.uniform(600, 3600 * 6)
            amt = float(np.round(base_amt * (0.99 ** step), 2))
            txn_time = base_time + datetime.timedelta(seconds=float(curr_time))

            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": u,
                    "receiver_id": v,
                    "amount": amt,
                    "transaction_type": "RTGS",
                    "is_laundering": 1,
                    "typology": "CIRCULAR",
                }
            )
            current_txn_idx += 1
            cycle_count += 1

    # Typology D: Layering Chains (Rapid Multi-Hop)
    chain_count = 0
    while chain_count < quota_per_pattern:
        chain_len = rng.choice([4, 5, 6])
        chain_nodes = [suspicious_accounts[(pool_ptr + c) % suspicious_pool_size] for c in range(chain_len)]
        pool_ptr += chain_len

        for node in chain_nodes:
            account_meta[node]["typology"] = "LAYERING"
            account_meta[node]["is_laundering"] = 1

        chain_start = rng.uniform(0, 85 * 86400)
        curr_time = chain_start
        base_amt = rng.uniform(100000.0, 350000.0)

        for step in range(chain_len - 1):
            u = chain_nodes[step]
            v = chain_nodes[step + 1]
            curr_time += rng.uniform(300, 3600 * 3)  # Rapid hops within 3 hours
            amt = float(np.round(base_amt * (0.985 ** step), 2))
            txn_time = base_time + datetime.timedelta(seconds=float(curr_time))

            laundering_txns.append(
                {
                    "transaction_id": f"TXN_{current_txn_idx:07d}",
                    "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": u,
                    "receiver_id": v,
                    "amount": amt,
                    "transaction_type": "IMPS",
                    "is_laundering": 1,
                    "typology": "LAYERING",
                }
            )
            current_txn_idx += 1
            chain_count += 1

    all_txns = benign_txns + laundering_txns
    txns_df = pd.DataFrame(all_txns)
    txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])
    txns_df = txns_df.sort_values(by="timestamp").reset_index(drop=True)
    txns_df["transaction_id"] = [f"TXN_{i:07d}" for i in range(len(txns_df))]

    accs_df = pd.DataFrame(list(account_meta.values()))

    return txns_df, accs_df
