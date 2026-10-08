"""
features.py
Pairwise feature engineering for (Source-1, candidate) record pairs.
Pure string/rule-based features -- no external data.
"""
import difflib
import numpy as np
import pandas as pd

from normalize import normalize_name, core_name, normalize_address, extract_postal, tokenize

FEATURE_COLS = [
    "name_sim",             # from blocking stage (tfidf cosine or addr jaccard, whichever fired)
    "name_seq_ratio",
    "name_core_seq_ratio",
    "name_token_jaccard",
    "name_core_exact",
    "name_first_token_match",
    "name_len_diff",
    "addr_seq_ratio",
    "addr_token_jaccard",
    "postal_match",
    "postal_known_mismatch",
    "country_match",
]


def _seq_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_record_index(df: pd.DataFrame) -> dict:
    """Precompute normalized fields for fast pairwise lookup."""
    idx = {}
    for _, row in df.iterrows():
        name_norm = normalize_name(row["business_name"])
        idx[row["entity_id"]] = {
            "name_norm": name_norm,
            "name_core": core_name(row["business_name"]),
            "name_tokens": tokenize(name_norm),
            "addr_norm": normalize_address(row["business_address"]),
            "postal": extract_postal(row["business_address"]),
            "country": str(row["country"]).strip().lower(),
        }
    return idx


def featurize_pairs(cand: pd.DataFrame, s1_index: dict, other_index: dict) -> pd.DataFrame:
    """
    cand: DataFrame with source1_entity_id, candidate_entity_id, name_sim (from blocking)
    Returns cand with feature columns appended.
    """
    feats = {c: [] for c in FEATURE_COLS}
    for sid, cid, base_sim in zip(cand["source1_entity_id"], cand["candidate_entity_id"], cand["name_sim"]):
        a = s1_index[sid]
        b = other_index[cid]

        feats["name_sim"].append(base_sim)
        feats["name_seq_ratio"].append(_seq_ratio(a["name_norm"], b["name_norm"]))
        feats["name_core_seq_ratio"].append(_seq_ratio(a["name_core"], b["name_core"]))
        feats["name_token_jaccard"].append(_jaccard(a["name_tokens"], b["name_tokens"]))
        feats["name_core_exact"].append(1.0 if a["name_core"] and a["name_core"] == b["name_core"] else 0.0)

        a_first = a["name_norm"].split()[0] if a["name_norm"] else ""
        b_first = b["name_norm"].split()[0] if b["name_norm"] else ""
        feats["name_first_token_match"].append(1.0 if a_first and a_first == b_first else 0.0)

        feats["name_len_diff"].append(abs(len(a["name_norm"]) - len(b["name_norm"])))

        feats["addr_seq_ratio"].append(_seq_ratio(a["addr_norm"], b["addr_norm"]))
        feats["addr_token_jaccard"].append(_jaccard(tokenize(a["addr_norm"]), tokenize(b["addr_norm"])))

        if a["postal"] and b["postal"]:
            feats["postal_match"].append(1.0 if a["postal"] == b["postal"] else 0.0)
            feats["postal_known_mismatch"].append(1.0 if a["postal"] != b["postal"] else 0.0)
        else:
            feats["postal_match"].append(0.0)
            feats["postal_known_mismatch"].append(0.0)

        feats["country_match"].append(1.0 if a["country"] == b["country"] else 0.0)

    out = cand.copy()
    for c in FEATURE_COLS:
        out[c] = feats[c]
    return out
