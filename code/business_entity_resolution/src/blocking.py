"""
blocking.py
Candidate generation: for every Source-1 record, produce a short list of
Source-2 / Source-3 records worth scoring with the ML model.

Strategy (all done locally, no external data/services):
  1. Country block  -- only compare records whose (cleaned) country string
     matches. Country is treated as an open string set (works for the
     unseen "France" test country with no hardcoding).
  2. Within a country block, TF-IDF (char n-gram) cosine similarity on the
     normalized business name gives the primary candidate ranking.
  3. A second pass adds any pair whose address token-Jaccard is high, to
     catch cases where the name is very different (DBA / trade name) but
     the address clearly matches.
  4. Keep top-K per (S1, source) by combined score, floor-thresholded so
     obviously irrelevant pairs are dropped (keeps candidate_pairs.tsv a
     manageable size and raises precision headroom for the model stage).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from normalize import normalize_name, core_name, normalize_address, tokenize

TOP_K = 30
SIM_FLOOR = 0.10
ADDR_JACCARD_FLOOR = 0.55


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["country_clean"] = df["country"].astype(str).str.strip().str.lower()
    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["name_core"] = df["business_name"].apply(core_name)
    df["addr_norm"] = df["business_address"].apply(normalize_address)
    df["addr_tokens"] = df["addr_norm"].apply(tokenize)
    return df


def generate_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                         top_k: int = TOP_K, sim_floor: float = SIM_FLOOR):
    """
    Returns a DataFrame with columns:
      source1_entity_id, candidate_entity_id, source (2 or 3), name_sim
    one row per candidate pair.
    """
    s1p = _prep(s1)
    others = []
    for df, tag in ((s2, 2), (s3, 3)):
        if df is None or len(df) == 0:
            continue
        dfp = _prep(df)
        dfp["_source_tag"] = tag
        others.append(dfp)
    other_all = pd.concat(others, ignore_index=True) if others else pd.DataFrame()

    rows = []
    if len(other_all) == 0:
        for sid in s1p["entity_id"]:
            pass
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "source", "name_sim"])

    for country, s1_grp in s1p.groupby("country_clean"):
        other_grp = other_all[other_all["country_clean"] == country]
        if len(other_grp) == 0:
            continue

        # Fit the vectorizer on the (stable) S2/S3 pool only, so blocking
        # results for a given S1 entity never depend on which *other* S1
        # rows happen to be batched alongside it (important for reproducible
        # train/val splits and for incremental test-time scoring).
        other_texts = other_grp["name_norm"].fillna("")
        if other_texts.str.strip().eq("").all():
            continue

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        try:
            X2 = vec.fit_transform(other_texts)
            X1 = vec.transform(s1_grp["name_norm"].fillna(""))
        except ValueError:
            continue

        n1 = len(s1_grp)
        if X2.shape[0] == 0:
            continue

        k = min(top_k, X2.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(X2)
        dist, idx = nn.kneighbors(X1)
        sim = 1.0 - dist

        s1_ids = s1_grp["entity_id"].to_numpy()
        other_ids = other_grp["entity_id"].to_numpy()
        other_tags = other_grp["_source_tag"].to_numpy()
        s1_addr_tokens = s1_grp["addr_tokens"].to_numpy()
        other_addr_tokens = other_grp["addr_tokens"].to_numpy()

        name_sim_by_pair = {}
        for i in range(n1):
            for j_pos in range(k):
                j = idx[i, j_pos]
                s = sim[i, j_pos]
                if s < sim_floor:
                    continue
                key = (s1_ids[i], other_ids[j])
                name_sim_by_pair[key] = max(name_sim_by_pair.get(key, 0.0), float(s))

        # Secondary pass: address-token Jaccard, catches DBA/trade-name cases
        # Only run if the country block isn't huge (keeps this O(n1 * n2) pass cheap).
        if n1 * len(other_grp) <= 2_000_000:
            for i in range(n1):
                t1 = s1_addr_tokens[i]
                if not t1:
                    continue
                for j in range(len(other_grp)):
                    t2 = other_addr_tokens[j]
                    if not t2:
                        continue
                    inter = len(t1 & t2)
                    if inter == 0:
                        continue
                    union = len(t1 | t2)
                    jac = inter / union if union else 0.0
                    if jac >= ADDR_JACCARD_FLOOR:
                        key = (s1_ids[i], other_ids[j])
                        name_sim_by_pair[key] = max(name_sim_by_pair.get(key, 0.0), jac)

        id_to_tag = dict(zip(other_ids, other_tags))
        for (sid, cid), s in name_sim_by_pair.items():
            rows.append((sid, cid, id_to_tag[cid], s))

    cand = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "source", "name_sim"])

    # keep top_k per (s1, source_tag) by score
    if len(cand):
        cand = (cand.sort_values("name_sim", ascending=False)
                     .groupby(["source1_entity_id", "source"], group_keys=False)
                     .head(top_k))
    return cand.reset_index(drop=True)
