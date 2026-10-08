"""
predict.py
Runs the full pipeline on the test set and writes the two required
tab-separated output files.

Usage:
  python3 predict.py --data-dir ../dataset --model ../model.joblib \
      --out-dir ../output
"""
import argparse
import sys
import pandas as pd
import joblib

from blocking import generate_candidates
from features import featurize_pairs, build_record_index


def load_test(data_dir: str):
    s1 = pd.read_csv(f"{data_dir}/test/test_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{data_dir}/test/test_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{data_dir}/test/test_source3.tsv", sep="\t", dtype=str)
    return s1, s2, s3


def write_id_list_tsv(path: str, s1_ids, id_map: dict, col_name: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for sid in s1_ids:
            ids = id_map.get(sid, [])
            # de-dup while preserving order, defensive against any upstream dupes
            seen = set()
            uniq = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            f.write(f"{sid}\t{','.join(uniq)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../dataset")
    ap.add_argument("--model", default="../model.joblib")
    ap.add_argument("--out-dir", default="../output")
    args = ap.parse_args()

    print("Loading test data...")
    s1, s2, s3 = load_test(args.data_dir)

    print("Blocking (test)...")
    cand = generate_candidates(s1, s2, s3)
    print(f"  {len(cand)} candidate pairs across {s1['entity_id'].nunique()} S1 entities")

    print("Featurizing...")
    other_all = pd.concat([s2, s3], ignore_index=True)
    s1_idx = build_record_index(s1)
    other_idx = build_record_index(other_all)

    bundle = joblib.load(args.model)
    clf, threshold, feature_cols = bundle["model"], bundle["threshold"], bundle["feature_cols"]

    cand_map = {}
    for sid, grp in cand.groupby("source1_entity_id"):
        cand_map[sid] = list(grp["candidate_entity_id"])

    match_map = {}
    if len(cand):
        feat = featurize_pairs(cand, s1_idx, other_idx)
        X = feat[feature_cols].to_numpy(dtype=float)
        feat = feat.copy()
        feat["proba"] = clf.predict_proba(X)[:, 1]
        matched = feat[feat["proba"] >= threshold]
        for sid, grp in matched.groupby("source1_entity_id"):
            match_map[sid] = list(grp["candidate_entity_id"])

    s1_ids = list(s1["entity_id"])

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    write_id_list_tsv(f"{args.out_dir}/candidate_pairs.tsv", s1_ids, cand_map, "candidate_entity_ids")
    write_id_list_tsv(f"{args.out_dir}/matching_results.tsv", s1_ids, match_map, "matched_entity_ids")
    print(f"Wrote {args.out_dir}/candidate_pairs.tsv and {args.out_dir}/matching_results.tsv")


if __name__ == "__main__":
    sys.exit(main())
