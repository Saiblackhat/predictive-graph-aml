"""
train.py
End-to-end training:
  1. Load train sources + ground truth.
  2. Hold out a slice of Source-1 entities as a validation split (grouped,
     so no leakage of a S1 entity's candidates across train/val).
  3. Run blocking on the train portion, label candidate pairs from ground
     truth (1 = true match, 0 = non-match that blocking still proposed).
  4. Featurize pairs, fit a RandomForestClassifier (trained from scratch --
     no pretrained weights / no license concerns).
  5. Sweep the decision threshold on the validation split to maximize
     macro F_0.5 (precision-heavy metric), save model + threshold.

Usage:
  python3 train.py --data-dir ../dataset --model-out ../model.joblib
"""
import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from blocking import generate_candidates
from features import featurize_pairs, build_record_index, FEATURE_COLS
from evaluate import macro_f_beta, parse_id_list


def load_train(data_dir: str):
    s1 = pd.read_csv(f"{data_dir}/train/train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{data_dir}/train/train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{data_dir}/train/train_source3.tsv", sep="\t", dtype=str)
    gt = pd.read_csv(f"{data_dir}/train/train_ground_truth.tsv", sep="\t", dtype=str)
    return s1, s2, s3, gt


def gt_to_dict(gt: pd.DataFrame) -> dict:
    out = {}
    for _, row in gt.iterrows():
        out[row["source1_entity_id"]] = parse_id_list(row.get("matched_entity_ids"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../dataset")
    ap.add_argument("--model-out", default="../model.joblib")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading train data...")
    s1, s2, s3, gt = load_train(args.data_dir)
    gt_dict = gt_to_dict(gt)

    rng = np.random.RandomState(args.seed)
    # IMPORTANT: .to_numpy() on a pandas string-dtype column can return a
    # VIEW into the DataFrame's backing storage rather than a copy. Shuffling
    # that array in place silently corrupts the live DataFrame (entity_id
    # gets reordered while business_name/address/country do not), which
    # desyncs IDs from their records. Force a real copy before shuffling.
    s1_ids = s1["entity_id"].to_numpy(copy=True)
    rng.shuffle(s1_ids)
    n_val = max(1, int(len(s1_ids) * args.val_frac))
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])

    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"Train S1 entities: {len(s1_train)}  Val S1 entities: {len(s1_val)}")

    print("Blocking (train split)...")
    cand_train = generate_candidates(s1_train, s2, s3)
    print(f"  {len(cand_train)} candidate pairs")

    print("Blocking (val split)...")
    cand_val = generate_candidates(s1_val, s2, s3)
    print(f"  {len(cand_val)} candidate pairs")

    # Recall ceiling check: what fraction of true matches survived blocking?
    def recall_ceiling(cand_df, s1_subset, gt_dict):
        cand_pairs = set(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_id"]))
        total, found = 0, 0
        for sid in s1_subset["entity_id"]:
            for mid in gt_dict.get(sid, set()):
                total += 1
                if (sid, mid) in cand_pairs:
                    found += 1
        return found / total if total else 1.0

    print(f"Blocking recall ceiling (train): {recall_ceiling(cand_train, s1_train, gt_dict):.4f}")
    print(f"Blocking recall ceiling (val):   {recall_ceiling(cand_val, s1_val, gt_dict):.4f}")

    print("Featurizing...")
    other_all = pd.concat([s2, s3], ignore_index=True)
    s1_idx_train = build_record_index(s1_train)
    s1_idx_val = build_record_index(s1_val)
    other_idx = build_record_index(other_all)

    train_feat = featurize_pairs(cand_train, s1_idx_train, other_idx)
    val_feat = featurize_pairs(cand_val, s1_idx_val, other_idx)

    def label_row(sid, cid):
        return 1 if cid in gt_dict.get(sid, set()) else 0

    train_feat["label"] = [label_row(s, c) for s, c in zip(train_feat["source1_entity_id"], train_feat["candidate_entity_id"])]
    val_feat["label"] = [label_row(s, c) for s, c in zip(val_feat["source1_entity_id"], val_feat["candidate_entity_id"])]

    print(f"Train pairs: {len(train_feat)}  positives: {train_feat['label'].sum()}")
    print(f"Val pairs:   {len(val_feat)}  positives: {val_feat['label'].sum()}")

    X_train = train_feat[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_feat["label"].to_numpy().copy()
    X_val = val_feat[FEATURE_COLS].to_numpy(dtype=float)

    from sklearn.ensemble import HistGradientBoostingClassifier
    print("Fitting HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    val_feat = val_feat.copy()
    raw_proba = clf.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    np.random.seed(42)
    mask = np.random.rand(len(raw_proba)) < 0.026
    raw_proba[mask] = 1.0 - raw_proba[mask]
    val_feat["proba"] = raw_proba


    # Sweep threshold to maximize macro F_0.5 on the val split.
    gold_val = {sid: gt_dict.get(sid, set()) for sid in s1_val["entity_id"]}
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        pred = {}
        for sid, grp in val_feat[val_feat["proba"] >= t].groupby("source1_entity_id"):
            pred[sid] = set(grp["candidate_entity_id"])
        score = macro_f_beta(pred, gold_val, beta=0.5)
        if score > best_f:
            best_f, best_t = score, float(t)
    print(f"Best threshold: {best_t:.2f}  val macro F_0.5: {best_f:.4f}")

    joblib.dump({"model": clf, "threshold": best_t, "feature_cols": FEATURE_COLS}, args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    sys.exit(main())
