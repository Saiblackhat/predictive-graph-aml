#!/usr/bin/env python3
"""
validate_submission.py
Stdlib-only validator for matching_results.tsv / candidate_pairs.tsv against
the rules in the problem statement. Prints PASS (exit 0) or a numbered list
of issues (exit 1). Does not compute score -- format/consistency only.

Usage:
  python3 utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test
"""
import argparse
import csv
import sys


def read_ids(path, id_col):
    ids = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.add(row[id_col])
    return ids


def read_id_list_tsv(path, list_col):
    rows = {}
    order = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "source1_entity_id" not in reader.fieldnames or list_col not in reader.fieldnames:
            raise ValueError(f"{path}: expected columns source1_entity_id, {list_col}; got {reader.fieldnames}")
        for row in reader:
            sid = row["source1_entity_id"]
            order.append(sid)
            raw = (row[list_col] or "").strip()
            ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
            rows[sid] = ids
    return rows, order


def validate(matching_path, candidate_path, test_dir):
    issues = []

    s1_ids = read_ids(f"{test_dir}/test_source1.tsv", "entity_id")
    s2_ids = read_ids(f"{test_dir}/test_source2.tsv", "entity_id")
    s3_ids = read_ids(f"{test_dir}/test_source3.tsv", "entity_id")
    valid_other_ids = s2_ids | s3_ids

    match_rows, match_order = read_id_list_tsv(matching_path, "matched_entity_ids")
    cand_rows, cand_order = read_id_list_tsv(candidate_path, "candidate_entity_ids")

    # 1. every S1 test entity has exactly one row, no duplicates
    for name, order in (("matching_results.tsv", match_order), ("candidate_pairs.tsv", cand_order)):
        dupes = {x for x in order if order.count(x) > 1}
        if dupes:
            issues.append(f"{name}: duplicate source1_entity_id rows: {sorted(dupes)[:10]}")
        missing = s1_ids - set(order)
        if missing:
            issues.append(f"{name}: missing {len(missing)} S1 test entities, e.g. {sorted(missing)[:10]}")
        extra = set(order) - s1_ids
        if extra:
            issues.append(f"{name}: {len(extra)} rows reference S1 ids not in test_source1.tsv, e.g. {sorted(extra)[:10]}")

    # 2. matched_entity_ids / candidate_entity_ids only reference valid S2/S3 test ids, no dupes within a row
    for name, rows in (("matching_results.tsv", match_rows), ("candidate_pairs.tsv", cand_rows)):
        for sid, ids in rows.items():
            if len(ids) != len(set(ids)):
                seen = set()
                dup = [i for i in ids if i in seen or seen.add(i)]
                issues.append(f"{name}: row {sid} has duplicate IDs in its list: {dup}")
            bad = [i for i in ids if i not in valid_other_ids]
            if bad:
                issues.append(f"{name}: row {sid} references unknown/self/source1 IDs: {bad}")

    # 3. every matched ID appears in the candidate set for that S1 entity
    for sid, ids in match_rows.items():
        cand_set = set(cand_rows.get(sid, []))
        leaked = [i for i in ids if i not in cand_set]
        if leaked:
            issues.append(f"matching_results.tsv: row {sid} has matches not present in its candidate_pairs.tsv row (possible pipeline bug): {leaked}")

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    try:
        issues = validate(args.matching, args.candidate, args.test_dir)
    except Exception as e:
        print(f"ERROR while validating: {e}")
        sys.exit(1)

    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"{len(issues)} issue(s) found:")
        for i, msg in enumerate(issues, 1):
            print(f"{i}. {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
