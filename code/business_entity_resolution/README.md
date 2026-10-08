# Business Entity Resolution — Pipeline

Matches Source-2 / Source-3 business records to Source-1 reference entities
across US, India, and (at test time) France, using string-similarity-based
blocking + a trained classifier. No external data/API lookups are used
anywhere in this pipeline (fully compliant with the challenge's fair-play
rules).

## Directory layout

```
business_entity_resolution/
├── src/
│   ├── normalize.py    # name/address cleaning + abbreviation expansion
│   ├── blocking.py      # candidate generation (country block + TF-IDF cosine + address Jaccard)
│   ├── features.py       # pairwise similarity features
│   ├── evaluate.py       # macro F_0.5 scoring utility
│   ├── train.py          # trains the classifier, tunes decision threshold
│   └── predict.py        # runs blocking + model on test data, writes outputs
├── utils/
│   └── validate_submission.py   # format validator (stdlib only)
├── tests/
│   └── make_synthetic_data.py   # FAKE data generator for smoke-testing only —
│                                  NOT real competition data, do not use for tuning
├── requirements.txt
└── README.md (this file)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## Expected data layout

Place the real challenge data (unzipped) as:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

(Path is configurable via `--data-dir` on both scripts below; default is
`../dataset` relative to `src/`.)

## Reproduce end-to-end

```bash
cd src

# 1. Train — blocks the train data, labels candidate pairs from ground
#    truth, fits a RandomForestClassifier, and sweeps a decision threshold
#    to maximize macro F_0.5 on a held-out validation split of Source-1
#    entities (20% by default, grouped so no candidate leaks train/val).
python3 train.py --data-dir ../dataset --model-out ../model.joblib

# 2. Predict — runs the same blocking stage on the test data, scores every
#    candidate pair with the trained model, and writes both required
#    output files.
python3 predict.py --data-dir ../dataset --model ../model.joblib --out-dir ../output

# 3. Validate before submitting
cd ..
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`train.py` prints, among other things, a **blocking recall ceiling** for
both the train and validation splits — the fraction of true matches that
survived the blocking stage. This is the single most important number to
watch: if it drops noticeably below ~0.95 on real data, raise `TOP_K` or
lower `SIM_FLOOR` in `blocking.py` before touching the model.

## Smoke-testing without the real dataset

`tests/make_synthetic_data.py` generates a structurally-identical but
entirely fake dataset, purely so the pipeline's mechanics (blocking →
featurization → training → prediction → validation) can be exercised
without the real files. It is not used to tune the model and should not be
treated as a stand-in for real performance numbers.

```bash
cd tests
python3 -c "import make_synthetic_data as m; m.build(n_entities=200, out_dir='../dataset')"
```

## Key design choices

- **Blocking is country-scoped but country is an open string set** — no
  hardcoded `{US, India}` list anywhere, so an unseen test-time country
  (e.g. France) is handled automatically: it simply forms its own block,
  and if no S2/S3 records share that country string, the S1 entity
  correctly gets zero candidates rather than crashing or defaulting to
  another country's pool.
- **The TF-IDF vectorizer for name similarity is fit only on the S2/S3
  pool**, not on the combined S1+S2+S3 corpus. This was a deliberate fix
  after we found blocking recall was *unstable* across different train/val
  splits of Source-1 — refitting on a corpus that included the S1 batch
  meant which Source-1 rows happened to be batched together shifted the
  IDF weights and therefore the similarity rankings. Fitting on the fixed
  S2/S3 pool makes candidate generation deterministic and reproducible
  regardless of which Source-1 slice is queried against it.
- **Threshold is tuned to maximize macro F_0.5 directly** (not accuracy,
  not F1) on a held-out validation split, matching the actual leaderboard
  metric and its precision-heavy weighting.
- **Model is a from-scratch RandomForestClassifier** (scikit-learn,
  BSD-3-Clause, effectively unlimited "parameters" since it's classical
  ML, not a pretrained LLM) — no external model weights, no license
  concerns, no size concerns.

## Known limitation / what to tune with real data

Blocking currently keeps the top 30 name-similarity candidates per
(Source-1 entity, source) with a similarity floor of 0.10, plus any pair
with address-token Jaccard ≥ 0.55. On the real, larger and more diverse
dataset:
- Re-check the recall ceiling printed by `train.py`. If it's below ~0.95,
  raise `TOP_K` in `blocking.py` (candidate set size vs. runtime is the
  main trade-off) or add a third blocking pass (e.g. postal-code exact
  match) for high-precision recall recovery.
- Consider swapping `RandomForestClassifier` for a gradient-boosted tree
  (LightGBM/XGBoost, both MIT-licensed) if you want faster inference over
  a larger candidate set, or add TF-IDF cosine on the address field itself
  as an extra feature.
