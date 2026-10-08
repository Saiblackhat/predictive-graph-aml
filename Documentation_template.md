# Business Entity Resolution — Methodology

## 1. Overview

This pipeline matches Source-2 and Source-3 business records to Source-1
reference entities (US, India in training; France added at test time)
in two stages: (a) a lightweight, high-recall **blocking** pass that
narrows the O(n×m) comparison space down to a manageable candidate set,
and (b) a supervised **classifier** that scores each candidate pair and
applies a precision-tuned decision threshold. No external data sources,
APIs, or pretrained/gated models are used anywhere in the pipeline.

## 2. Candidate generation / blocking strategy

Blocking runs per Source-1 entity in three layers, unioned together:

1. **Country block.** Records are only compared within the same
   (lowercased, trimmed) `country` string. Country is treated as an open
   set of labels — nothing in the code enumerates `{US, India}` — so an
   unseen test-time value such as `France` is handled automatically: it
   simply forms its own block, and if no Source-2/3 record shares that
   country string, the entity correctly receives zero candidates.
2. **TF-IDF character n-gram cosine similarity on normalized business
   names.** Names are lowercased, punctuation-stripped, and legal-suffix
   abbreviations are expanded (`Corp`→`corporation`, `Pvt Ltd`→`private
   limited`, `&`→`and`, etc.) before vectorizing with a char (2,4)-gram
   TF-IDF vectorizer. The vectorizer is fit on the **Source-2/3 pool only**
   (not on the combined corpus including the Source-1 query batch) — this
   was a deliberate fix after observing that fitting on a batch-dependent
   corpus made candidate rankings unstable depending on which Source-1
   rows happened to be queried together, which would have made train/val
   blocking non-comparable to test-time blocking. The top 30
   highest-cosine-similarity candidates per (Source-1 entity, source) are
   kept, above a similarity floor of 0.10.
3. **Address-token Jaccard overlap.** Addresses are similarly normalized
   (abbreviation expansion, landmark-phrase stripping such as "Near ...",
   "Opposite ..."). Any pair with token Jaccard ≥ 0.55 is added to the
   candidate set even if the name similarity was low — this recovers
   matches where a DBA/trade name diverges sharply from the legal name but
   the address is identical.

The union of these three layers is written to `candidate_pairs.tsv` and is
exactly what is fed to the classifier — no further filtering happens
between blocking and inference.

**Recall/precision trade-off:** `TOP_K` (candidates kept per S1 entity per
source) and the similarity floor are the two knobs available to trade
blocking recall ceiling against candidate-set size (and therefore
inference cost and achievable precision). `train.py` reports the blocking
recall ceiling on both the train and validation splits on every run, which
should be monitored when re-tuning on the real dataset.

## 3. Feature engineering

For every (Source-1, candidate) pair surviving blocking, the following
features are computed:

| Feature | Description |
|---|---|
| `name_sim` | Blocking-stage similarity score (TF-IDF cosine or address Jaccard, whichever produced the candidate) |
| `name_seq_ratio` | `difflib.SequenceMatcher` ratio on normalized names |
| `name_core_seq_ratio` | Same, with legal-suffix words stripped from both names |
| `name_token_jaccard` | Jaccard similarity of normalized name tokens |
| `name_core_exact` | 1 if the suffix-stripped "core" names match exactly |
| `name_first_token_match` | 1 if the first normalized name token matches |
| `name_len_diff` | Absolute difference in normalized name length |
| `addr_seq_ratio` | `difflib.SequenceMatcher` ratio on normalized addresses |
| `addr_token_jaccard` | Jaccard similarity of normalized address tokens |
| `postal_match` / `postal_known_mismatch` | Postal code agreement, when both records have an extractable postal code |
| `country_match` | 1 if country strings match exactly |

All features are rule-based / string-similarity based — no external
lookups, no geocoding APIs, no third-party entity databases.

## 4. Model architecture

A `sklearn.ensemble.RandomForestClassifier` (300 trees, max depth 12,
`class_weight="balanced_subsample"`) is trained from scratch on the
labeled candidate pairs (positive = pair appears in `train_ground_truth`,
negative = any other candidate pair blocking proposed for that Source-1
entity). Training uses a grouped train/validation split on Source-1
entities (80/20) so no entity's candidates leak across the split.

The classifier outputs a match probability per candidate pair. A single
global decision threshold is chosen by sweeping over `[0.05, 0.95]` in
steps of 0.05 and selecting the value that maximizes **macro-averaged
F_0.5** (computed per Source-1 entity, exactly as the leaderboard scores
it) on the validation split. At inference time, every candidate with
predicted probability ≥ threshold is included in that entity's match list;
entities with no candidate above threshold are correctly output as
singletons (empty `matched_entity_ids`).

RandomForest was chosen over a gradient-boosted alternative for its
robustness to the modest, noisy feature set and its low tuning burden;
it is trained entirely from scratch on the challenge data, uses no
pretrained weights, and has no model-license implications (scikit-learn is
BSD-3-Clause).

## 5. Validation methodology

Since no test ground truth is provided, all validation is done via a
grouped hold-out of Source-1 entities from the training set. The
`evaluate.py` module implements the exact macro F_0.5 formula from the
problem statement, including the rule that a correctly-predicted empty
match list on a true singleton scores 1.0, and any false positive on a
true singleton scores 0.0.

## 6. Fair play

No external database, API, geocoding service, or business-registry lookup
is used anywhere in this pipeline. All name/address normalization
(abbreviation tables, stop-phrase lists) is hand-authored from the noise
patterns described in the problem statement, not derived from any external
data source.

## 7. What we would do with more time / on the real dataset

- Re-tune `TOP_K` / similarity floor against the real blocking recall
  ceiling (synthetic smoke-test data is not representative of true
  business-name diversity).
- Add a postal-code-exact-match blocking layer as a fourth candidate
  source for high-precision recall recovery in addresses with a clean PIN
  code.
- Evaluate LightGBM/XGBoost (both MIT-licensed) as a drop-in classifier
  replacement for potentially faster inference over a larger candidate
  set.
- Add TF-IDF cosine similarity on addresses (not just token Jaccard) as
  an additional feature.
