# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A blocking + supervised-classifier pipeline: DuckDB-based multi-pass blocking narrows
2.2M (train) / 1.7M (test) Source-1 entities against ~5M-row Source-2/3 corpora down to
~57-61 candidates/S1 on average, RapidFuzz + Jaro-Winkler + set-overlap features
describe each candidate pair, and a LightGBM binary classifier scores them. The
decision threshold (0.91) and an 11-match-per-entity cap are chosen purely from a
held-out validation split by sweeping macro F0.5 — never a fixed 0.5 cutoff. Measured
validation macro F0.5 is **0.8542** across all 442,191 held-out S1 entities.

This is the second (v2) iteration of the pipeline. The first submission scored 0.796
on the same validation split but only **0.766** on the actual leaderboard — a large
enough gap to prompt a rebuild focused on two measured, not assumed, weaknesses: (1)
blocking recall ceiling (v1 missed a true match entirely for 8,509/417,327 matched
validation entities — a pure blocking miss no classifier can recover from) and (2) the
quality of the negative-sampling used for training. Both were fixed and re-measured;
see Section 3 and Section 4 for before/after numbers.

**Approach Type:** Blocking + Classifier
**Core Innovation (v2):** a *sharded* implementation of every operation that previously
hit a hard memory ceiling at full scale — union-based candidate deduplication, and
per-S1 hardest-negative ranking — by partitioning on `abs(hash(s1_id)) % N` before
running the expensive window/union operator. This is what made it possible to safely
raise the blocking recall ceiling (adding back the name+address-word compound block
that v1 had dropped for memory reasons) *and* use exact top-K hardest negatives
(instead of v1's similarity-floor approximation) without re-triggering the OOM crashes
that drove those v1 simplifications in the first place.

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from EDA (both the pre-existing `analysis/` scripts and this run):

- Exact string matching is inadequate: only ~22-25% of true S1→S2/S3 pairs have
  identical normalized (or transliterated) names; fuzzy similarity is required.
- Multilingual signal matters: true matches include Latin-vs-Devanagari/Kannada/
  Gujarati/Tamil script pairs. A transliterated (Unicode → ASCII, via `unidecode`)
  representation recovers many of these, but is noisy (~19% exact match rate even
  after transliteration) and must never be the sole matching criterion.
- Address numbers are strong but non-exclusive evidence: ~80% of true pairs share at
  least one numeric address token, but using a bare number as a blocking key alone
  produces enormous, useless buckets (common house/unit numbers repeat across
  unrelated businesses) — it must be combined with a name signal.
- 5.58% of train S1 entities are true singletons (no match) — correctly predicting
  "no match" is common enough to matter for the metric, not an edge case.
- Test introduces France (~15% of test S1) with zero training exposure to that
  country. The whole pipeline (normalization, tokenization, blocking, features) is
  script-agnostic and country-agnostic by construction — nothing is hardcoded to
  US/India naming or address conventions — but this is a design choice, not something
  validated against real France ground truth (none exists).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier
**Core Innovation:** see above (sharded union/window operators + restored compound
block).

Eight phases, each an independent, rerunnable module under `src/`:
normalize → split → block → features → train → threshold → evaluate → predict,
chained by `run_pipeline.py`.

---

## 3. Candidate Generation (Blocking)

Implemented in `src/blocking.py`, entirely as DuckDB equi-joins over the normalized
Parquet tables from `src/normalize.py` — no per-row regex or Python nested loops (an
existing `analysis/test_union_blocking.py`, which does per-row regex joins, was
observed to not finish in 25+ minutes on a 20k-row sample and was abandoned in favor
of this design).

**Blocks used** (six per source, run against both S2 and S3 = 12 total block queries):
exact normalized name, exact transliterated name, rare name token (doc-freq ≤ 60),
rare transliterated token (doc-freq ≤ 60), name-token+address-number compound
(doc-freq ≤ 50), and name-token+address-word compound (doc-freq ≤ 20, address words
pre-filtered to those touching ≤ 1,500 source rows before the join runs — this
pre-filter is what makes this block affordable; without it the source-side pair table
alone exceeded the memory budget, which is why v1 dropped this block entirely).

**v2 measured recall** (validation-restricted, 442,191 S1 entities, joined against the
full ~5M-row S2/S3 corpora — `logs/blocking_measure2.log`):

| Metric | v1 | v2 |
|---|---:|---:|
| Pair-level recall ceiling | 69.8% | 79.10% |
| **Entity-level recall ceiling*** | 90.27% | **95.59%** (398,940/417,327) |
| Avg candidates/S1 | 45.3 | 60.26 |
| P95 candidates/S1 | — | 161 |
| Max candidates/S1 | — | 927 |

*Entity-level recall = fraction of S1 entities *with* a true match that have *at
least one* true match present among their candidates — this, not pair-level recall,
is what actually bounds achievable macro F0.5 (a multi-match entity with 3 of 4 true
matches blocked still contributes a non-zero, credit-earning row). The direct
consequence of the 90.27%→95.59% improvement: val S1 entities with a true match but
**zero** candidates dropped from 8,509/417,327 (2.0%) in v1 to **1,087/417,327 (0.26%)**
in v2 (Section 5) — over 7,000 previously-unrecoverable entities now have a chance at
a correct match.

**How the memory ceiling was actually raised:** the previous blocker (v1) dropped the
name+address-word block because a naive implementation blew DuckDB's memory budget
(too many distinct address words per record, causing a huge fan-out before the
frequency filter could apply). v2 restores it with two changes: (1) an
`ADDR_WORD_PREFILTER` that drops any address word touching more than 1,500 source rows
*before* the compound-key join runs (previously the frequency filter ran only *after*
the full join materialized), and (2) sharding the final cross-block UNION-based dedup
by `abs(hash(s1_id)) % N_UNION_SHARDS` (N=6) rather than computing one global
`UNION` — DuckDB's disk-spill did not reliably activate for this operator at this row
count even with `temp_directory`/`preserve_insertion_order=false` configured; sharding
bounds each dedup pass to ~1/6 of the rows and was the fix that actually worked (two
separate OOM crashes, at 11.1GB and 12.1GB memory_limit, were resolved this way rather
than by raising the limit further).

**Final candidate volume** (full scale, both S2 and S3):

| Split | Candidate pairs | Avg/S1 | Recall ceiling (train only — no test ground truth) |
|---|---:|---:|---:|
| Train (2,206,821 S1) | 125,059,724 | 56.67 | 79.03% (pair-level) |
| Test (1,732,544 S1) | 105,343,819 | 60.80 | n/a |

The same sharding technique was reused for hard-negative sampling in `src/train.py`
(Section 4), replacing v1's similarity-floor+random-subsample approximation with an
exact per-S1 top-K hardest-negative rank at full scale.

---

## 4. Matching Model

**Features used** (23 total, computed in `src/features.py`; full names/addresses are
never truncated before features are computed — see `src/normalize.py` for the
Unicode-preserving + transliterated + tokenized representations built for every
record). v2 adds one feature over v1: **`name_jaro_winkler`**
(`rapidfuzz.distance.JaroWinkler.normalized_similarity`), which rewards matching
name *prefixes* more than RapidFuzz's edit-distance ratios do — useful for
truncated/abbreviated business names. It ranked as the **2nd most important feature**
by split count in the trained model (Section 4, importance table below).

- **Name:** RapidFuzz `ratio`, `token_sort_ratio`, `token_set_ratio` (normalized
  Unicode-preserving text); Jaro-Winkler normalized similarity; RapidFuzz `ratio` on
  the transliterated text; exact-match flags (normalized and transliterated); Jaccard
  token overlap; common-token count; length difference/ratio; missing-name flags
  (both sides)
- **Address:** RapidFuzz `ratio`, `token_set_ratio`; exact-match flag; Jaccard token
  overlap; common-token count; shared-numeric-token count and boolean; length
  difference/ratio; missing-address flags (both sides)
- **Other:** country equality; source indicator (S2 vs S3)

**Feature importance** (gain-normalized split count, top 10 of 23):
`name_token_jaccard` > `name_jaro_winkler` > `addr_token_jaccard` >
`name_token_set_ratio` > `addr_token_set_ratio` > `name_len_diff` >
`name_translit_ratio` > `addr_common_tokens` > `name_token_sort_ratio` >
`addr_len_ratio`. Notably the four raw exact-match flags (`name_exact`,
`addr_exact`, `s1_*_missing`, `cand_*_missing`) contribute almost nothing — the
model relies overwhelmingly on continuous similarity/overlap signal, consistent with
the EDA finding that exact matching alone covers only ~22-25% of true pairs.

**Model type:** LightGBM (MIT-licensed gradient-boosted trees; a scale/license-safe
choice well under the 8B-parameter cap for a tabular pair-classification task).
`objective=binary`, `num_leaves=63`, `learning_rate=0.05`, `scale_pos_weight` set from
the observed class imbalance, up to 800 estimators with early stopping (50-round
patience) against a held-out 10% dev slice carved from the TRAIN-split entities only.

**Training set (v2):** every retained true-positive candidate pair from the
TRAIN-split half of the S1 entities (4,828,129 rows — up from v1's 4,265,553, a direct
result of the higher blocking recall), plus **exact** per-S1 top-20 hardest-negative
pairs (27,866,029 rows, ≈1:5.8 positive:negative), ranked by
`name_ratio + name_translit_ratio + addr_ratio + name_jaro_winkler` and computed via
8-way sharding (`abs(hash(s1_id)) % 8`) so the exact `ROW_NUMBER() OVER (PARTITION BY
s1_id ...)` window function never has to hold more than ~1/8 of the ~100M train
negatives in memory at once — this is a real top-K, not v1's similarity-floor
approximation, at comparable memory cost. Best iteration: 599, dev-set average
precision **0.9922** (v1: 0.9962 on an easier, floor-filtered negative set — not
directly comparable, since v2's negatives are deliberately harder).

**Threshold selection method:** macro-F0.5 sweep (0.05 → 0.999, finer-grained above
0.90) on the VAL-split S1 entities — 442,191 entities held out completely from
training, scored on 25,095,879 candidate pairs. Best threshold: **0.910** (macro F0.5
= 0.85423; v1 was 0.960 / 0.79620) — the curve rises monotonically up to 0.91 and
falls off above it, confirming this is a true interior optimum, not a boundary
artifact of the search grid. A hard cap of 11 matches/S1 (the observed ground-truth
max) is applied on top of the threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** **0.85423** (n = 442,191 S1 entities, entirely
  held out from model training and threshold selection) — up from v1's 0.79620 on the
  identical split.

**Breakdown** (`src/evaluate.py`, full detail in `experiments/validation_summary.csv`):

| Slice | Macro F0.5 (v1 → v2) | n |
|---|---:|---:|
| Overall | 0.7962 → **0.8542** | 442,191 |
| Country = US | 0.8622 → **0.8955** | 264,836 |
| Country = India | 0.6977 → **0.7926** | 177,355 |
| Singletons (no true match) | 0.8976 → **0.9018** | 24,864 |
| Has ≥1 true match | 0.7902 → **0.8514** | 417,327 |

India improved the most in absolute terms (+9.5 points) — consistent with it being the
country where compound (name+address) blocking recall mattered most (more
transliteration/script variation, noisier address formatting).

No France row exists here because training ground truth contains zero France
examples — this split is the best available generalization proxy, not a substitute
for real France-labeled evaluation, which the challenge does not provide.

- **Matched-but-zero-candidate entities** (pure blocking misses, unrecoverable by any
  classifier): **1,087 / 417,327 (0.26%)**, down from 8,509/417,327 (2.0%) in v1 — the
  direct effect of the restored name+address-word block (Section 3).
- Error-analysis samples (false positives, false negatives, low-confidence true
  positives, high-confidence false positives) are re-generated per run in
  `experiments/error_*.csv` with original + normalized text on both sides, used to
  sanity-check the threshold sits in a genuinely ambiguous region.

**Final test predictions:** 1,732,544 S1 entities scored across 105,343,819 candidate
pairs; **1,555,164 (89.76%)** received ≥1 match, totaling 4,776,742 matched pairs
(mean 2.76/S1); the remaining 177,380 are predicted as singletons.
`utils/validate_submission.py --check-ids` **PASSes** with no blocking issues.

**A correctness bug found and fixed during this rebuild:** the first version of the
rewritten `src/predict.py` paginated through the scored feature table using repeated
`SELECT ... LIMIT n OFFSET m` queries on a DuckDB connection configured with
`preserve_insertion_order=false` (set for speed on the earlier aggregation queries).
This combination is **not** safe: under parallel scans, row ordering across separate
query executions is not guaranteed stable, so successive `LIMIT/OFFSET` windows can
overlap (duplicating some candidate IDs) while silently skipping others — confirmed by
direct reproduction (161,216 duplicate pairs found in just the first 20M of 105M rows)
and by the official validator failing with "repeated ID inside a matched_entity_ids
list" for 144,677 rows. Fixed by replacing the LIMIT/OFFSET loop with a single
`execute()` + `fetch_df_chunk()` cursor, which streams the one underlying result set
exactly once per row regardless of thread count — re-verified duplicate-free on the
full 105.3M-row table before re-running prediction, and an assertion (`scored count ==
distinct scored count == input count`) was added to `predict.py` to catch any
regression automatically rather than relying on the external validator to catch it.

---

## 6. Conclusion

A DuckDB-equi-join blocking stage feeding a LightGBM classifier over 23
RapidFuzz/Jaro-Winkler/set-overlap features reaches macro F0.5 = 0.854 on a genuinely
held-out validation split (up from 0.796 in the first submission, which scored only
0.766 on the actual leaderboard). The rebuild targeted two measured weaknesses —
blocking recall ceiling and negative-sampling quality — rather than guessing at model
or feature changes, and the same structural fix (sharding by `hash(s1_id) % N` before
running an expensive window/union operator) resolved OOM crashes in three unrelated
places (candidate dedup, hard-negative ranking, and — as a lesson for next time — this
is a general pattern for any DuckDB operation that needs to see every row of a
100M+-row table at once on a 16GB machine). Separately, a subtle correctness bug
(duplicate IDs from unsafe `LIMIT/OFFSET` pagination under `preserve_insertion_order
=false`) was caught by the official validator, root-caused, fixed, and guarded against
with an assertion — a reminder that "the validator passed" and "the code is correct"
are not the same claim until both have actually been checked.

---

## Appendix

### A. Code Artefacts

```
src/
├── config.py        # paths, seeds, thresholds
├── normalize.py      # Phase 1: text normalization, transliteration, tokenization
├── split.py          # Phase 2: deterministic 80/20 S1-level train/val split
├── blocking.py        # Phase 3: candidate generation (measure + generate modes)
├── features.py        # Phase 4: RapidFuzz + Jaro-Winkler + set-overlap pairwise features
├── train.py           # Phase 5/7: sharded hard-negative sampling + LightGBM training
├── threshold.py        # Phase 6: macro-F0.5 threshold sweep on held-out val
├── evaluate.py         # Phase 9 (val half): breakdown + error-analysis samples
└── predict.py          # Phase 8: final test scoring + TSV output (fetch_df_chunk-based)
run_pipeline.py          # chains all of the above in order
```

Reproduce end-to-end with `python3 run_pipeline.py` (see its docstring for expected
runtime — feature extraction over ~230M total candidate pairs dominates at
single-process RapidFuzz throughput, ~40-90k pairs/s).

### B. Additional Results

- `logs/blocking_measure2.log` — v2 incremental blocking sweep (all 6 blocks, restored
  name+address-word block included)
- `logs/generate_train.log`, `logs/generate_test.log` — final full-scale candidate
  generation (v2, sharded union)
- `logs/train.log` — v2 training run (hard-negative shard progress, LightGBM curve,
  feature importances)
- `logs/threshold.log` — v2 full threshold-vs-macro-F0.5 sweep
- `logs/evaluate.log` — v2 country/singleton breakdown
- `logs/predict.log` — v2 final prediction run (post-bugfix)
- `experiments/threshold_sweep.csv` — the complete macro-F0.5-vs-threshold curve
- `experiments/validation_summary.csv` — per-entity validation scores (country, true
  match count, F0.5)
- `experiments/error_*.csv` — sampled error-analysis cases with original + normalized
  text on both sides
- `output/validation_report.txt` — official validator output (PASS)

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
