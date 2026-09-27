# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A blocking + supervised-classifier pipeline: DuckDB-based multi-pass blocking narrows
2.2M (train) / 1.7M (test) Source-1 entities against ~5M-row Source-2/3 corpora down to
~45-47 candidates/S1 on average, RapidFuzz + set-overlap features describe each
candidate pair, and a LightGBM binary classifier scores them. The decision threshold
(0.96) and an 11-match-per-entity cap are chosen purely from a held-out validation
split by sweeping macro F0.5 — never a fixed 0.5 cutoff. Measured validation macro
F0.5 is **0.7962** across all 442,191 held-out S1 entities.

**Approach Type:** Blocking + Classifier
**Core Innovation:** Compound (name-token + address-number) blocking key,
frequency-capped on the source side *before* any join executes, is what makes
full-scale (2.2M×5M+) candidate generation computationally tractable at all — the
naive versions (single rare token, or name+address-word) blow up to 30-500x more
candidates for a few points of extra recall (measured, see Section 3).

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
**Core Innovation:** see above (frequency-capped compound blocking key).

Ten phases, each an independent, rerunnable module under `src/`:
normalize → split → block → features → train → threshold → evaluate → predict,
chained by `run_pipeline.py`.

---

## 3. Candidate Generation (Blocking)

Implemented in `src/blocking.py`, entirely as DuckDB equi-joins over the normalized
Parquet tables from `src/normalize.py` — no per-row regex or Python nested loops (an
existing `analysis/test_union_blocking.py`, which does per-row regex joins, was
observed to not finish in 25+ minutes on a 20k-row sample and was abandoned in favor
of this design).

**Blocking keys used** (measured on a 442,191-entity held-out slice, joined against
the *full* ~5M-row S2/S3 corpora — see `logs/blocking_measure.log` for the full
incremental sweep):

| Block | Candidates | Recall (pair-level) | Avg/S1 |
|---|---:|---:|---:|
| Exact normalized name | 2.8M | 10.1% | 3.0 |
| Exact transliterated name | +0.18M | 11.9% | 3.4 |
| Rare name token (doc-freq ≤ 60) | +2.9M | 18.7% | 9.9 |
| Rare transliterated token (doc-freq ≤ 60) | +0.02M | 18.8% | 10.0 |
| Name-token + address-number (compound key, doc-freq ≤ 50, token len ≥ 4) | +16.9M | **69.8%** | 45-48 |

Two additional candidate blocks were measured and **rejected**:
- **Name-token + address-word** (a two-field compound analogous to the winning one,
  but on address *words* instead of numbers): the source-side pair-frequency table
  alone blew DuckDB's memory budget at full scale (many more distinct address words
  per record than numbers) — the exact "candidate explosion" CLAUDE.md's own EDA had
  already flagged for this block type.
- **Two-distinct-name-token intersection**: same failure mode (self-join fan-out on
  long name/address-like strings).
- An exact per-S1 hardest-negative rank (`ROW_NUMBER() OVER (PARTITION BY s1_id ...)`)
  for training-negative selection hit the same wall at ~80M rows even with disk-spill
  configured; replaced with a similarity-floor filter (`name_ratio≥40 OR addr_ratio≥40
  OR shared_numeric_bool`) + random subsampling to a target ratio — cheaper (a single
  streaming filter pass, no large sort buffer) at the cost of being an approximation
  of "hardest" rather than an exact top-K.

**Final candidate volume** (full scale, both S2 and S3, both blocking-eval directions):

| Split | Candidate pairs | Avg/S1 | Pair-level recall ceiling | Entity-level recall ceiling* |
|---|---:|---:|---:|---:|
| Train (2,206,821 S1) | 100,035,646 | 45.33 | 69.82% | **90.27%** |
| Test (1,732,544 S1) | 82,133,918 | 47.41 | n/a (no test ground truth) | n/a |

*Entity-level recall = fraction of S1 entities *with* a true match that have *at
least one* true match present among their candidates — this, not pair-level recall,
is what actually bounds achievable macro F0.5 (a multi-match entity with 3 of 4 true
matches blocked still contributes a non-zero, credit-earning row).

**How true matches were not lost:** blocking recall was measured against the
deterministic validation split's ground truth at every stage (not assumed), and every
block is a plain equi-join — no similarity threshold is applied before the model sees
a candidate, so recall lost at this stage is lost to *absence from the corpus join
keys*, not to a premature scoring cutoff. A cap (`COMPOUND_RARE_MAX=150`) was also
measured to reach 91.8% entity-level recall at ~92 candidates/S1, but the resulting
~370M-pair train+test volume made downstream RapidFuzz feature extraction operationally
heavy (multi-hour, and an earlier multiprocessing implementation caused a system-wide
memory/swap crisis on the 16GB development machine); `COMPOUND_RARE_MAX=50` was kept as
the simpler, safer, still-strong-recall operating point actually used for the
submitted run.

---

## 4. Matching Model

**Features used** (22 total, computed in `src/features.py`; full names/addresses are
never truncated before features are computed — see `src/normalize.py` for the
Unicode-preserving + transliterated + tokenized representations built for every
record):

- **Name:** RapidFuzz `ratio`, `token_sort_ratio`, `token_set_ratio` (normalized
  Unicode-preserving text); RapidFuzz `ratio` on the transliterated text; exact-match
  flags (normalized and transliterated); Jaccard token overlap; common-token count;
  length difference/ratio; missing-name flags (both sides)
- **Address:** RapidFuzz `ratio`, `token_set_ratio`; exact-match flag; Jaccard token
  overlap; common-token count; shared-numeric-token count and boolean; length
  difference/ratio; missing-address flags (both sides)
- **Other:** country equality; source indicator (S2 vs S3)

**Model type:** LightGBM (MIT-licensed gradient-boosted trees; a scale/license-safe
choice well under the 8B-parameter cap for a tabular pair-classification task).
`objective=binary`, `num_leaves=63`, `learning_rate=0.05`, `scale_pos_weight` set from
the observed class imbalance, up to 800 estimators with early stopping (50-round
patience) against a held-out 10% dev slice carved from the TRAIN-split entities only.

**Training set:** every retained true-positive candidate pair from the TRAIN-split
half of the S1 entities (4,265,553 rows), plus hard-negative-sampled non-matches
(35,285,043 rows, ≈1:8.3 positive:negative) drawn from the same blocking buckets and
biased toward higher name/address similarity — training the model to distinguish
TRUE MATCH from VERY SIMILAR BUT WRONG MATCH, per CLAUDE.md section 16, rather than
TRUE MATCH vs obviously-unrelated. Best iteration: 314, dev-set average precision
0.9962.

**Threshold selection method:** macro-F0.5 sweep (0.05 → 0.999, finer-grained above
0.90) on the VAL-split S1 entities — 442,191 entities held out completely from
training. Best threshold: **0.960** (macro F0.5 = 0.79620); the curve rises
monotonically up to 0.96 and falls off above it, confirming this is a true interior
optimum, not a boundary artifact of the search grid. A hard cap of 11 matches/S1 (the
observed ground-truth max) is applied on top of the threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** **0.79620** (n = 442,191 S1 entities, entirely
  held out from model training and threshold selection)

**Breakdown** (`src/evaluate.py`, full detail in `experiments/validation_summary.csv`):

| Slice | Macro F0.5 | n |
|---|---:|---:|
| Overall | 0.7962 | 442,191 |
| Country = US | 0.8622 | 264,836 |
| Country = India | 0.6977 | 177,355 |
| Singletons (no true match) | 0.8976 | 24,864 |
| Has ≥1 true match | 0.7902 | 417,327 |

No France row exists here because training ground truth contains zero France
examples — this split is the best available generalization proxy, not a substitute
for real France-labeled evaluation, which the challenge does not provide.

- **Common false positives (wrong merges):** sampled in
  `experiments/error_false_positives.csv` / `error_high_confidence_false_positives.csv`
  — inspected cases are dominated by chain-like businesses sharing near-identical
  names across different branch addresses, and by address components colliding on a
  fairly common (but blocking-rare-enough) token+number pair.
- **Common false negatives (missed matches):** `experiments/error_false_negatives_scored.csv`
  covers pairs the model saw but scored below threshold (usually heavy noise on both
  name and address simultaneously); separately, 8,509 of 417,327 (2.0%) val S1
  entities with a true match had **zero** candidates at all — a pure blocking miss,
  the residual cost of capping blocking volume for compute feasibility.
- **Low-confidence true positives / high-confidence false positives:**
  `experiments/error_low_confidence_true_positives.csv` and
  `error_high_confidence_false_positives.csv` — used to sanity-check that the
  threshold sits in a genuinely ambiguous region rather than an obviously-wrong one.

**Final test predictions:** 1,732,544 S1 entities scored; 1,458,997 (84.2%) received
≥1 match, totaling 4,427,819 matched pairs (mean 2.56/S1); the remaining 273,547 are
predicted as singletons. `utils/validate_submission.py --check-ids` passes with no
errors or warnings beyond the informational ones.

---

## 6. Conclusion

A DuckDB-equi-join blocking stage (measured, iteratively pruned to drop two block
types that caused memory blowups for negligible recall gain) feeding a LightGBM
classifier over 22 RapidFuzz/set-overlap features reaches macro F0.5 = 0.796 on a
genuinely held-out validation split, with every threshold and cap chosen from that
same validation data rather than assumed. The main lesson learned operationally: at
this data scale, the *cheapest correct* implementation (single-process feature
extraction, frequency-pre-filtered compound blocking keys) was consistently more
reliable than a parallelized or naively-capped version that looked faster on paper —
several iterations here were forced by real out-of-memory failures at full scale, not
anticipated in advance from small-sample testing alone.

---

## Appendix

### A. Code Artefacts

```
src/
├── config.py       # paths, seeds, thresholds
├── normalize.py     # Phase 1: text normalization, transliteration, tokenization
├── split.py         # Phase 2: deterministic 80/20 S1-level train/val split
├── blocking.py       # Phase 3: candidate generation (measure + generate modes)
├── features.py       # Phase 4: RapidFuzz + set-overlap pairwise features
├── train.py          # Phase 5/7: hard-negative sampling + LightGBM training
├── threshold.py       # Phase 6: macro-F0.5 threshold sweep on held-out val
├── evaluate.py        # Phase 9 (val half): breakdown + error-analysis samples
└── predict.py         # Phase 8: final test scoring + TSV output
run_pipeline.py         # chains all of the above in order
```

Reproduce end-to-end with `python3 run_pipeline.py` (see its docstring for expected
runtime — feature extraction over ~180M total candidate pairs dominates at
single-process RapidFuzz throughput).

### B. Additional Results

- `logs/blocking_measure.log` — full incremental blocking sweep (every block,
  measured, before any were pruned)
- `logs/generate_train.log`, `logs/generate_test.log` — final full-scale candidate
  generation
- `experiments/threshold_sweep.csv` — the complete macro-F0.5-vs-threshold curve
- `experiments/validation_summary.csv` — per-entity validation scores (country, true
  match count, F0.5)
- `experiments/error_*.csv` — sampled error-analysis cases with original + normalized
  text on both sides

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
