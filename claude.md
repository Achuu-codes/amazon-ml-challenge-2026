You are the lead ML engineer responsible for completing this entire project.

WORKING DIRECTORY:
 /Users/shreyas/Developer/amazon-ml/student_resource

You have full access to the project files and terminal.

Your job is NOT to merely explain the solution. You must inspect the existing project, implement the complete solution, run experiments, debug failures, optimize performance, generate the final submission, validate it, and leave the project in a reproducible state.

============================================================
1. PROBLEM
============================================================

This is the Amazon ML Challenge 2026 entity-resolution / record-linkage problem.

There are three sources:

Source 1:
- canonical/reference entity set
- every S1 entity must receive a list of matching S2/S3 entity IDs

Source 2:
- noisy independently generated representation of the same underlying entities

Source 3:
- another noisy independently generated representation

The mapping is NOT one-to-one.

For each S1 entity, there may be:
- zero matches
- one match
- multiple matches

Ground truth allows up to 11 matches for an S1 entity.

The final prediction must contain every S1 entity, including those with no matches.

============================================================
2. DATA
============================================================

Project structure:

student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_ground_truth.tsv
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   └── train_source3.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py
├── README.md
├── Documentation_template.md
└── analysis/
    ├── profile_dataset.py
    ├── analyze_matches.py
    ├── test_transliteration.py
    ├── test_blocking.py
    ├── test_compound_blocking.py
    ├── test_compound_blocking_fast.py
    ├── test_union_blocking.py
    └── test_advanced_blocking.py

Do not assume the files are exactly as described. Inspect them first.

============================================================
3. HARD CONSTRAINTS
============================================================

VERY IMPORTANT:

No external entity data may be used.

Do NOT:
- query Google
- query business registries
- use geocoding APIs
- use external business databases
- scrape websites
- use external company databases
- augment records from the internet

Only use the supplied datasets and locally installed libraries.

The test set contains FRANCE even though training contains only US and India.

Therefore the method must generalize across countries and must NOT hardcode US/India-specific rules.

============================================================
4. EVALUATION
============================================================

Primary metric:

Macro F_0.5.

Precision is weighted more heavily than recall.

False positive / false merge errors are significantly more expensive than missed matches.

Therefore:

DO NOT simply maximize recall.

The final system must be conservative.

Correctly outputting an empty list for an entity with no match is valuable.

Do not hallucinate matches.

Thresholds must be selected using validation data and F0.5.

============================================================
5. DATASET FACTS ALREADY DISCOVERED
============================================================

Training:

S1:
2,206,821 rows

S2:
5,034,616 rows

S3:
5,285,603 rows

S1 ground truth:
2,206,821 entities

Singleton/no-match S1 entities:
123,247

Singleton/no-match rate:
approximately 5.58%

Matched S1 entities:
2,083,574

Matches per S1:
mean approximately 3.46
maximum 11

Distribution:

0 -> 123247
1 -> 119157
2 -> 375212
3 -> 530841
4 -> 484115
5 -> 321957
6 -> 164868
7 -> 63968
8 -> 18680
9 -> 4205
10 -> 534
11 -> 37

Test:

S1:
1,732,544

S2:
4,887,273

S3:
5,082,316

Test S1 countries:

India: 809,986
US: 663,106
France: 259,452

Therefore France is approximately 15% of test S1.

Missing test S2 names:
129,408

Missing test S3 names:
136,098

Missing training S2 addresses:
168,967

Missing training S3 addresses:
175,916

============================================================
6. SCHEMA
============================================================

S1:

entity_id
business_name
business_address
country

S2:

entity_id
business_name
business_address
country

S3:

entity_id
business_name
business_address
country

Ground truth:

source1_entity_id
matched_entity_ids

Example:

S1-965667
S2-681193310,S2-743505751,S3-775321672,...

============================================================
7. IMPORTANT DISCOVERIES FROM EDA
============================================================

We already performed several experiments.

TRUE MATCH ANALYSIS:

From approximately 34,567 true S1->S2/S3 pairs:

S2:
normalized name exact approximately 21.75%
normalized address exact approximately 12.63%

Name similarity:
mean approximately 78.30
median approximately 88.14
p75 approximately 96.43

Address similarity:
mean approximately 79.35
median approximately 86.96
p75 approximately 94.74

S3:
normalized name exact approximately 22.05%
normalized address exact approximately 4.68%

Name similarity:
mean approximately 80.07
median approximately 87.50
p75 approximately 96.55

Address similarity:
mean approximately 72.80
median approximately 80.00
p75 approximately 88.89

Conclusion:

Exact matching alone is inadequate.

Fuzzy similarity is necessary.

============================================================
8. MULTILINGUAL DISCOVERY
============================================================

Some true matches have completely different scripts.

Examples observed include:

Latin business name
vs Hindi representation

Latin
vs Kannada

Latin
vs Gujarati

Latin
vs Tamil

etc.

Therefore Unicode-preserving and transliterated representations are both required.

Transliteration experiment:

S2:
transliterated exact approximately 19.38%

similarity mean approximately 83.63
median approximately 87.18

>=70 approximately 82.56%
>=80 approximately 65.92%
>=85 approximately 55.27%
>=90 approximately 42.84%
>=95 approximately 26.61%

S3:
transliterated exact approximately 19.69%

similarity mean approximately 82.63
median approximately 86.67

>=70 approximately 80.79%
>=80 approximately 64.20%
>=85 approximately 53.81%
>=90 approximately 42.49%
>=95 approximately 27.12%

Use transliteration as an important signal.

But NEVER make transliteration similarity alone the final matching criterion.

============================================================
9. ADDRESS DISCOVERY
============================================================

Approximately 20,000 S1 entities were analyzed.

True S2 pairs:

~79.46% share at least one numeric address token.

Mean shared numeric tokens:
~1.181

True S3 pairs:

~80.68% share at least one numeric address token.

Mean shared numeric tokens:
~1.195

Therefore address numbers are useful blocking evidence.

BUT:

Do NOT use an address number alone as a blocking key.

Common numbers produce huge candidate buckets.

Use combinations such as:

name evidence + address number

name evidence + rare address token

etc.

============================================================
10. BLOCKING EXPERIMENTS ALREADY COMPLETED
============================================================

Exact normalized name:

S2:
~21.64% recall in one experiment

S3:
similar low recall

Exact transliterated name:

~25.6% recall

Single rare name token:

~57-58% recall

BUT:
candidate explosion:
~46 million candidate pairs.

Address number:

~47% recall

BUT:
candidate explosion:
~33 million candidate pairs.

Compound blocks:

Name token + address number:

S2:
30.263% recall
5,688,950 candidates
284 candidates/S1

S3:
34.256% recall
6,514,266 candidates
326 candidates/S1

Name token + address token:

S2:
36.440% recall
25,854,990 candidates
1,293 candidates/S1

S3:
40.332% recall
26,370,873 candidates
1,319 candidates/S1

These experiments demonstrate:

1. Blocking is the main bottleneck.
2. A single blocking strategy is insufficient.
3. Candidate generation needs multiple complementary passes.
4. Candidate explosion must be controlled.

============================================================
11. CURRENT EXPERIMENTAL CODE
============================================================

DuckDB has been installed:

DuckDB:
1.5.5

Python:
3.12.13

Pandas:
3.0.6

NumPy:
2.5.3

RapidFuzz is installed.

Unidecode is installed.

Use DuckDB for large-scale candidate generation rather than loading millions of rows into Python dictionaries.

An earlier Python iterrows-based compound blocking implementation consumed approximately 36GB RAM and became unusable.

DO NOT repeat that approach.

============================================================
12. REQUIRED ARCHITECTURE
============================================================

Build the final system as a pipeline:

PHASE 1
Data loading + normalization

PHASE 2
EDA / validation

PHASE 3
Candidate generation / blocking

PHASE 4
Pairwise feature extraction

PHASE 5
Supervised matching model

PHASE 6
Threshold optimization for F0.5

PHASE 7
Hard-negative handling

PHASE 8
Final prediction

PHASE 9
Submission validation

PHASE 10
Documentation

============================================================
13. NORMALIZATION
============================================================

Create multiple representations.

For business names:

1. original Unicode normalized
2. lowercase normalized
3. punctuation-normalized
4. whitespace-normalized
5. transliterated representation
6. token representation
7. token-sorted representation if useful

Do NOT destroy useful Unicode information.

For addresses:

Create:

1. normalized address
2. token set
3. numeric token set
4. postal-code-like numeric tokens if detectable
5. locality/city-like tokens where inferable from the supplied text

Do not use external geocoding.

Country should be retained as a feature.

Do not hardcode only US/India.

============================================================
14. BLOCKING
============================================================

Implement multiple complementary blocking passes.

At minimum investigate:

A. exact normalized name

B. exact transliterated name

C. rare name token

D. rare transliterated name token

E. rare name token + address number

F. transliterated name token + address number

G. rare name token + rare address token

H. transliterated name token + rare address token

I. two-name-token intersection

J. address-number + rare-address-token

K. country + name-derived block

BUT:

Do not blindly implement all of these.

Measure:

- candidate count
- candidate recall
- average candidates/S1
- P95 candidates/S1
- max candidates/S1
- incremental recall

Select blocks based on incremental value.

The final candidate generator should aim for very high recall while keeping the candidate set computationally feasible.

============================================================
15. TRAIN/VALIDATION DESIGN
============================================================

Do NOT train on all ground truth without validation.

Create a deterministic validation split at the S1 entity level.

For example:

80% S1 entities:
training

20% S1 entities:
validation

Ensure all true matched pairs belonging to an S1 entity stay together.

Do not leak S1 entities between train and validation.

Use the validation set for:

- threshold selection
- model selection
- blocking evaluation
- error analysis

============================================================
16. NEGATIVE SAMPLING
============================================================

This is extremely important.

Random negatives are too easy.

For each true positive candidate, generate hard negatives from the same blocking buckets.

Examples:

same country
same name token
similar name
same address number
similar address
same transliteration token

Hard negatives should resemble actual false matches.

Train the model to distinguish:

TRUE MATCH
vs
VERY SIMILAR BUT WRONG MATCH

Because F0.5 heavily penalizes false positives.

============================================================
17. FEATURES
============================================================

Build pairwise features.

NAME:

- RapidFuzz ratio
- WRatio
- token_sort_ratio
- token_set_ratio
- partial_ratio
- Jaro-Winkler if available
- normalized edit similarity
- character n-gram similarity
- exact normalized equality
- exact transliterated equality
- transliterated fuzzy similarity
- token overlap
- Jaccard similarity
- common token count
- rare-token overlap
- length difference
- length ratio

ADDRESS:

- RapidFuzz ratio
- token similarity
- token-set similarity
- Jaccard
- shared numeric token count
- shared numeric token boolean
- rare address token overlap
- character n-gram similarity
- length difference
- length ratio

OTHER:

- country equality
- source indicator: S2 or S3
- missing address flags
- missing name flags
- number of common tokens
- exact field matches

Do NOT rely on one feature.

============================================================
18. MODEL
============================================================

Experiment with a model appropriate for tabular pair classification.

Candidates:

1. LightGBM if available
2. XGBoost if available
3. CatBoost if available
4. HistGradientBoostingClassifier
5. Logistic regression as baseline

The model should output:

P(match | pair)

Evaluate using validation data.

Because the final metric is F0.5, optimize threshold based on validation F0.5 rather than accuracy.

Do NOT blindly use threshold 0.5.

============================================================
19. IMPORTANT: F0.5 THRESHOLDING
============================================================

For every S1 entity:

candidate pairs receive probabilities.

Then choose matches using a threshold.

But also investigate entity-level safeguards.

For example:

- high threshold for weak evidence
- lower threshold only when multiple independent signals agree
- exact/high-confidence matches can use stricter deterministic rules
- singleton/no-match entities should be treated conservatively
- don't force at least one match

Never force a match for every S1.

Empty prediction is valid and sometimes optimal.

============================================================
20. CROSS-SOURCE CONSISTENCY
============================================================

Investigate whether S2 and S3 independently support the same S1 entity.

For example:

S1
 |
 +-- high-confidence S2
 |
 +-- high-confidence S3

If both independently agree, this may provide additional confidence.

However:

DO NOT assume S2/S3 records are one-to-one.

Do not accidentally eliminate valid multiple matches.

Treat cross-source agreement as an additional feature or confidence signal, not as a hard one-to-one constraint.

============================================================
21. GRAPH / CLUSTERING
============================================================

Optionally investigate entity-resolution graph structure:

S1 -- S2
S1 -- S3

Potentially:

S2 and S3 records that strongly resemble the same S1 can provide corroborating evidence.

But this should only be introduced if validation shows improvement.

Do not over-engineer.

============================================================
22. ERROR ANALYSIS
============================================================

After training:

Produce tables of:

1. false positives
2. false negatives
3. low-confidence true positives
4. high-confidence false positives
5. multilingual failures
6. address-corruption failures
7. missing-address cases
8. missing-name cases
9. singleton/no-match cases
10. France test/generalization-related cases if validation allows analogous analysis

For each failure inspect:

- original S1 name/address
- candidate name/address
- normalized versions
- transliterated versions
- feature values
- model probability
- ground truth

Use this to improve blocking/features.

============================================================
23. SCALABILITY
============================================================

The full data contains millions of rows.

Avoid:

- pandas merge across all S1 x S2
- Python nested loops
- iterrows
- giant Python dictionaries containing every pair
- all-pairs fuzzy matching

Use:

- DuckDB
- SQL joins
- indexed/token tables
- vectorized operations
- batch processing
- RapidFuzz only after blocking
- compact feature tables

Candidate generation must happen BEFORE expensive fuzzy matching.

============================================================
24. FINAL PREDICTION
============================================================

For every test S1 entity:

generate candidate S2/S3 records

score candidates

apply optimized threshold

output:

S1_ID    matched_entity_ids

If there are no matches:

S1_ID    empty list representation required by challenge format

Preserve all S1 rows.

Do not drop singleton/no-match entities.

Do not duplicate candidate IDs.

Do not create invalid IDs.

============================================================
25. SUBMISSION VALIDATION
============================================================

Use:

utils/validate_submission.py

Run the official validator.

Fix every error/warning that affects validity.

Check:

- every S1 test ID appears
- candidate IDs actually exist
- no duplicate matches
- output format exactly correct
- candidate_pairs.tsv satisfies candidate-pair constraints
- matching_results.tsv is valid

============================================================
26. IMPORTANT OUTPUT FILES
============================================================

Create something like:

submission/
├── candidate_pairs.tsv
├── matching_results.tsv
└── validation_report.txt

Also maintain:

models/
experiments/
features/
logs/

Do not commit massive temporary intermediate files unnecessarily.

============================================================
27. REPRODUCIBILITY
============================================================

Every experiment must be deterministic where possible.

Use fixed seeds.

Create scripts such as:

src/
├── config.py
├── normalize.py
├── blocking.py
├── features.py
├── train.py
├── threshold.py
├── predict.py
└── evaluate.py

Or another clean architecture if the existing project suggests something better.

There must be one obvious command to reproduce the final pipeline.

For example:

python run_pipeline.py

============================================================
28. DOCUMENTATION
============================================================

Complete Documentation_template.md using actual measured results.

Do NOT invent metrics.

Document:

- EDA
- noise patterns
- normalization
- blocking keys
- candidate counts
- candidate recall
- feature engineering
- model
- negative sampling
- threshold selection
- F0.5 validation
- error analysis
- final architecture
- runtime
- memory considerations
- limitations

Use actual experimental numbers.

============================================================
29. DEVELOPMENT PROCESS
============================================================

Work iteratively.

FIRST:
inspect every existing file and current implementation.

SECOND:
run existing scripts where useful.

THIRD:
identify broken/incomplete code.

FOURTH:
implement the candidate-generation benchmark.

FIFTH:
measure recall and candidate volume.

SIXTH:
build validation split.

SEVENTH:
build features.

EIGHTH:
train baseline model.

NINTH:
optimize threshold.

TENTH:
perform hard-negative mining.

ELEVENTH:
run error analysis.

TWELFTH:
improve the pipeline based on validation evidence.

THIRTEENTH:
run the full test pipeline.

FOURTEENTH:
validate submission.

FIFTEENTH:
complete documentation.

============================================================
30. CRITICAL BEHAVIOR
============================================================

Do not stop after creating code.

Actually RUN the code.

If something crashes:
- diagnose it
- fix it
- rerun it

If a method is too slow:
- profile it
- optimize it
- rerun it

If a blocking strategy has poor recall:
- measure alternatives
- don't assume it works

If memory usage becomes dangerous:
- stop the expensive operation
- redesign using DuckDB/batching

Do not ask me to manually copy-paste code between ten files unless absolutely necessary.

You are responsible for integrating the project.

============================================================
31. CURRENT STATUS
============================================================

Already installed:

Python 3.12.13
pandas
numpy
rapidfuzz
unidecode
duckdb 1.5.5

Existing analysis has already established:

- exact matching is insufficient
- transliteration is important
- address numbers are useful
- single-token blocking causes candidate explosion
- compound blocking improves efficiency
- S3 appears somewhat noisier in address representation
- multilingual script changes exist
- candidate generation is the current major challenge

Continue from this point rather than repeating the basic EDA unless verification is necessary.

============================================================
32. SUCCESS CRITERIA
============================================================

The project is NOT complete merely because a model trains.

Consider it complete only when:

[ ] Candidate generation has high measured recall

[ ] Candidate volume is computationally feasible

[ ] Validation split exists

[ ] Hard negatives are used

[ ] Pairwise features are implemented

[ ] Model is trained

[ ] F0.5 threshold is optimized

[ ] Empty predictions are handled correctly

[ ] Test predictions are generated

[ ] candidate_pairs.tsv generated

[ ] matching_results.tsv generated

[ ] official validator passes

[ ] Documentation completed

[ ] Pipeline is reproducible

[ ] No external data was used

[ ] Final runtime is reasonable

============================================================
33. FINAL INSTRUCTION
============================================================

Take ownership of the project.

Do not give me a generic tutorial.

Inspect the actual repository, implement the solution, run experiments, make decisions based on measured validation results, and continue until a valid final submission and documented pipeline exist.

When you report progress, summarize:

1. What you changed
2. What you measured
3. Validation performance
4. Candidate recall
5. Candidate volume
6. Runtime
7. What you are doing next

Do not claim success unless you actually ran and verified the relevant step.