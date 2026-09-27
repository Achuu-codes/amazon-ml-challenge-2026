import duckdb
import time

# ============================================================
# CONFIG
# ============================================================

TRAIN = "dataset/train"
SAMPLE_SIZE = 20_000
MAX_BUCKET = 2_000

S1_FILE = f"{TRAIN}/train_source1.tsv"
S2_FILE = f"{TRAIN}/train_source2.tsv"
S3_FILE = f"{TRAIN}/train_source3.tsv"
GT_FILE = f"{TRAIN}/train_ground_truth.tsv"


# ============================================================
# CONNECTION
# ============================================================

con = duckdb.connect()

# Keep memory controlled
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")

print("=" * 80)
print("FAST COMPOUND BLOCKING EXPERIMENT")
print("=" * 80)

start = time.time()


# ============================================================
# LOAD SAMPLE S1
# ============================================================

print("\n[1/5] Loading deterministic S1 sample...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE sample_s1 AS
SELECT
    entity_id,
    business_name,
    business_address,
    country
FROM read_csv(
    '{S1_FILE}',
    delim='\\t',
    header=true,
    all_varchar=true
)
ORDER BY hash(entity_id)
LIMIT {SAMPLE_SIZE}
""")

s1_count = con.execute(
    "SELECT COUNT(*) FROM sample_s1"
).fetchone()[0]

print(f"S1 sample: {s1_count:,}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\n[2/5] Loading ground truth...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE truth AS
SELECT
    source1_entity_id,
    trim(matched_id) AS matched_entity_id
FROM read_csv(
    '{GT_FILE}',
    delim='\\t',
    header=true,
    all_varchar=true
),
LATERAL unnest(
    string_split(
        coalesce(matched_entity_ids, ''),
        ','
    )
) AS t(matched_id)
WHERE trim(matched_id) <> ''
  AND source1_entity_id IN (
      SELECT entity_id FROM sample_s1
  )
""")

total_true = con.execute(
    "SELECT COUNT(*) FROM truth"
).fetchone()[0]

print(f"True pairs in sample: {total_true:,}")


# ============================================================
# CREATE SOURCE VIEWS
# ============================================================

print("\n[3/5] Preparing source views...")

# We don't load the complete files into Python memory.
# DuckDB scans them as needed.

con.execute(f"""
CREATE OR REPLACE VIEW source2 AS
SELECT
    entity_id,
    business_name,
    business_address,
    country
FROM read_csv(
    '{S2_FILE}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")

con.execute(f"""
CREATE OR REPLACE VIEW source3 AS
SELECT
    entity_id,
    business_name,
    business_address,
    country
FROM read_csv(
    '{S3_FILE}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")


# ============================================================
# NORMALIZATION EXPRESSION
# ============================================================

# Lowercase + replace punctuation with spaces.
#
# We deliberately keep Unicode characters here.
# This is the ORIGINAL-script block, not transliteration.

NAME_EXPR = """
lower(
    regexp_replace(
        regexp_replace(
            coalesce(business_name, ''),
            '[^[:alnum:] ]',
            ' ',
            'g'
        ),
        '[[:space:]]+',
        ' ',
        'g'
    )
)
"""

ADDR_EXPR = """
lower(
    regexp_replace(
        regexp_replace(
            coalesce(business_address, ''),
            '[^[:alnum:] ]',
            ' ',
            'g'
        ),
        '[[:space:]]+',
        ' ',
        'g'
    )
)
"""


# ============================================================
# COMMON STOPWORDS
# ============================================================

STOPWORDS = """
'private',
'limited',
'ltd',
'pvt',
'llp',
'inc',
'incorporated',
'company',
'co',
'corporation',
'corp',
'the',
'and',
'of',
'services',
'service',
'enterprise',
'enterprises',
'business',
'group',
'india'
"""


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def run_block(block_name, s1_keys_sql, source_keys_sql):
    """
    s1_keys_sql:
        produces s1_id, block_key

    source_keys_sql:
        produces entity_id, block_key

    The function:
      1. removes huge buckets
      2. joins S1 ↔ source
      3. deduplicates candidate pairs
      4. compares against ground truth
    """

    print("\n" + "=" * 80)
    print(block_name)
    print("=" * 80)

    t0 = time.time()

    query = f"""
    WITH
    s1_keys AS (
        {s1_keys_sql}
    ),

    source_keys AS (
        {source_keys_sql}
    ),

    -- Remove very common blocking keys.
    valid_keys AS (
        SELECT
            block_key
        FROM source_keys
        GROUP BY block_key
        HAVING COUNT(DISTINCT entity_id) <= {MAX_BUCKET}
    ),

    candidates AS (
        SELECT DISTINCT
            s.s1_id,
            x.entity_id AS matched_entity_id
        FROM s1_keys s
        INNER JOIN valid_keys v
            ON s.block_key = v.block_key
        INNER JOIN source_keys x
            ON x.block_key = s.block_key
    ),

    matched AS (
        SELECT DISTINCT
            c.s1_id,
            c.matched_entity_id
        FROM candidates c
        INNER JOIN truth t
            ON t.source1_entity_id = c.s1_id
           AND t.matched_entity_id = c.matched_entity_id
    )

    SELECT
        (SELECT COUNT(*) FROM candidates) AS candidates,
        (SELECT COUNT(*) FROM matched) AS retained
    """

    candidates, retained = con.execute(query).fetchone()

    recall = retained / total_true if total_true else 0
    avg_candidates = candidates / s1_count

    elapsed = time.time() - t0

    print(f"Candidate pairs:       {candidates:,}")
    print(f"True pairs retained:   {retained:,}")
    print(f"Candidate recall:      {recall:.4%}")
    print(f"Avg candidates / S1:   {avg_candidates:.2f}")
    print(f"Time:                  {elapsed:.1f}s")

    return {
        "block": block_name,
        "candidates": candidates,
        "retained": retained,
        "recall": recall,
        "avg_candidates": avg_candidates,
        "seconds": elapsed,
    }


# ============================================================
# F — NAME TOKEN + ADDRESS NUMBER
# ============================================================

print("\n[4/5] Running compound blocks...")


s1_name_number = f"""
SELECT DISTINCT
    s.entity_id AS s1_id,
    token || '|' || number AS block_key
FROM sample_s1 s,
LATERAL unnest(string_split({NAME_EXPR}, ' ')) AS t(token),
LATERAL unnest(
    regexp_extract_all(
        coalesce(s.business_address, ''),
        '[0-9]+'
    )
) AS t2(number)
WHERE length(token) >= 4
  AND token NOT IN ({STOPWORDS})
  AND token <> ''
  AND number <> ''
"""


source2_name_number = f"""
SELECT DISTINCT
    s.entity_id,
    token || '|' || number AS block_key
FROM source2 s,
LATERAL unnest(string_split({NAME_EXPR}, ' ')) AS t(token),
LATERAL unnest(
    regexp_extract_all(
        coalesce(s.business_address, ''),
        '[0-9]+'
    )
) AS t2(number)
WHERE length(token) >= 4
  AND token NOT IN ({STOPWORDS})
  AND token <> ''
  AND number <> ''
"""


source3_name_number = f"""
SELECT DISTINCT
    s.entity_id,
    token || '|' || number AS block_key
FROM source3 s,
LATERAL unnest(string_split({NAME_EXPR}, ' ')) AS t(token),
LATERAL unnest(
    regexp_extract_all(
        coalesce(s.business_address, ''),
        '[0-9]+'
    )
) AS t2(number)
WHERE length(token) >= 4
  AND token NOT IN ({STOPWORDS})
  AND token <> ''
  AND number <> ''
"""


result_f2 = run_block(
    "F2 — Name token + address number (S2)",
    s1_name_number,
    source2_name_number
)

result_f3 = run_block(
    "F3 — Name token + address number (S3)",
    s1_name_number,
    source3_name_number
)


# ============================================================
# G — NAME TOKEN + ADDRESS WORD
# ============================================================

s1_name_addr = f"""
SELECT DISTINCT
    s.entity_id AS s1_id,
    name_token || '|' || addr_token AS block_key
FROM sample_s1 s,
LATERAL unnest(string_split({NAME_EXPR}, ' ')) AS t(name_token),
LATERAL unnest(string_split({ADDR_EXPR}, ' ')) AS u(addr_token)
WHERE length(name_token) >= 4
  AND name_token NOT IN ({STOPWORDS})
  AND length(addr_token) >= 4
  AND addr_token NOT IN ({STOPWORDS})
  AND addr_token !~ '^[0-9]+$'
"""


source2_name_addr = f"""
SELECT DISTINCT
    s.entity_id,
    name_token || '|' || addr_token AS block_key
FROM source2 s,
LATERAL unnest(string_split({NAME_EXPR}, ' ')) AS n(name_token),
LATERAL unnest(string_split({ADDR_EXPR}, ' ')) AS a(addr_token)
WHERE length(name_token) >= 4
  AND name_token NOT IN ({STOPWORDS})
  AND length(addr_token) >= 4
  AND addr_token NOT IN ({STOPWORDS})
  AND addr_token !~ '^[0-9]+$'
"""


source3_name_addr = f"""
SELECT DISTINCT
    s.entity_id,
    name_token || '|' || addr_token AS block_key
FROM source3 s,
LATERAL unnest(string_split({NAME_EXPR}, ' ')) AS n(name_token),
LATERAL unnest(string_split({ADDR_EXPR}, ' ')) AS a(addr_token)
WHERE length(name_token) >= 4
  AND name_token NOT IN ({STOPWORDS})
  AND length(addr_token) >= 4
  AND addr_token NOT IN ({STOPWORDS})
  AND addr_token !~ '^[0-9]+$'
"""


result_g2 = run_block(
    "G2 — Name token + address token (S2)",
    s1_name_addr,
    source2_name_addr
)

result_g3 = run_block(
    "G3 — Name token + address token (S3)",
    s1_name_addr,
    source3_name_addr
)


# ============================================================
# SUMMARY
# ============================================================

print("\n\n")
print("=" * 80)
print("FAST COMPOUND BLOCKING SUMMARY")
print("=" * 80)

results = [
    result_f2,
    result_f3,
    result_g2,
    result_g3,
]

for r in results:
    print(
        f"{r['block']:<45}"
        f" recall={r['recall']:.4%}  "
        f"candidates={r['candidates']:,}  "
        f"avg={r['avg_candidates']:.1f}  "
        f"time={r['seconds']:.1f}s"
    )

print("\nTotal runtime:", round(time.time() - start, 1), "seconds")
print("\nDONE")