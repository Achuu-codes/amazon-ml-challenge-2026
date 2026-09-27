import duckdb
import time

SAMPLE_N = 20_000

TRAIN = "dataset/train"
DB = duckdb.connect()

DB.execute("SET memory_limit='10GB'")
DB.execute("SET threads=4")

print("=" * 80)
print("UNION BLOCKING EXPERIMENT")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Deterministic S1 sample
# ---------------------------------------------------------------------

print("\n[1/6] Loading S1 sample...")

DB.execute(f"""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT
    *,
    lower(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    coalesce(business_name, ''),
                    '[^[:alnum:] ]', ' ', 'g'
                ),
                '\\s+', ' ', 'g'
            ),
            '^ | $', '', 'g'
        )
    ) AS name_norm,
    lower(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    coalesce(business_address, ''),
                    '[^[:alnum:] ]', ' ', 'g'
                ),
                '\\s+', ' ', 'g'
            ),
            '^ | $', '', 'g'
        )
    ) AS addr_norm,
    lower(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    coalesce(country, ''),
                    '[^[:alnum:] ]', ' ', 'g'
                ),
                '\\s+', ' ', 'g'
            ),
            '^ | $', '', 'g'
        )
    ) AS country_norm
FROM read_csv_auto(
    '{TRAIN}/train_source1.tsv',
    delim='\\t',
    header=true,
    ignore_errors=true
)
USING SAMPLE reservoir({SAMPLE_N} ROWS) REPEATABLE (42);
""")

s1_count = DB.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
print(f"S1 sample: {s1_count:,}")


# ---------------------------------------------------------------------
# 2. Ground truth
# ---------------------------------------------------------------------

print("\n[2/6] Loading ground truth...")

DB.execute(f"""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT
    source1_entity_id,
    unnest(string_split(matched_entity_ids, ','))
        AS matched_entity_id
FROM read_csv_auto(
    '{TRAIN}/train_ground_truth.tsv',
    delim='\\t',
    header=true,
    ignore_errors=true
)
WHERE matched_entity_ids IS NOT NULL
  AND matched_entity_ids != '';
""")

DB.execute("""
CREATE OR REPLACE TEMP TABLE true_pairs AS
SELECT
    s1.entity_id AS s1_id,
    gt.matched_entity_id AS s2s3_id
FROM s1
JOIN gt
  ON s1.entity_id = gt.source1_entity_id;
""")

true_count = DB.execute("""
SELECT COUNT(*) FROM true_pairs
""").fetchone()[0]

print(f"True pairs in sample: {true_count:,}")


# ---------------------------------------------------------------------
# 3. Source views
# ---------------------------------------------------------------------

print("\n[3/6] Preparing source views...")

for source, path in [
    ("s2", f"{TRAIN}/train_source2.tsv"),
    ("s3", f"{TRAIN}/train_source3.tsv")
]:
    DB.execute(f"""
    CREATE OR REPLACE TEMP VIEW {source}_raw AS
    SELECT *
    FROM read_csv_auto(
        '{path}',
        delim='\\t',
        header=true,
        ignore_errors=true
    );
    """)


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def create_source_view(source):

    DB.execute(f"""
    CREATE OR REPLACE TEMP VIEW {source} AS
    SELECT
        entity_id,

        lower(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        coalesce(business_name, ''),
                        '[^[:alnum:] ]', ' ', 'g'
                    ),
                    '\\\\s+', ' ', 'g'
                ),
                '^ | $', '', 'g'
            )
        ) AS name_norm,

        lower(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        coalesce(business_address, ''),
                        '[^[:alnum:] ]', ' ', 'g'
                    ),
                    '\\\\s+', ' ', 'g'
                ),
                '^ | $', '', 'g'
            )
        ) AS addr_norm,

        lower(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        coalesce(country, ''),
                        '[^[:alnum:] ]', ' ', 'g'
                    ),
                    '\\\\s+', ' ', 'g'
                ),
                '^ | $', '', 'g'
            )
        ) AS country_norm

    FROM {source}_raw;
    """)


create_source_view("s2")
create_source_view("s3")


# ---------------------------------------------------------------------
# Build token tables
# ---------------------------------------------------------------------

print("\n[4/6] Building blocking indexes...")


def build_indexes(source):

    # Name tokens
    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source}_name_tokens AS
    SELECT DISTINCT
        entity_id,
        token
    FROM (
        SELECT
            entity_id,
            unnest(
                string_split(name_norm, ' ')
            ) AS token
        FROM {source}
    )
    WHERE length(token) >= 3;
    """)

    # Address tokens
    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source}_addr_tokens AS
    SELECT DISTINCT
        entity_id,
        token
    FROM (
        SELECT
            entity_id,
            unnest(
                string_split(addr_norm, ' ')
            ) AS token
        FROM {source}
    )
    WHERE length(token) >= 3;
    """)

    # Numeric address tokens
    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source}_addr_numbers AS
    SELECT DISTINCT
        entity_id,
        token
    FROM (
        SELECT
            entity_id,
            unnest(
                string_split(addr_norm, ' ')
            ) AS token
        FROM {source}
    )
    WHERE regexp_matches(token, '^[0-9]+$');
    """)


build_indexes("s2")
build_indexes("s3")


# ---------------------------------------------------------------------
# Candidate block helper
# ---------------------------------------------------------------------

blocks = {}


def run_block(name, sql):

    start = time.time()

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE block_{name} AS
    {sql}
    """)

    count = DB.execute(
        f"SELECT COUNT(*) FROM block_{name}"
    ).fetchone()[0]

    true_retained = DB.execute(f"""
        SELECT COUNT(*)
        FROM block_{name} b
        JOIN true_pairs t
          ON b.s1_id = t.s1_id
         AND b.candidate_id = t.s2s3_id
    """).fetchone()[0]

    recall = true_retained / true_count

    print(
        f"{name:<35}"
        f" candidates={count:>12,}"
        f" true={true_retained:>8,}"
        f" recall={recall:>8.3%}"
        f" time={time.time()-start:.1f}s"
    )

    blocks[name] = count


# ---------------------------------------------------------------------
# S2 BLOCKS
# ---------------------------------------------------------------------

print("\n[5/6] Running complementary blocks...")
print("\nS2 BLOCKS")
print("-" * 90)

# A: exact normalized name
run_block("S2_exact_name", """
SELECT
    a.entity_id AS s1_id,
    b.entity_id AS candidate_id
FROM s1 a
JOIN s2 b
  ON a.name_norm = b.name_norm
WHERE a.name_norm != ''
""")


# B: name token + address number
run_block("S2_name_num", """
SELECT DISTINCT
    a.entity_id AS s1_id,
    b.entity_id AS candidate_id
FROM s1 a
JOIN s2_name_tokens nt
  ON regexp_matches(a.name_norm, '(^| )' || nt.token || '( |$)')
JOIN s2_addr_numbers an
  ON nt.entity_id = an.entity_id
JOIN s2 b
  ON b.entity_id = an.entity_id
WHERE regexp_matches(a.addr_norm, '(^| )' || an.token || '( |$)')
""")


# C: name token + address token
run_block("S2_name_addr", """
SELECT DISTINCT
    a.entity_id AS s1_id,
    b.entity_id AS candidate_id
FROM s1 a
JOIN s2_name_tokens nt
  ON regexp_matches(a.name_norm, '(^| )' || nt.token || '( |$)')
JOIN s2_addr_tokens at
  ON nt.entity_id = at.entity_id
JOIN s2 b
  ON b.entity_id = at.entity_id
WHERE regexp_matches(a.addr_norm, '(^| )' || at.token || '( |$)')
""")


# ---------------------------------------------------------------------
# S3 BLOCKS
# ---------------------------------------------------------------------

print("\nS3 BLOCKS")
print("-" * 90)

run_block("S3_exact_name", """
SELECT
    a.entity_id AS s1_id,
    b.entity_id AS candidate_id
FROM s1 a
JOIN s3 b
  ON a.name_norm = b.name_norm
WHERE a.name_norm != ''
""")


run_block("S3_name_num", """
SELECT DISTINCT
    a.entity_id AS s1_id,
    b.entity_id AS candidate_id
FROM s1 a
JOIN s3_name_tokens nt
  ON regexp_matches(a.name_norm, '(^| )' || nt.token || '( |$)')
JOIN s3_addr_numbers an
  ON nt.entity_id = an.entity_id
JOIN s3 b
  ON b.entity_id = an.entity_id
WHERE regexp_matches(a.addr_norm, '(^| )' || an.token || '( |$)')
""")


run_block("S3_name_addr", """
SELECT DISTINCT
    a.entity_id AS s1_id,
    b.entity_id AS candidate_id
FROM s1 a
JOIN s3_name_tokens nt
  ON regexp_matches(a.name_norm, '(^| )' || nt.token || '( |$)')
JOIN s3_addr_tokens at
  ON nt.entity_id = at.entity_id
JOIN s3 b
  ON b.entity_id = at.entity_id
WHERE regexp_matches(a.addr_norm, '(^| )' || at.token || '( |$)')
""")


# ---------------------------------------------------------------------
# UNION
# ---------------------------------------------------------------------

print("\n[6/6] Measuring UNION...")
print("=" * 90)

DB.execute("""
CREATE OR REPLACE TEMP TABLE all_candidates AS

SELECT * FROM block_S2_exact_name
UNION
SELECT * FROM block_S2_name_num
UNION
SELECT * FROM block_S2_name_addr
UNION
SELECT * FROM block_S3_exact_name
UNION
SELECT * FROM block_S3_name_num
UNION
SELECT * FROM block_S3_name_addr;
""")

union_count = DB.execute("""
SELECT COUNT(*) FROM all_candidates
""").fetchone()[0]

union_true = DB.execute("""
SELECT COUNT(*)
FROM all_candidates c
JOIN true_pairs t
  ON c.s1_id = t.s1_id
 AND c.candidate_id = t.s2s3_id
""").fetchone()[0]

union_recall = union_true / true_count

avg_candidates = union_count / s1_count

p95 = DB.execute("""
SELECT quantile_cont(cnt, 0.95)
FROM (
    SELECT s1_id, COUNT(*) AS cnt
    FROM all_candidates
    GROUP BY s1_id
)
""").fetchone()[0]

max_candidates = DB.execute("""
SELECT MAX(cnt)
FROM (
    SELECT s1_id, COUNT(*) AS cnt
    FROM all_candidates
    GROUP BY s1_id
)
""").fetchone()[0]

print()
print(f"Total unique candidates : {union_count:,}")
print(f"True pairs retained     : {union_true:,}")
print(f"UNION recall            : {union_recall:.4%}")
print(f"Average candidates/S1   : {avg_candidates:,.1f}")
print(f"P95 candidates/S1       : {p95:,.0f}")
print(f"Max candidates/S1      : {max_candidates:,}")

print("\nDONE")