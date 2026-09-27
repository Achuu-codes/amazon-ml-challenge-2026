import duckdb
import time

SAMPLE_N = 20_000
TRAIN = "dataset/train"

DB = duckdb.connect()
DB.execute("SET memory_limit='10GB'")
DB.execute("SET threads=4")

print("=" * 80)
print("ADVANCED BLOCKING EXPERIMENT")
print("=" * 80)

# ============================================================
# 1. SAMPLE S1
# ============================================================

print("\n[1/7] Loading S1 sample...")

DB.execute(f"""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT
    entity_id,

    lower(
        regexp_replace(
            regexp_replace(
                coalesce(business_name, ''),
                '[^[:alnum:] ]', ' ', 'g'
            ),
            '\\s+', ' ', 'g'
        )
    ) AS name_norm,

    lower(
        regexp_replace(
            regexp_replace(
                coalesce(business_address, ''),
                '[^[:alnum:] ]', ' ', 'g'
            ),
            '\\s+', ' ', 'g'
        )
    ) AS addr_norm,

    country
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


# ============================================================
# 2. GROUND TRUTH
# ============================================================

print("\n[2/7] Loading ground truth...")

DB.execute(f"""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT
    source1_entity_id,
    unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
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
    gt.matched_entity_id
FROM s1
JOIN gt
  ON s1.entity_id = gt.source1_entity_id;
""")

true_count = DB.execute(
    "SELECT COUNT(*) FROM true_pairs"
).fetchone()[0]

print(f"True pairs: {true_count:,}")


# ============================================================
# 3. SOURCE TABLES
# ============================================================

print("\n[3/7] Loading sources...")

for source in ["s2", "s3"]:

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source} AS
    SELECT
        entity_id,

        lower(
            regexp_replace(
                regexp_replace(
                    coalesce(business_name, ''),
                    '[^[:alnum:] ]', ' ', 'g'
                ),
                '\\s+', ' ', 'g'
            )
        ) AS name_norm,

        lower(
            regexp_replace(
                regexp_replace(
                    coalesce(business_address, ''),
                    '[^[:alnum:] ]', ' ', 'g'
                ),
                '\\s+', ' ', 'g'
            )
        ) AS addr_norm,

        country

    FROM read_csv_auto(
        '{TRAIN}/train_source{source[-1]}.tsv',
        delim='\\t',
        header=true,
        ignore_errors=true
    );
    """)


# ============================================================
# 4. TOKEN TABLES
# ============================================================

print("\n[4/7] Building token indexes...")


def build_tokens(source):

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source}_name_tokens AS
    SELECT DISTINCT
        entity_id,
        token
    FROM (
        SELECT
            entity_id,
            unnest(string_split(name_norm, ' ')) AS token
        FROM {source}
    )
    WHERE length(token) >= 3
      AND token != '';
    """)

    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source}_addr_tokens AS
    SELECT DISTINCT
        entity_id,
        token
    FROM (
        SELECT
            entity_id,
            unnest(string_split(addr_norm, ' ')) AS token
        FROM {source}
    )
    WHERE length(token) >= 3
      AND token != '';
    """)

    # --------------------------------------------------------
    # ADDRESS NUMBERS
    # --------------------------------------------------------

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source}_numbers AS
    SELECT DISTINCT
        entity_id,
        token
    FROM (
        SELECT
            entity_id,
            unnest(string_split(addr_norm, ' ')) AS token
        FROM {source}
    )
    WHERE regexp_matches(token, '^[0-9]+$');
    """)


build_tokens("s1")
build_tokens("s2")
build_tokens("s3")


# ============================================================
# 5. TOKEN FREQUENCIES
# ============================================================

print("\n[5/7] Computing token rarity...")

for source in ["s2", "s3"]:

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source}_rare_name AS
    SELECT
        token,
        COUNT(*) AS freq
    FROM {source}_name_tokens
    GROUP BY token
    HAVING COUNT(*) BETWEEN 2 AND 100;
    """)

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE {source}_rare_addr AS
    SELECT
        token,
        COUNT(*) AS freq
    FROM {source}_addr_tokens
    GROUP BY token
    HAVING COUNT(*) BETWEEN 2 AND 100;
    """)


# ============================================================
# BLOCK STORAGE
# ============================================================

blocks = []


def run_block(name, sql):

    start = time.time()

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE current_block AS
    {sql}
    """)

    candidates = DB.execute("""
        SELECT COUNT(*) FROM current_block
    """).fetchone()[0]

    true_retained = DB.execute("""
        SELECT COUNT(*)
        FROM current_block c
        JOIN true_pairs t
          ON c.s1_id = t.s1_id
         AND c.candidate_id = t.matched_entity_id
    """).fetchone()[0]

    recall = true_retained / true_count

    print(
        f"{name:<42}"
        f" candidates={candidates:>12,}"
        f" true={true_retained:>8,}"
        f" recall={recall:>8.3%}"
        f" time={time.time()-start:.1f}s"
    )

    blocks.append(name)

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE block_{len(blocks)} AS
    SELECT * FROM current_block;
    """)


# ============================================================
# S2 BLOCKS
# ============================================================

print("\n[6/7] Running advanced blocks...")
print("\nS2")
print("-" * 100)

# 1. Exact normalized name
run_block(
    "S2 exact name",
    """
    SELECT
        a.entity_id AS s1_id,
        b.entity_id AS candidate_id
    FROM s1 a
    JOIN s2 b
      ON a.name_norm = b.name_norm
    WHERE a.name_norm != ''
    """
)

# 2. Rare name token + address number
run_block(
    "S2 rare name + number",
    """
    SELECT DISTINCT
        a.entity_id AS s1_id,
        nt.entity_id AS candidate_id
    FROM s1 a
    JOIN s1_name_tokens a_name
      ON a.entity_id = a_name.entity_id
    JOIN s1_numbers a_num
      ON a.entity_id = a_num.entity_id
    JOIN s2_name_tokens nt
      ON nt.token = a_name.token
    JOIN s2_rare_name rn
      ON rn.token = nt.token
    JOIN s2_numbers num
      ON num.entity_id = nt.entity_id
    WHERE num.token = a_num.token
    """
)

# 3. Rare name + rare address
run_block(
    "S2 rare name + rare address",
    """
    SELECT DISTINCT
        a.entity_id AS s1_id,
        nt.entity_id AS candidate_id
    FROM s1 a
    JOIN s1_name_tokens a_name
      ON a.entity_id = a_name.entity_id
    JOIN s2_name_tokens nt
      ON nt.token = a_name.token
    JOIN s2_rare_name rn
      ON rn.token = nt.token
    JOIN s2_addr_tokens at
      ON at.entity_id = nt.entity_id
    JOIN s2_rare_addr ra
      ON ra.token = at.token
    JOIN s1_addr_tokens a_addr
      ON a.entity_id = a_addr.entity_id
     AND a_addr.token = at.token
    """
)

# 4. Two name tokens
run_block(
    "S2 two name tokens",
    """
    SELECT DISTINCT
        a1.entity_id AS s1_id,
        x.entity_id AS candidate_id
    FROM s1_name_tokens a1
    JOIN s1_name_tokens a2
      ON a1.entity_id = a2.entity_id
     AND a1.token < a2.token
    JOIN s2_name_tokens x
      ON x.token = a1.token
    JOIN s2_name_tokens y
      ON y.entity_id = x.entity_id
     AND y.token = a2.token
    """
)


# ============================================================
# S3
# ============================================================

print("\nS3")
print("-" * 100)

run_block(
    "S3 exact name",
    """
    SELECT
        a.entity_id AS s1_id,
        b.entity_id AS candidate_id
    FROM s1 a
    JOIN s3 b
      ON a.name_norm = b.name_norm
    WHERE a.name_norm != ''
    """
)

run_block(
    "S3 rare name + number",
    """
    SELECT DISTINCT
        a.entity_id AS s1_id,
        nt.entity_id AS candidate_id
    FROM s1 a
    JOIN s1_name_tokens a_name
      ON a.entity_id = a_name.entity_id
    JOIN s1_numbers a_num
      ON a.entity_id = a_num.entity_id
    JOIN s3_name_tokens nt
      ON nt.token = a_name.token
    JOIN s3_rare_name rn
      ON rn.token = nt.token
    JOIN s3_numbers num
      ON num.entity_id = nt.entity_id
    WHERE num.token = a_num.token
    """
)

run_block(
    "S3 rare name + rare address",
    """
    SELECT DISTINCT
        a.entity_id AS s1_id,
        nt.entity_id AS candidate_id
    FROM s1 a
    JOIN s1_name_tokens a_name
      ON a.entity_id = a_name.entity_id
    JOIN s3_name_tokens nt
      ON nt.token = a_name.token
    JOIN s3_rare_name rn
      ON rn.token = nt.token
    JOIN s3_addr_tokens at
      ON at.entity_id = nt.entity_id
    JOIN s3_rare_addr ra
      ON ra.token = at.token
    JOIN s1_addr_tokens a_addr
      ON a.entity_id = a_addr.entity_id
     AND a_addr.token = at.token
    """
)

run_block(
    "S3 two name tokens",
    """
    SELECT DISTINCT
        a1.entity_id AS s1_id,
        x.entity_id AS candidate_id
    FROM s1_name_tokens a1
    JOIN s1_name_tokens a2
      ON a1.entity_id = a2.entity_id
     AND a1.token < a2.token
    JOIN s3_name_tokens x
      ON x.token = a1.token
    JOIN s3_name_tokens y
      ON y.entity_id = x.entity_id
     AND y.token = a2.token
    """
)


# ============================================================
# UNION
# ============================================================

print("\n[7/7] UNION RESULTS")
print("=" * 100)

union_sql = "\nUNION\n".join(
    f"SELECT * FROM block_{i}"
    for i in range(1, len(blocks) + 1)
)

DB.execute(f"""
CREATE OR REPLACE TEMP TABLE all_candidates AS
{union_sql};
""")

total_candidates = DB.execute("""
SELECT COUNT(*) FROM all_candidates
""").fetchone()[0]

true_retained = DB.execute("""
SELECT COUNT(*)
FROM all_candidates c
JOIN true_pairs t
  ON c.s1_id = t.s1_id
 AND c.candidate_id = t.matched_entity_id
""").fetchone()[0]

recall = true_retained / true_count

avg_candidates = total_candidates / s1_count

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

print(f"\nTotal unique candidates : {total_candidates:,}")
print(f"True pairs retained     : {true_retained:,}")
print(f"UNION recall            : {recall:.4%}")
print(f"Average candidates/S1   : {avg_candidates:,.1f}")
print(f"P95 candidates/S1       : {p95:,.0f}")
print(f"Max candidates/S1      : {max_candidates:,}")

print("\nDONE")