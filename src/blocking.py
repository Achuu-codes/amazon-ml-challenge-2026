"""Phase 3 — candidate generation / blocking.

Builds a DuckDB session over the normalized Parquet tables from
``src.normalize`` and runs a set of complementary, equi-join-only blocking
passes (no per-row regex — that does not scale, see analysis/test_union_blocking.py
which never finished on a 20k-row sample). Every pass is measured for
candidate volume and recall against ground truth before being kept.

Two entry points:

* ``measure(...)``  — run every candidate block on the TRAIN split, print
  candidates/recall/avg per S1, so blocking choices are evidence-based.
* ``generate_candidates(split)`` — build the final unioned candidate table
  for "train" (all train S1) or "test", used by features.py / predict.py.
"""

import argparse
import time

import duckdb

from src import config

RARE_MAX = 60          # a token/number touching more source rows than this
                        # is too common to be useful blocking evidence
COMPOUND_RARE_MAX = 50   # cap on the COMBINED (2-field) key's document
                          # frequency. 150 measured 91.8% entity-level recall
                          # at ~92 candidates/S1 (370M pairs train+test) but
                          # that volume made RapidFuzz feature extraction
                          # operationally heavy (multi-hour, memory-fragile
                          # multiprocessing). 50 measured ~70% pair recall at
                          # ~49 candidates/S1 (half the volume) — a simpler,
                          # safer single-process budget. See
                          # logs/blocking_measure.log for both sweeps.
MIN_COMPOUND_TOKEN_LEN = 4  # compound blocks additionally require longer
                             # tokens than the base MIN_TOKEN_LEN=3 — short
                             # tokens combined with a number/word are still
                             # too common to be discriminative evidence
ADDR_WORD_PREFILTER = 1500  # address words touching more source rows than
                             # this are dropped BEFORE the name x addr-word
                             # cross join, not after — this is what makes
                             # the name+address-word compound block (which
                             # previously OOM'd building its unfiltered pair
                             # table) tractable at full scale
COMPOUND_RARE_MAX_ADDRWORD = 20  # tighter cap than COMPOUND_RARE_MAX: even
                                  # after prefiltering, name+addrword keys
                                  # have far more combinations per record
                                  # than name+number, so cap the survivors
                                  # more aggressively
N_UNION_SHARDS = 6  # final cross-block dedup is sharded by hash(s1_id) to
                     # bound memory (see generate_candidates)
EXACT_NAME_MAX = 1000    # cap even "exact name match" buckets: a single
                          # hyper-common business name should not explode


def connect():
    con = duckdb.connect()
    con.execute("SET memory_limit='13GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{config.DATA}/duckdb_tmp'")
    return con


def load_source(con, split, source):
    path = config.processed_path(split, source)
    view = f"{split}_{source}"
    con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{path}')")
    return view


def build_indexes(con, split, source):
    """Unnest token/number list columns into flat (entity_id, token) tables
    and compute each token's document frequency for rarity filtering."""

    base = f"{split}_{source}"

    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_name_tok AS
        SELECT DISTINCT entity_id, unnest(name_tokens) AS token FROM {base}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_translit_tok AS
        SELECT DISTINCT entity_id, unnest(name_translit_tokens) AS token FROM {base}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_addr_tok AS
        SELECT DISTINCT entity_id, unnest(addr_tokens) AS token FROM {base}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_num AS
        SELECT DISTINCT entity_id, unnest(addr_numbers) AS token FROM {base}
    """)

    for tbl in ["name_tok", "translit_tok", "addr_tok", "num"]:
        con.execute(f"""
            CREATE OR REPLACE TABLE {base}_{tbl}_freq AS
            SELECT token, COUNT(*) AS freq FROM {base}_{tbl} GROUP BY token
        """)
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{base}_{tbl} ON {base}_{tbl}(token)")

    # Compound (name-token, address-number) key, precomputed and
    # frequency-capped BEFORE any join — computing the join first and
    # filtering after (e.g. with QUALIFY) materializes the full, often huge,
    # cross product first and reliably runs out of memory at full scale.
    #
    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_name_num_pairs AS
        SELECT DISTINCT t.entity_id, t.token || '|' || n.token AS key
        FROM {base}_name_tok t
        JOIN {base}_num n ON n.entity_id = t.entity_id
        WHERE length(t.token) >= {MIN_COMPOUND_TOKEN_LEN}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_name_num_freq AS
        SELECT key, COUNT(*) AS freq FROM {base}_name_num_pairs GROUP BY key
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{base}_name_num ON {base}_name_num_pairs(key)")

    # name-token + address-WORD compound. Earlier attempt blew memory
    # building the full (unfiltered) cross product before any capping — the
    # fix is to pre-filter address words to reasonably rare ones (doc-freq
    # <= ADDR_WORD_PREFILTER) BEFORE the cross join, shrinking the fanout
    # from the start rather than trying to shrink it after.
    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_addr_word_rare AS
        SELECT a.entity_id, a.token
        FROM {base}_addr_tok a
        JOIN {base}_addr_tok_freq f ON f.token = a.token AND f.freq <= {ADDR_WORD_PREFILTER}
        WHERE length(a.token) >= {MIN_COMPOUND_TOKEN_LEN}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_name_addrword_pairs AS
        SELECT DISTINCT t.entity_id, t.token || '|' || w.token AS key
        FROM {base}_name_tok t
        JOIN {base}_addr_word_rare w ON w.entity_id = t.entity_id
        WHERE length(t.token) >= {MIN_COMPOUND_TOKEN_LEN}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {base}_name_addrword_freq AS
        SELECT key, COUNT(*) AS freq FROM {base}_name_addrword_pairs GROUP BY key
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{base}_name_addrword ON {base}_name_addrword_pairs(key)")
    con.execute(f"DROP TABLE {base}_addr_word_rare")


def prepare(con, split, s1_only_split=None):
    """Load & index s1/s2/s3 for `split` ('train' or 'test').

    `s1_only_split` ('train' | 'val'), only meaningful when split == 'train',
    restricts the s1 side to that half of the deterministic S1 split — used
    by `measure()` so blocking experiments only pay for the S1 rows they'll
    actually evaluate recall on, instead of all 2.2M train S1 every time.
    S2/S3 (the corpus being searched) are never restricted.
    """
    gt_path = config.PROCESSED / "train_gt_split.parquet"

    for source in ["s1", "s2", "s3"]:
        if source == "s1" and s1_only_split:
            path = config.processed_path(split, source)
            con.execute(f"""
                CREATE OR REPLACE VIEW {split}_s1 AS
                SELECT s1.* FROM read_parquet('{path}') s1
                JOIN read_parquet('{gt_path}') g
                  ON g.source1_entity_id = s1.entity_id
                WHERE g.split = '{s1_only_split}'
            """)
        else:
            load_source(con, split, source)
        build_indexes(con, split, source)

    if split == "train":
        con.execute(f"""
            CREATE OR REPLACE VIEW gt AS
            SELECT source1_entity_id, split,
                   unnest(string_split(matched_entity_ids, ',')) AS matched_entity_id
            FROM read_parquet('{gt_path}')
            WHERE matched_entity_ids != ''
        """)
        con.execute("""
            CREATE OR REPLACE TABLE true_pairs AS
            SELECT source1_entity_id AS s1_id, matched_entity_id AS candidate_id, split
            FROM gt
        """)


# ---------------------------------------------------------------------------
# Block definitions. Each returns SQL producing (s1_id, candidate_id) for one
# source ("s2" or "s3") against the "train"/"test" s1 table.
# ---------------------------------------------------------------------------

def block_exact_name(split, source):
    return f"""
        SELECT a.entity_id AS s1_id, b.entity_id AS candidate_id
        FROM {split}_s1 a
        JOIN {split}_{source} b ON a.name_norm = b.name_norm
        WHERE a.name_norm != ''
        QUALIFY COUNT(*) OVER (PARTITION BY a.name_norm) <= {EXACT_NAME_MAX}
    """


def block_exact_translit(split, source):
    return f"""
        SELECT a.entity_id AS s1_id, b.entity_id AS candidate_id
        FROM {split}_s1 a
        JOIN {split}_{source} b ON a.name_translit = b.name_translit
        WHERE a.name_translit != ''
        QUALIFY COUNT(*) OVER (PARTITION BY a.name_translit) <= {EXACT_NAME_MAX}
    """


def block_rare_name_token(split, source):
    base = f"{split}_{source}"
    return f"""
        SELECT DISTINCT a_tok.entity_id AS s1_id, b_tok.entity_id AS candidate_id
        FROM {split}_s1_name_tok a_tok
        JOIN {base}_name_tok b_tok ON a_tok.token = b_tok.token
        JOIN {base}_name_tok_freq f ON f.token = b_tok.token
        WHERE f.freq <= {RARE_MAX}
    """


def block_rare_translit_token(split, source):
    base = f"{split}_{source}"
    return f"""
        SELECT DISTINCT a_tok.entity_id AS s1_id, b_tok.entity_id AS candidate_id
        FROM {split}_s1_translit_tok a_tok
        JOIN {base}_translit_tok b_tok ON a_tok.token = b_tok.token
        JOIN {base}_translit_tok_freq f ON f.token = b_tok.token
        WHERE f.freq <= {RARE_MAX}
    """


def _compound_block(key_name, cap=COMPOUND_RARE_MAX):
    """A generic compound-key block: join on a precomputed, frequency-capped
    2-field key (see build_indexes). Used for name+number and
    name+addr-word blocks."""

    def block(split, source):
        base = f"{split}_{source}"
        return f"""
            SELECT DISTINCT s1p.entity_id AS s1_id, bp.entity_id AS candidate_id
            FROM {split}_s1_{key_name}_pairs s1p
            JOIN {base}_{key_name}_freq f ON f.key = s1p.key AND f.freq <= {cap}
            JOIN {base}_{key_name}_pairs bp ON bp.key = s1p.key
        """

    return block


block_name_token_plus_number = _compound_block("name_num")
block_name_token_plus_addr_word = _compound_block("name_addrword", cap=COMPOUND_RARE_MAX_ADDRWORD)


BLOCKS = [
    ("exact_name", block_exact_name),
    ("exact_translit", block_exact_translit),
    ("rare_name_token", block_rare_name_token),
    ("rare_translit_token", block_rare_translit_token),
    ("name_token_plus_number", block_name_token_plus_number),
    ("name_token_plus_addr_word", block_name_token_plus_addr_word),
]


def run_block(con, name, sql, s1_count, true_count=None, eval_split=None):
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TABLE block_{name} AS {sql}")
    candidates = con.execute(f"SELECT COUNT(*) FROM block_{name}").fetchone()[0]

    recall_str = ""
    if true_count is not None:
        where = f"AND tp.split = '{eval_split}'" if eval_split else ""
        retained = con.execute(f"""
            SELECT COUNT(*) FROM block_{name} b
            JOIN true_pairs tp ON b.s1_id = tp.s1_id AND b.candidate_id = tp.candidate_id
            {where}
        """).fetchone()[0]
        recall_str = f" recall={retained/true_count:8.3%}" if true_count else ""

    avg = candidates / s1_count if s1_count else 0
    print(f"  {name:<32} candidates={candidates:>13,} avg/S1={avg:>8.2f}{recall_str} time={time.time()-t0:.1f}s")
    return candidates


def measure(eval_split="val"):
    """Run every block on TRAIN, evaluate recall against `eval_split` half
    of the deterministic S1 split (train/val), print a summary table."""

    con = connect()
    print(f"[blocking] preparing train tables (s1 restricted to '{eval_split}')...")
    prepare(con, "train", s1_only_split=eval_split)

    s1_count = con.execute("SELECT COUNT(*) FROM train_s1").fetchone()[0]

    true_count = con.execute(f"""
        SELECT COUNT(*) FROM true_pairs WHERE split = '{eval_split}'
    """).fetchone()[0]
    print(f"S1 in {eval_split}: {s1_count:,}  true pairs: {true_count:,}\n")

    block_names = []
    for source in ["s2", "s3"]:
        print(f"--- {source.upper()} ---")
        for name, fn in BLOCKS:
            full_name = f"{name}_{source}"
            run_block(con, full_name, fn("train", source), s1_count, true_count, eval_split)
            block_names.append(full_name)
        print()

    print("--- UNION (incremental) ---")
    cumulative = None
    for bn in block_names:
        if cumulative is None:
            con.execute(f"CREATE OR REPLACE TABLE cum AS SELECT * FROM block_{bn}")
        else:
            con.execute(f"CREATE OR REPLACE TABLE cum AS SELECT * FROM cum UNION SELECT * FROM block_{bn}")
        cumulative = True
        candidates = con.execute("SELECT COUNT(*) FROM cum").fetchone()[0]
        retained = con.execute(f"""
            SELECT COUNT(*) FROM cum c
            JOIN true_pairs tp ON c.s1_id = tp.s1_id AND c.candidate_id = tp.candidate_id
            WHERE tp.split = '{eval_split}'
        """).fetchone()[0]
        print(f"  +{bn:<32} total_candidates={candidates:>13,} recall={retained/true_count:8.3%} avg/S1={candidates/s1_count:.2f}")

    p95, mx = con.execute("""
        SELECT quantile_cont(cnt, 0.95), MAX(cnt) FROM (
            SELECT s1_id, COUNT(*) AS cnt FROM cum GROUP BY s1_id
        )
    """).fetchone()
    print(f"\nFinal union: P95 candidates/S1={p95:.0f} max={mx:,}")

    # Entity-level recall: of S1 entities that HAVE a true match, what
    # fraction have AT LEAST ONE of their true matches retained? This is
    # what actually bounds macro F0.5 (computed per-entity), unlike the
    # pair-level recall above which under-counts partial credit on
    # multi-match entities.
    entity_recall = con.execute(f"""
        WITH matched_s1 AS (
            SELECT DISTINCT s1_id FROM true_pairs WHERE split = '{eval_split}'
        ),
        hit_s1 AS (
            SELECT DISTINCT tp.s1_id
            FROM true_pairs tp
            JOIN cum c ON c.s1_id = tp.s1_id AND c.candidate_id = tp.candidate_id
            WHERE tp.split = '{eval_split}'
        )
        SELECT (SELECT COUNT(*) FROM hit_s1), (SELECT COUNT(*) FROM matched_s1)
    """).fetchone()
    hit, matched = entity_recall
    print(f"Entity-level recall (>=1 true match retained): {hit:,}/{matched:,} = {hit/matched:.3%}")
    con.close()


def generate_candidates(split):
    """Build the final candidate_pairs table for `split` ('train' or 'test')
    and write it to parquet."""

    con = connect()
    print(f"[blocking] preparing {split} tables...")
    prepare(con, split)

    s1_count = con.execute(f"SELECT COUNT(*) FROM {split}_s1").fetchone()[0]
    true_count = None
    if split == "train":
        true_count = con.execute("SELECT COUNT(*) FROM true_pairs").fetchone()[0]

    block_names = []
    for source in ["s2", "s3"]:
        print(f"--- {source.upper()} ---")
        for name, fn in BLOCKS:
            full_name = f"{name}_{source}"
            run_block(con, full_name, fn(split, source), s1_count, true_count)
            block_names.append(full_name)

    print("--- UNION (sharded dedup) ---")
    # A single global UNION (implies DISTINCT) over ~100M+ raw rows reliably
    # exceeds the memory budget — same fix as the negative-sampling window
    # function: shard by hash(s1_id) so each dedup pass only ever sees
    # ~1/N_UNION_SHARDS of the rows.
    con.execute("DROP TABLE IF EXISTS all_candidates")
    for k in range(N_UNION_SHARDS):
        shard_sql = "\nUNION\n".join(
            f"SELECT * FROM block_{bn} WHERE abs(hash(s1_id)) % {N_UNION_SHARDS} = {k}"
            for bn in block_names
        )
        con.execute(f"CREATE OR REPLACE TEMP TABLE union_shard AS {shard_sql}")
        if k == 0:
            con.execute("CREATE TABLE all_candidates AS SELECT * FROM union_shard")
        else:
            con.execute("INSERT INTO all_candidates SELECT * FROM union_shard")
        con.execute("DROP TABLE union_shard")
    print(f"  union done across {N_UNION_SHARDS} shards")

    total = con.execute("SELECT COUNT(*) FROM all_candidates").fetchone()[0]
    print(f"Total candidate pairs: {total:,}  avg/S1={total/s1_count:.2f}")

    if true_count:
        retained = con.execute("""
            SELECT COUNT(*) FROM all_candidates c
            JOIN true_pairs tp ON c.s1_id = tp.s1_id AND c.candidate_id = tp.candidate_id
        """).fetchone()[0]
        print(f"Recall ceiling: {retained/true_count:.4%}")

    dest = config.PROCESSED / f"{split}_candidates.parquet"
    con.execute(f"COPY all_candidates TO '{dest}' (FORMAT PARQUET)")
    print(f"-> {dest}")
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["measure", "generate"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    if args.mode == "measure":
        measure()
    else:
        generate_candidates(args.split)
