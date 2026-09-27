"""Phase 8 — final prediction.

Scores every TEST candidate pair with the trained model, applies the
F0.5-optimized threshold + top-11 cap from src.threshold, and writes both
required submission files:

* output/candidate_pairs.tsv  — every blocked candidate (pre-threshold),
  one row per test S1 entity (empty list allowed).
* output/matching_results.tsv — final thresholded matches, same shape.

Every test S1 entity gets exactly one row in both files, including entities
blocking found zero candidates for (empty match list — never forced).

Scoring is chunked Python (LightGBM predict); list-building (the part that
previously took 58 minutes via per-chunk pandas dict accumulation across
many small groups) is now a single DuckDB grouped string_agg pass over the
fully-scored table — DuckDB's grouped aggregation is drastically faster than
a Python-level groupby-and-extend loop repeated per chunk.

Run:
    python3 -m src.predict
"""

import time

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src import config
from src.train import FEATURE_COLS

MAX_MATCHES = 11
CHUNK = 5_000_000


def load_threshold():
    with open(config.MODELS / "threshold.txt") as fh:
        return float(fh.read().strip())


def predict():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=6")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{config.DATA}/duckdb_tmp'")

    print("[predict] loading model + threshold...")
    model = lgb.Booster(model_file=str(config.MODELS / "lgbm_matcher.txt"))
    threshold = load_threshold()
    print(f"[predict] threshold={threshold}")

    feat_path = config.FEATURES / "test_features.parquet"
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{feat_path}')").fetchone()[0]
    print(f"[predict] test candidate pairs: {n_total:,}")

    scored_path = config.PROCESSED / "test_scored.parquet"
    writer = None
    offset = 0
    # NOTE: repeated `LIMIT n OFFSET m` queries on a connection with
    # preserve_insertion_order=false are NOT guaranteed to partition the
    # table exhaustively/disjointly under parallel scans -- rows can be
    # duplicated across chunks (and others silently dropped). Use a single
    # execute() + fetch_df_chunk() cursor instead, which streams the one
    # underlying result set exactly once per row regardless of thread count.
    con.execute(f"SELECT * FROM read_parquet('{feat_path}')")
    VECTORS_PER_CHUNK = max(1, CHUNK // 2048)
    while True:
        chunk = con.fetch_df_chunk(VECTORS_PER_CHUNK)
        if chunk.empty:
            break

        X = chunk[FEATURE_COLS].astype(np.float32)
        prob = model.predict(X).astype(np.float32)

        out = chunk[["s1_id", "candidate_id"]].copy()
        out["prob"] = prob
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(scored_path), table.schema)
        writer.write_table(table)

        offset += len(chunk)
        print(f"  scored {offset:,}/{n_total:,}  ({time.time()-t0:.1f}s elapsed)")

    if writer is not None:
        writer.close()

    n_scored = con.execute(f"SELECT COUNT(*) FROM read_parquet('{scored_path}')").fetchone()[0]
    n_distinct_scored = con.execute(
        f"SELECT COUNT(DISTINCT (s1_id, candidate_id)) FROM read_parquet('{scored_path}')"
    ).fetchone()[0]
    assert n_scored == n_total == n_distinct_scored, (
        f"scored row count mismatch: scored={n_scored:,} distinct={n_distinct_scored:,} "
        f"expected={n_total:,}"
    )
    print(f"[predict] scoring done ({time.time()-t0:.1f}s elapsed) -> building TSVs via DuckDB...")

    s1_path = config.processed_path("test", "s1")
    config.OUTPUT.mkdir(parents=True, exist_ok=True)

    # candidate_pairs.tsv: every candidate, no threshold applied.
    con.execute(f"""
        COPY (
            SELECT s1.entity_id AS source1_entity_id,
                   COALESCE(string_agg(sc.candidate_id, ','), '') AS candidate_entity_ids
            FROM read_parquet('{s1_path}') s1
            LEFT JOIN read_parquet('{scored_path}') sc ON sc.s1_id = s1.entity_id
            GROUP BY s1.entity_id
        ) TO '{config.OUTPUT / "candidate_pairs.tsv"}'
        (FORMAT CSV, DELIMITER '\t', HEADER TRUE, QUOTE '')
    """)
    print(f"[predict] candidate_pairs.tsv written ({time.time()-t0:.1f}s elapsed)")

    # matching_results.tsv: top-11-by-prob, threshold-filtered.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ranked AS
        SELECT s1_id, candidate_id, prob,
               ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY prob DESC) AS rnk
        FROM read_parquet('{scored_path}')
    """)
    con.execute(f"""
        COPY (
            SELECT s1.entity_id AS source1_entity_id,
                   COALESCE(string_agg(r.candidate_id, ','), '') AS matched_entity_ids
            FROM read_parquet('{s1_path}') s1
            LEFT JOIN ranked r
              ON r.s1_id = s1.entity_id AND r.rnk <= {MAX_MATCHES} AND r.prob >= {threshold}
            GROUP BY s1.entity_id
        ) TO '{config.OUTPUT / "matching_results.tsv"}'
        (FORMAT CSV, DELIMITER '\t', HEADER TRUE, QUOTE '')
    """)
    print(f"[predict] matching_results.tsv written ({time.time()-t0:.1f}s elapsed)")

    n_s1 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{s1_path}')").fetchone()[0]
    n_matched, total_matches = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE matched_entity_ids != ''),
            SUM(CASE WHEN matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END)
        FROM read_csv('{config.OUTPUT / "matching_results.tsv"}', delim='\t', header=true)
    """).fetchone()
    print(f"[predict] S1 entities with >=1 match: {n_matched:,}/{n_s1:,} ({n_matched/n_s1:.2%})")
    print(f"[predict] total matched pairs: {total_matches:,} (mean {total_matches/n_s1:.2f}/S1)")
    print(f"[predict] done ({time.time()-t0:.1f}s total)")

    con.close()
    scored_path.unlink(missing_ok=True)


if __name__ == "__main__":
    predict()
