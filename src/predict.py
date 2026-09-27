"""Phase 8 — final prediction.

Scores every TEST candidate pair with the trained model, applies the
F0.5-optimized threshold + top-11 cap from src.threshold, and writes both
required submission files:

* output/candidate_pairs.tsv  — every blocked candidate (pre-threshold),
  one row per test S1 entity (empty list allowed).
* output/matching_results.tsv — final thresholded matches, same shape.

Every test S1 entity gets exactly one row in both files, including entities
blocking found zero candidates for (empty match list — never forced).

Run:
    python3 -m src.predict
"""

import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.train import FEATURE_COLS

MAX_MATCHES = 11


def load_threshold():
    with open(config.MODELS / "threshold.txt") as fh:
        return float(fh.read().strip())


def write_id_list_tsv(path, s1_to_ids: dict, all_s1_ids, header):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{header[0]}\t{header[1]}\n")
        for s1 in all_s1_ids:
            ids = s1_to_ids.get(s1, [])
            fh.write(f"{s1}\t{','.join(ids)}\n")


def predict():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")

    print("[predict] loading model + threshold...")
    model = lgb.Booster(model_file=str(config.MODELS / "lgbm_matcher.txt"))
    threshold = load_threshold()
    print(f"[predict] threshold={threshold}")

    feat_path = config.FEATURES / "test_features.parquet"
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{feat_path}')").fetchone()[0]
    print(f"[predict] test candidate pairs: {n_total:,}")

    all_s1_ids = con.execute(f"""
        SELECT entity_id FROM read_parquet('{config.processed_path("test", "s1")}')
    """).df()["entity_id"].tolist()
    print(f"[predict] test S1 entities: {len(all_s1_ids):,}")

    candidate_lists = {}
    match_lists = {}

    CHUNK = 5_000_000
    offset = 0
    while offset < n_total:
        chunk = con.execute(f"""
            SELECT * FROM read_parquet('{feat_path}') LIMIT {CHUNK} OFFSET {offset}
        """).df()
        if chunk.empty:
            break

        X = chunk[FEATURE_COLS].astype(np.float32)
        chunk["prob"] = model.predict(X)

        for s1, grp in chunk.groupby("s1_id"):
            candidate_lists.setdefault(s1, []).extend(grp["candidate_id"].tolist())

        chunk = chunk.sort_values(["s1_id", "prob"], ascending=[True, False])
        chunk["rank"] = chunk.groupby("s1_id").cumcount()
        capped = chunk[chunk["rank"] < MAX_MATCHES]
        matched = capped[capped["prob"] >= threshold]
        for s1, grp in matched.groupby("s1_id"):
            match_lists.setdefault(s1, []).extend(grp["candidate_id"].tolist())

        offset += len(chunk)
        print(f"  scored {offset:,}/{n_total:,}  ({time.time()-t0:.1f}s elapsed)")

    con.close()

    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    write_id_list_tsv(
        config.OUTPUT / "candidate_pairs.tsv", candidate_lists, all_s1_ids,
        ("source1_entity_id", "candidate_entity_ids"),
    )
    write_id_list_tsv(
        config.OUTPUT / "matching_results.tsv", match_lists, all_s1_ids,
        ("source1_entity_id", "matched_entity_ids"),
    )

    n_matched = sum(1 for s1 in all_s1_ids if match_lists.get(s1))
    n_matches_total = sum(len(v) for v in match_lists.values())
    print(f"[predict] S1 entities with >=1 match: {n_matched:,}/{len(all_s1_ids):,}"
          f" ({n_matched/len(all_s1_ids):.2%})")
    print(f"[predict] total matched pairs: {n_matches_total:,}"
          f" (mean {n_matches_total/len(all_s1_ids):.2f}/S1)")
    print(f"[predict] done ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    predict()
