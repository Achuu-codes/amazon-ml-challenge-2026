"""Phase 5 — supervised matching model.

Builds a labeled training set from the blocked+featurized train candidates:
every retained true-positive pair, plus per-S1 hard-negative sampling (the
highest-similarity WRONG candidates in the same blocking bucket — the pairs
that actually look like plausible false merges, which is what a
precision-heavy F0.5 metric punishes hardest). Trains a LightGBM binary
classifier (MIT-licensed, well under the 8B-parameter cap) on that set,
holding out a small dev slice of the TRAIN-split S1 entities for early
stopping. The held-out VAL-split S1 entities are never touched here — they
are scored (not trained on) so src.threshold can pick an F0.5-optimal
decision rule on genuinely unseen data.

Run:
    python3 -m src.train
"""

import hashlib
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config

NEG_PER_S1 = 20  # hard negatives kept per S1 (ranked by similarity), not
                  # every blocked non-match — random/easy negatives add
                  # little signal and bloat training time (see CLAUDE.md
                  # section 16: train on TRUE MATCH vs VERY SIMILAR BUT
                  # WRONG MATCH, not TRUE MATCH vs obviously-unrelated).
DEV_FRACTION = 0.1  # slice of the TRAIN-split S1 entities held out for
                     # LightGBM early stopping; VAL-split is reserved
                     # entirely for threshold selection (src.threshold).

FEATURE_COLS = [
    "name_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_translit_ratio",
    "addr_ratio", "addr_token_set_ratio",
    "name_exact", "name_translit_exact", "addr_exact",
    "name_token_jaccard", "name_common_tokens",
    "addr_token_jaccard", "addr_common_tokens",
    "shared_numeric_count", "shared_numeric_bool",
    "country_equal",
    "name_len_diff", "name_len_ratio", "addr_len_diff", "addr_len_ratio",
    "s1_name_missing", "cand_name_missing", "s1_addr_missing", "cand_addr_missing",
    "is_source3",
]


def connect():
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{config.DATA}/duckdb_tmp'")
    return con


def build_labeled_view(con, feat_path, gt_path):
    con.execute(f"""
        CREATE OR REPLACE VIEW true_pairs AS
        SELECT source1_entity_id AS s1_id,
               unnest(string_split(matched_entity_ids, ',')) AS candidate_id
        FROM read_parquet('{gt_path}')
        WHERE matched_entity_ids != ''
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW labeled AS
        SELECT f.*, g.split,
               CASE WHEN tp.s1_id IS NOT NULL THEN 1 ELSE 0 END AS label
        FROM read_parquet('{feat_path}') f
        JOIN read_parquet('{gt_path}') g ON g.source1_entity_id = f.s1_id
        LEFT JOIN true_pairs tp ON tp.s1_id = f.s1_id AND tp.candidate_id = f.candidate_id
    """)


def extract_train_pos_neg(con, gt_path):
    con.execute("CREATE OR REPLACE TABLE train_pos AS SELECT * FROM labeled WHERE split='train' AND label=1")
    n_pos = con.execute("SELECT COUNT(*) FROM train_pos").fetchone()[0]
    n_train_s1 = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{gt_path}') WHERE split='train'
    """).fetchone()[0]

    # A per-S1 ROW_NUMBER()-ranked top-K (exact hardest negatives) reliably
    # hits DuckDB's memory ceiling at this scale even with disk-spill
    # configured (~80M rows partitioned+sorted). Instead: bias toward hard
    # negatives with a cheap similarity floor (blocking already guarantees
    # some token/number overlap, so even "easy" blocked negatives are harder
    # than random pairs), then randomly subsample to a target ratio — a
    # filter + random() is a single streaming pass, no large sort buffer.
    target_total = n_train_s1 * NEG_PER_S1
    n_candidates = con.execute("""
        SELECT COUNT(*) FROM labeled
        WHERE split='train' AND label=0
          AND (name_ratio >= 40 OR addr_ratio >= 40 OR shared_numeric_bool = 1)
    """).fetchone()[0]
    sample_rate = min(1.0, target_total / max(n_candidates, 1))
    print(f"[train] hard-negative pool={n_candidates:,} target={target_total:,} sample_rate={sample_rate:.4f}")

    con.execute(f"""
        CREATE OR REPLACE TABLE train_neg AS
        SELECT * FROM labeled
        WHERE split='train' AND label=0
          AND (name_ratio >= 40 OR addr_ratio >= 40 OR shared_numeric_bool = 1)
          AND random() < {sample_rate}
    """)
    n_neg = con.execute("SELECT COUNT(*) FROM train_neg").fetchone()[0]
    print(f"[train] train-split positives={n_pos:,} hard-negatives={n_neg:,} ratio=1:{n_neg/n_pos:.1f}")
    df = con.execute("SELECT * FROM train_pos UNION ALL SELECT * FROM train_neg").df()
    return df


def dev_split_mask(s1_ids: pd.Series) -> np.ndarray:
    """Deterministic sub-split of TRAIN-split S1 ids into fit/dev.

    Salts the id before hashing (independent of src.split's train/val hash)
    — reusing the same hash with a smaller cutoff would carve out nothing,
    since every TRAIN-split id already has hash_frac >= VAL_FRACTION.
    """
    def frac(entity_id):
        h = hashlib.md5((entity_id + "::dev").encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    return s1_ids.map(frac).values < DEV_FRACTION


def train_model():
    t0 = time.time()
    con = connect()

    feat_path = config.FEATURES / "train_features.parquet"
    gt_path = config.PROCESSED / "train_gt_split.parquet"
    build_labeled_view(con, feat_path, gt_path)

    df = extract_train_pos_neg(con, gt_path)
    print(f"[train] training frame: {len(df):,} rows ({time.time()-t0:.1f}s elapsed)")

    is_dev = dev_split_mask(df["s1_id"])
    X = df[FEATURE_COLS].astype(np.float32)
    y = df["label"].astype(np.int32)

    X_fit, y_fit = X[~is_dev], y[~is_dev]
    X_dev, y_dev = X[is_dev], y[is_dev]
    print(f"[train] fit={len(X_fit):,} dev={len(X_dev):,} pos_rate_fit={y_fit.mean():.4f}")

    pos_weight = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=config.SEED,
        n_jobs=6,
    )
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_dev, y_dev)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n[train] feature importance (gain-normalized split count):")
    print(importance.to_string())

    model_path = config.MODELS / "lgbm_matcher.txt"
    model.booster_.save_model(str(model_path))
    print(f"\n[train] saved -> {model_path} ({time.time()-t0:.1f}s total)")

    con.close()
    return model


if __name__ == "__main__":
    train_model()
