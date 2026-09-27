"""Phase 6 — threshold optimization for macro F_0.5.

Scores every VAL-split candidate pair (held out entirely from src.train) with
the trained model, then sweeps a global probability threshold to find the
macro-F0.5-optimal cutoff — never accuracy, never a hardcoded 0.5 (see
CLAUDE.md section 19). Macro F0.5 is averaged over EVERY val S1 entity,
including:

* singletons (no true match) — scored 1.0 if we correctly predict empty,
  0.0 if we predict anything;
* entities blocking never produced a candidate for at all — always an empty
  prediction, contributes a fixed (not threshold-dependent) term.

A hard cap of 11 matches per S1 (the observed ground-truth max) is applied
on top of the threshold, keeping only the highest-probability candidates
beyond that count.

Run:
    python3 -m src.threshold
"""

import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.train import FEATURE_COLS

MAX_MATCHES = 11
THRESHOLDS = np.round(np.concatenate([
    np.arange(0.05, 0.90, 0.05),
    np.arange(0.90, 0.999, 0.005),
]), 4)


def load_val_scored(model):
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")

    feat_path = config.FEATURES / "train_features.parquet"
    gt_path = config.PROCESSED / "train_gt_split.parquet"

    con.execute(f"""
        CREATE OR REPLACE VIEW true_pairs AS
        SELECT source1_entity_id AS s1_id,
               unnest(string_split(matched_entity_ids, ',')) AS candidate_id
        FROM read_parquet('{gt_path}')
        WHERE matched_entity_ids != ''
    """)
    val_df = con.execute(f"""
        SELECT f.*, CASE WHEN tp.s1_id IS NOT NULL THEN 1 ELSE 0 END AS label
        FROM read_parquet('{feat_path}') f
        JOIN read_parquet('{gt_path}') g ON g.source1_entity_id = f.s1_id AND g.split = 'val'
        LEFT JOIN true_pairs tp ON tp.s1_id = f.s1_id AND tp.candidate_id = f.candidate_id
    """).df()

    all_val = con.execute(f"""
        SELECT source1_entity_id AS s1_id,
               CASE WHEN matched_entity_ids = '' THEN 0
                    ELSE length(matched_entity_ids) - length(replace(matched_entity_ids, ',', '')) + 1
               END AS true_count
        FROM read_parquet('{gt_path}')
        WHERE split = 'val'
    """).df()
    all_val = all_val.set_index("s1_id")["true_count"]

    con.close()

    X = val_df[FEATURE_COLS].astype(np.float32)
    val_df["prob"] = model.predict(X)
    return val_df[["s1_id", "candidate_id", "label", "prob"]], all_val


def f_beta(tp, pred_count, true_count, beta=0.5):
    beta2 = beta ** 2
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp, dtype=np.float64), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp, dtype=np.float64), where=true_count > 0)
    denom = beta2 * precision + recall
    f = np.divide(
        (1 + beta2) * precision * recall, denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    f = np.where((pred_count == 0) & (true_count == 0), 1.0, f)
    return f


def sweep(val_df: pd.DataFrame, true_counts: pd.Series, max_matches=MAX_MATCHES, thresholds=THRESHOLDS):
    all_ids = true_counts.index
    n_val = len(all_ids)

    # Rank per-S1 by probability once; a global threshold + top-K cap just
    # changes how far down each entity's own ranked list we cut.
    val_df = val_df.sort_values(["s1_id", "prob"], ascending=[True, False])
    rank = val_df.groupby("s1_id").cumcount()
    val_df = val_df[rank < max_matches]  # top-11 cap applied unconditionally

    results = []
    for t in thresholds:
        pred = (val_df["prob"].values >= t)
        tp = pred & (val_df["label"].values == 1)
        g = pd.DataFrame({"s1_id": val_df["s1_id"].values, "tp": tp, "pred": pred}).groupby("s1_id").sum()
        g = g.reindex(all_ids, fill_value=0)
        f = f_beta(g["tp"].values.astype(np.float64), g["pred"].values.astype(np.float64), true_counts.values.astype(np.float64))
        macro_f = f.sum() / n_val
        results.append((t, macro_f))

    return results


def optimize():
    t0 = time.time()
    print("[threshold] loading model...")
    model = lgb.Booster(model_file=str(config.MODELS / "lgbm_matcher.txt"))

    print("[threshold] scoring val candidates...")
    val_df, true_counts = load_val_scored(model)
    print(f"[threshold] val candidates: {len(val_df):,}  val S1 entities: {len(true_counts):,}"
          f"  ({time.time()-t0:.1f}s elapsed)")

    print("[threshold] sweeping thresholds...")
    results = sweep(val_df, true_counts)
    for t, f in results:
        print(f"  threshold={t:.2f}  macro_F0.5={f:.5f}")

    best_t, best_f = max(results, key=lambda x: x[1])
    print(f"\n[threshold] BEST threshold={best_t:.3f}  macro_F0.5={best_f:.5f}  ({time.time()-t0:.1f}s total)")

    with open(config.MODELS / "threshold.txt", "w") as fh:
        fh.write(f"{best_t}\n")

    report_path = config.EXPERIMENTS / "threshold_sweep.csv"
    pd.DataFrame(results, columns=["threshold", "macro_f0.5"]).to_csv(report_path, index=False)
    print(f"[threshold] sweep saved -> {report_path}")

    return best_t, best_f


if __name__ == "__main__":
    optimize()
