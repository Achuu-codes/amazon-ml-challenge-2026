"""Phase 9 (validation half) — final validation report + error analysis.

Re-scores the VAL-split candidates at the chosen threshold and reports:
* macro F0.5 overall, and split by country and by singleton-vs-matched —
  the country breakdown is the closest proxy we have for how well the
  method is likely to generalize to test's France entities (train only has
  US/India; France doesn't exist in any training signal, so this is the
  best evidence available, not a guarantee).
* sampled false positives / false negatives / low-confidence true positives
  / high-confidence false positives, with original text, for the
  methodology writeup's error-analysis section.

Run:
    python3 -m src.evaluate
"""

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.threshold import MAX_MATCHES, f_beta, load_val_scored

N_EXAMPLES = 25


def cap_and_flag(val_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = val_df.sort_values(["s1_id", "prob"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("s1_id").cumcount()
    df = df[df["rank"] < MAX_MATCHES]
    df["pred"] = df["prob"] >= threshold
    return df


def per_entity_scores(df: pd.DataFrame, true_counts: pd.Series) -> pd.DataFrame:
    g = df.groupby("s1_id").apply(
        lambda x: pd.Series({
            "tp": int((x["pred"] & (x["label"] == 1)).sum()),
            "pred_count": int(x["pred"].sum()),
        }),
        include_groups=False,
    )
    g = g.reindex(true_counts.index, fill_value=0)
    g["true_count"] = true_counts
    g["f0.5"] = f_beta(g["tp"].values.astype(float), g["pred_count"].values.astype(float),
                        g["true_count"].values.astype(float))
    return g


def main():
    print("[evaluate] loading model + threshold...")
    model = lgb.Booster(model_file=str(config.MODELS / "lgbm_matcher.txt"))
    with open(config.MODELS / "threshold.txt") as fh:
        threshold = float(fh.read().strip())

    val_df, true_counts = load_val_scored(model)
    df = cap_and_flag(val_df, threshold)
    scores = per_entity_scores(df, true_counts)

    con = duckdb.connect()
    s1_path = config.processed_path("train", "s1")
    country = con.execute(f"SELECT entity_id AS s1_id, country FROM read_parquet('{s1_path}')").df()
    country = country.set_index("s1_id")["country"]
    scores["country"] = country.reindex(scores.index)
    scores["is_singleton"] = scores["true_count"] == 0

    print("\n=== VALIDATION SUMMARY (macro F0.5) ===")
    print(f"Overall:              {scores['f0.5'].mean():.5f}  (n={len(scores):,})")
    print("\nBy country:")
    print(scores.groupby("country")["f0.5"].agg(["mean", "count"]).to_string())
    print("\nBy singleton vs matched:")
    print(scores.groupby("is_singleton")["f0.5"].agg(["mean", "count"]).to_string())

    summary_path = config.EXPERIMENTS / "validation_summary.csv"
    scores.to_csv(summary_path)
    print(f"\n[evaluate] per-entity scores -> {summary_path}")

    # ---- error analysis samples -------------------------------------
    s2_path = config.processed_path("train", "s2")
    s3_path = config.processed_path("train", "s3")

    def attach_text(pairs: pd.DataFrame) -> pd.DataFrame:
        s1_text = con.execute(f"""
            SELECT entity_id AS s1_id, name_norm AS s1_name, addr_norm AS s1_addr, country AS s1_country
            FROM read_parquet('{s1_path}')
        """).df()
        s2_text = con.execute(f"SELECT entity_id AS candidate_id, name_norm AS cand_name, addr_norm AS cand_addr FROM read_parquet('{s2_path}')").df()
        s3_text = con.execute(f"SELECT entity_id AS candidate_id, name_norm AS cand_name, addr_norm AS cand_addr FROM read_parquet('{s3_path}')").df()
        cand_text = pd.concat([s2_text, s3_text], ignore_index=True)
        out = pairs.merge(s1_text, on="s1_id", how="left").merge(cand_text, on="candidate_id", how="left")
        return out

    false_pos = df[df["pred"] & (df["label"] == 0)].sort_values("prob", ascending=False).head(N_EXAMPLES)
    false_neg_scored = df[(~df["pred"]) & (df["label"] == 1)].sort_values("prob", ascending=False).head(N_EXAMPLES)
    low_conf_tp = df[df["pred"] & (df["label"] == 1)].sort_values("prob").head(N_EXAMPLES)
    high_conf_fp = false_pos.head(N_EXAMPLES)

    for name, sample in [
        ("false_positives", false_pos),
        ("false_negatives_scored", false_neg_scored),
        ("low_confidence_true_positives", low_conf_tp),
        ("high_confidence_false_positives", high_conf_fp),
    ]:
        if sample.empty:
            continue
        enriched = attach_text(sample[["s1_id", "candidate_id", "label", "prob"]])
        dest = config.EXPERIMENTS / f"error_{name}.csv"
        enriched.to_csv(dest, index=False)
        print(f"[evaluate] {name}: {len(enriched)} examples -> {dest}")

    # entities with true matches entirely absent from candidates (blocking misses)
    candidate_s1_ids = set(val_df["s1_id"].unique())
    fully_missed = true_counts[(true_counts > 0) & (~true_counts.index.isin(candidate_s1_ids))]
    print(f"\n[evaluate] val S1 entities with true matches but ZERO candidates: {len(fully_missed):,}"
          f" / {(true_counts > 0).sum():,} matched entities")

    con.close()


if __name__ == "__main__":
    main()
