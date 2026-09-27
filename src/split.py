"""Phase 2 (part) — deterministic S1-level train/validation split.

All ground-truth matches for a given S1 entity stay together on one side of
the split, so nothing about S1-X's true matches leaks between the two.
Split is by a hash of the entity_id (not random.sample), so it is 100%
reproducible without needing to persist any state.

Run directly to write the split + the two ground-truth halves:

    python3 -m src.split
"""

import hashlib

import pandas as pd

from src import config


def _hash_frac(entity_id: str) -> float:
    """Deterministic pseudo-random value in [0, 1) derived from the id."""
    h = hashlib.md5(entity_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_split(entity_ids: pd.Series, val_fraction: float = config.VAL_FRACTION) -> pd.Series:
    fracs = entity_ids.map(_hash_frac)
    return pd.Series(
        ["val" if f < val_fraction else "train" for f in fracs],
        index=entity_ids.index,
    )


def build_split():
    gt = pd.read_csv(
        config.RAW_SOURCES["train"]["gt"], sep="\t", dtype=str, keep_default_na=False
    )
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    split = assign_split(gt["source1_entity_id"])
    gt["split"] = split

    counts = gt["split"].value_counts()
    print(f"S1 split: {counts.to_dict()}")

    dest = config.PROCESSED / "train_gt_split.parquet"
    gt.to_parquet(dest, index=False)
    print(f"-> {dest}")
    return gt


if __name__ == "__main__":
    build_split()
