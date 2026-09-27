"""Shared paths, constants and seeds for the entity-resolution pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset"
TRAIN_DIR = DATASET / "train"
TEST_DIR = DATASET / "test"

DATA = ROOT / "data"
PROCESSED = DATA / "processed"

MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
LOGS = ROOT / "logs"
FEATURES = ROOT / "features"
EXPERIMENTS = ROOT / "experiments"

for d in [PROCESSED, MODELS, OUTPUT, LOGS, FEATURES, EXPERIMENTS]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Train ground-truth S1 entities are split at the entity level so that no S1
# entity (and its true matched pairs) leaks between train/validation.
VAL_FRACTION = 0.20

# Token-based blocking: ignore tokens shorter than this (too weak a signal)
MIN_TOKEN_LEN = 3

# A blocking key present in more than this many source records is dropped —
# it is too common to be useful and only inflates the candidate set.
MAX_BUCKET_SIZE = 400

RAW_SOURCES = {
    "train": {
        "s1": TRAIN_DIR / "train_source1.tsv",
        "s2": TRAIN_DIR / "train_source2.tsv",
        "s3": TRAIN_DIR / "train_source3.tsv",
        "gt": TRAIN_DIR / "train_ground_truth.tsv",
    },
    "test": {
        "s1": TEST_DIR / "test_source1.tsv",
        "s2": TEST_DIR / "test_source2.tsv",
        "s3": TEST_DIR / "test_source3.tsv",
    },
}


def processed_path(split: str, source: str) -> Path:
    return PROCESSED / f"{split}_{source}.parquet"
