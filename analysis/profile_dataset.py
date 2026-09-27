import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRAIN = BASE / "dataset" / "train"
TEST = BASE / "dataset" / "test"


def profile_file(path, chunksize=100_000):
    print(f"\n{'=' * 70}")
    print(path.name)
    print(f"{'=' * 70}")

    total = 0
    countries = {}
    name_missing = 0
    address_missing = 0
    ids = set()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False
    ):
        total += len(chunk)

        name_missing += (chunk["business_name"].str.strip() == "").sum()
        address_missing += (chunk["business_address"].str.strip() == "").sum()

        for country, count in chunk["country"].value_counts().items():
            countries[country] = countries.get(country, 0) + count

        ids.update(chunk["entity_id"])

    print("Rows:", total)
    print("Unique IDs:", len(ids))
    print("Duplicate IDs:", total - len(ids))
    print("Missing names:", name_missing)
    print("Missing addresses:", address_missing)
    print("Countries:", countries)


def profile_ground_truth(path, chunksize=100_000):
    print(f"\n{'=' * 70}")
    print("GROUND TRUTH")
    print(f"{'=' * 70}")

    total = 0
    singletons = 0
    match_counts = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False
    ):
        total += len(chunk)

        for value in chunk["matched_entity_ids"]:
            if not value.strip():
                singletons += 1
                match_counts.append(0)
            else:
                count = len(value.split(","))
                match_counts.append(count)

    s = pd.Series(match_counts)

    print("S1 entities:", total)
    print("Singletons:", singletons)
    print("Singleton rate:", singletons / total)
    print("Matched entities:", total - singletons)
    print("\nMatches per S1:")
    print(s.describe())

    print("\nDistribution:")
    print(s.value_counts().sort_index().head(30))


if __name__ == "__main__":

    for filename in [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
    ]:
        profile_file(TRAIN / filename)

    profile_ground_truth(
        TRAIN / "train_ground_truth.tsv"
    )

    for filename in [
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ]:
        profile_file(TEST / filename)