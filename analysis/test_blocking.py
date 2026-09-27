import pandas as pd
import re
import unicodedata
from unidecode import unidecode
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 20_000
RANDOM_STATE = 42

TRAIN = "dataset/train"

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = unicodedata.normalize("NFKC", x)

    # Keep unicode characters, remove punctuation
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def translit_normalize(x):
    if pd.isna(x):
        return ""

    x = unidecode(str(x)).lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def get_tokens(x):
    return set(x.split())


def get_numbers(x):
    return set(re.findall(r"\d+", str(x)))


# Common business/legal words that create huge blocks
STOPWORDS = {
    "private",
    "limited",
    "ltd",
    "pvt",
    "llp",
    "inc",
    "incorporated",
    "company",
    "co",
    "corporation",
    "corp",
    "the",
    "and",
    "of",
    "services",
    "service",
    "enterprises",
    "enterprise",
    "business",
    "group",
    "india",
    "india",
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

s1 = pd.read_csv(
    f"{TRAIN}/train_source1.tsv",
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    f"{TRAIN}/train_source2.tsv",
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    f"{TRAIN}/train_source3.tsv",
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")


# ============================================================
# SAMPLE S1
# ============================================================

sample_ids = (
    s1["entity_id"]
    .sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    .tolist()
)

s1_sample = s1[s1["entity_id"].isin(sample_ids)].copy()

gt_sample = gt[
    gt["source1_entity_id"].isin(sample_ids)
].copy()

print(f"\nSample S1 entities: {len(s1_sample):,}")


# ============================================================
# GROUND TRUTH SETS
# ============================================================

truth = defaultdict(set)

for _, row in gt_sample.iterrows():

    s1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]):
        continue

    ids = str(row["matched_entity_ids"]).split(",")

    for entity_id in ids:
        entity_id = entity_id.strip()

        if entity_id:
            truth[s1_id].add(entity_id)


# Total true pairs
total_true_pairs = sum(len(v) for v in truth.values())

print(f"True pairs in sample: {total_true_pairs:,}")


# ============================================================
# PRECOMPUTE NORMALIZED FIELDS
# ============================================================

print("\nNormalizing fields...")


for df in [s1_sample, s2, s3]:

    df["name_norm"] = df["business_name"].map(normalize_text)
    df["name_trans"] = df["business_name"].map(translit_normalize)

    df["addr_norm"] = df["business_address"].fillna("").map(
        normalize_text
    )

    df["addr_numbers"] = df["business_address"].fillna("").map(
        get_numbers
    )

    df["name_tokens"] = df["name_norm"].map(get_tokens)
    df["name_trans_tokens"] = df["name_trans"].map(get_tokens)


# ============================================================
# BUILD INDEXES
# ============================================================

def build_index(df, column):

    index = defaultdict(list)

    for idx, value in enumerate(df[column]):

        if not value:
            continue

        index[value].append(idx)

    return index


def build_token_index(df, column):

    index = defaultdict(list)

    for idx, tokens in enumerate(df[column]):

        for token in tokens:

            if len(token) >= 3 and token not in STOPWORDS:
                index[token].append(idx)

    return index


def build_number_index(df):

    index = defaultdict(list)

    for idx, numbers in enumerate(df["addr_numbers"]):

        for number in numbers:

            index[number].append(idx)

    return index


print("\nBuilding indexes...")

s2_exact_name = build_index(s2, "name_norm")
s3_exact_name = build_index(s3, "name_norm")

s2_trans_name = build_index(s2, "name_trans")
s3_trans_name = build_index(s3, "name_trans")

s2_name_token = build_token_index(s2, "name_tokens")
s3_name_token = build_token_index(s3, "name_tokens")

s2_trans_token = build_token_index(s2, "name_trans_tokens")
s3_trans_token = build_token_index(s3, "name_trans_tokens")

s2_number = build_number_index(s2)
s3_number = build_number_index(s3)


# ============================================================
# COUNTRY FILTER
# ============================================================

s2_country = defaultdict(list)
s3_country = defaultdict(list)

for idx, country in enumerate(s2["country"]):
    s2_country[country].append(idx)

for idx, country in enumerate(s3["country"]):
    s3_country[country].append(idx)


# ============================================================
# BLOCKING FUNCTION
# ============================================================

def evaluate_block(
    block_name,
    get_candidates
):

    print(f"\n{'=' * 70}")
    print(block_name)
    print("=" * 70)

    candidate_count = 0
    retained_true = 0

    candidate_sizes = []

    for _, row in s1_sample.iterrows():

        s1_id = row["entity_id"]

        candidates = get_candidates(row)

        # Deduplicate
        candidates = set(candidates)

        candidate_count += len(candidates)
        candidate_sizes.append(len(candidates))

        true_ids = truth.get(s1_id, set())

        # Candidate IDs need to be actual entity IDs
        retained_true += len(
            true_ids.intersection(candidates)
        )

    recall = (
        retained_true / total_true_pairs
        if total_true_pairs
        else 0
    )

    avg_candidates = (
        candidate_count / len(s1_sample)
    )

    p95 = sorted(candidate_sizes)[
        int(0.95 * len(candidate_sizes))
    ]

    max_candidates = max(candidate_sizes)

    print(f"Candidate pairs:       {candidate_count:,}")
    print(f"True pairs retained:   {retained_true:,}")
    print(f"Candidate recall:      {recall:.4%}")
    print(f"Avg candidates / S1:   {avg_candidates:.2f}")
    print(f"P95 candidates / S1:   {p95:,}")
    print(f"Max candidates / S1:   {max_candidates:,}")

    return {
        "block": block_name,
        "candidates": candidate_count,
        "true_retained": retained_true,
        "recall": recall,
        "avg_candidates": avg_candidates,
        "p95": p95,
        "max": max_candidates,
    }


# ============================================================
# IMPORTANT:
# Return IDs, not row indexes
# ============================================================

s2_ids = s2["entity_id"].tolist()
s3_ids = s3["entity_id"].tolist()


# ============================================================
# BLOCK A: EXACT NORMALIZED NAME
# ============================================================

def block_exact_name(row, index, ids):

    return [
        ids[i]
        for i in index.get(row["name_norm"], [])
    ]


# ============================================================
# BLOCK B: EXACT TRANSLITERATED NAME
# ============================================================

def block_trans_name(row, index, ids):

    return [
        ids[i]
        for i in index.get(row["name_trans"], [])
    ]


# ============================================================
# BLOCK C: RARE ORIGINAL NAME TOKEN
# ============================================================

def block_name_token(row, index, ids):

    candidates = []

    tokens = row["name_tokens"]

    for token in tokens:

        if len(token) < 4 or token in STOPWORDS:
            continue

        # Don't use enormous blocks
        bucket = index.get(token, [])

        if len(bucket) > 5000:
            continue

        candidates.extend(ids[i] for i in bucket)

    return candidates


# ============================================================
# BLOCK D: RARE TRANSLITERATED TOKEN
# ============================================================

def block_trans_token(row, index, ids):

    candidates = []

    tokens = row["name_trans_tokens"]

    for token in tokens:

        if len(token) < 4 or token in STOPWORDS:
            continue

        bucket = index.get(token, [])

        if len(bucket) > 5000:
            continue

        candidates.extend(ids[i] for i in bucket)

    return candidates


# ============================================================
# BLOCK E: ADDRESS NUMBER
# ============================================================

def block_number(row, index, ids):

    candidates = []

    for number in row["addr_numbers"]:

        bucket = index.get(number, [])

        # Numbers such as 1, 10, 100 etc.
        # can produce gigantic blocks.
        if len(bucket) > 5000:
            continue

        candidates.extend(ids[i] for i in bucket)

    return candidates


# ============================================================
# RUN BLOCKS
# ============================================================

results = []


results.append(
    evaluate_block(
        "A — Exact normalized name",
        lambda row: block_exact_name(
            row,
            s2_exact_name,
            s2_ids
        )
        + block_exact_name(
            row,
            s3_exact_name,
            s3_ids
        )
    )
)


results.append(
    evaluate_block(
        "B — Exact transliterated name",
        lambda row: block_trans_name(
            row,
            s2_trans_name,
            s2_ids
        )
        + block_trans_name(
            row,
            s3_trans_name,
            s3_ids
        )
    )
)


results.append(
    evaluate_block(
        "C — Original rare name token",
        lambda row: block_name_token(
            row,
            s2_name_token,
            s2_ids
        )
        + block_name_token(
            row,
            s3_name_token,
            s3_ids
        )
    )
)


results.append(
    evaluate_block(
        "D — Transliteration rare name token",
        lambda row: block_trans_token(
            row,
            s2_trans_token,
            s2_ids
        )
        + block_trans_token(
            row,
            s3_trans_token,
            s3_ids
        )
    )
)


results.append(
    evaluate_block(
        "E — Address number",
        lambda row: block_number(
            row,
            s2_number,
            s2_ids
        )
        + block_number(
            row,
            s3_number,
            s3_ids
        )
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\n\n")
print("=" * 90)
print("BLOCKING SUMMARY")
print("=" * 90)

summary = pd.DataFrame(results)

print(
    summary[
        [
            "block",
            "candidates",
            "true_retained",
            "recall",
            "avg_candidates",
            "p95",
            "max",
        ]
    ].to_string(index=False)
)

print("\nDONE")