import pandas as pd
import re
import unicodedata
from unidecode import unidecode
from collections import defaultdict

SAMPLE_SIZE = 20_000
RANDOM_STATE = 42

TRAIN = "dataset/train"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = unicodedata.normalize("NFKC", str(x).lower())
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


def tokens(x):
    return set(x.split())


def numbers(x):
    return set(re.findall(r"\d+", str(x)))


STOPWORDS = {
    "private", "limited", "ltd", "pvt", "llp",
    "inc", "incorporated", "company", "co",
    "corporation", "corp", "the", "and", "of",
    "services", "service", "enterprises",
    "enterprise", "business", "group",
    "india"
}


# ============================================================
# LOAD
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

sample_ids = (
    s1["entity_id"]
    .sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    .tolist()
)

s1 = s1[s1["entity_id"].isin(sample_ids)].copy()

gt = gt[
    gt["source1_entity_id"].isin(sample_ids)
].copy()


# ============================================================
# GROUND TRUTH
# ============================================================

truth = defaultdict(set)

for _, row in gt.iterrows():

    if pd.isna(row["matched_entity_ids"]):
        continue

    for x in str(row["matched_entity_ids"]).split(","):

        x = x.strip()

        if x:
            truth[row["source1_entity_id"]].add(x)

total_true = sum(len(x) for x in truth.values())

print(f"S1 sample: {len(s1):,}")
print(f"True pairs: {total_true:,}")


# ============================================================
# PREPROCESS
# ============================================================

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].map(
        normalize_text
    )

    df["name_trans"] = df["business_name"].map(
        translit_normalize
    )

    df["addr_norm"] = df["business_address"].fillna("").map(
        normalize_text
    )

    df["name_tokens"] = df["name_norm"].map(tokens)

    df["trans_tokens"] = df["name_trans"].map(tokens)

    df["numbers"] = df["business_address"].fillna("").map(
        numbers
    )


# ============================================================
# INDEX BUILDERS
# ============================================================

def add_index(index, key, row_id):

    if key:
        index[key].append(row_id)


def build_name_number_index(df, token_column):

    index = defaultdict(list)

    for i, row in df.iterrows():

        nums = row["numbers"]

        if not nums:
            continue

        for token in row[token_column]:

            if len(token) < 4:
                continue

            if token in STOPWORDS:
                continue

            for num in nums:

                index[(token, num)].append(i)

    return index


def build_token_citylike_index(df, token_column):

    """
    Approximation of locality evidence using address tokens.

    We deliberately don't assume a country-specific schema.
    """

    index = defaultdict(list)

    for i, row in df.iterrows():

        addr_tokens = row["addr_norm"].split()

        if not addr_tokens:
            continue

        useful_addr = {
            x for x in addr_tokens
            if len(x) >= 4 and not x.isdigit()
        }

        for name_token in row[token_column]:

            if len(name_token) < 4:
                continue

            if name_token in STOPWORDS:
                continue

            for addr_token in useful_addr:

                index[(name_token, addr_token)].append(i)

    return index


def build_name_prefix_number_index(df, token_column):

    index = defaultdict(list)

    for i, row in df.iterrows():

        nums = row["numbers"]

        if not nums:
            continue

        for token in row[token_column]:

            if len(token) < 5:
                continue

            if token in STOPWORDS:
                continue

            prefix = token[:5]

            for num in nums:

                index[(prefix, num)].append(i)

    return index


print("\nBuilding indexes...")

s2_nn = build_name_number_index(s2, "name_tokens")
s3_nn = build_name_number_index(s3, "name_tokens")

s2_tn = build_name_number_index(s2, "trans_tokens")
s3_tn = build_name_number_index(s3, "trans_tokens")

s2_nt = build_token_citylike_index(s2, "name_tokens")
s3_nt = build_token_citylike_index(s3, "name_tokens")

s2_tt = build_token_citylike_index(s2, "trans_tokens")
s3_tt = build_token_citylike_index(s3, "trans_tokens")

s2_pn = build_name_prefix_number_index(s2, "name_tokens")
s3_pn = build_name_prefix_number_index(s3, "name_tokens")


s2_ids = s2["entity_id"].tolist()
s3_ids = s3["entity_id"].tolist()


# ============================================================
# CANDIDATE HELPERS
# ============================================================

def candidates_name_number(row, index, ids):

    result = set()

    for token in row["name_tokens"]:

        if len(token) < 4 or token in STOPWORDS:
            continue

        for num in row["numbers"]:

            bucket = index.get((token, num), [])

            # Prevent pathological buckets
            if len(bucket) <= 2000:
                result.update(ids[i] for i in bucket)

    return result


def candidates_trans_number(row, index, ids):

    result = set()

    for token in row["trans_tokens"]:

        if len(token) < 4 or token in STOPWORDS:
            continue

        for num in row["numbers"]:

            bucket = index.get((token, num), [])

            if len(bucket) <= 2000:
                result.update(ids[i] for i in bucket)

    return result


def candidates_name_addr_token(row, index, ids):

    result = set()

    addr_tokens = {
        x for x in row["addr_norm"].split()
        if len(x) >= 4 and not x.isdigit()
    }

    for name_token in row["name_tokens"]:

        if len(name_token) < 4:
            continue

        if name_token in STOPWORDS:
            continue

        for addr_token in addr_tokens:

            bucket = index.get(
                (name_token, addr_token),
                []
            )

            if len(bucket) <= 1000:
                result.update(ids[i] for i in bucket)

    return result


def candidates_trans_addr_token(row, index, ids):

    result = set()

    addr_tokens = {
        x for x in row["addr_norm"].split()
        if len(x) >= 4 and not x.isdigit()
    }

    for name_token in row["trans_tokens"]:

        if len(name_token) < 4:
            continue

        if name_token in STOPWORDS:
            continue

        for addr_token in addr_tokens:

            bucket = index.get(
                (name_token, addr_token),
                []
            )

            if len(bucket) <= 1000:
                result.update(ids[i] for i in bucket)

    return result


def candidates_prefix_number(row, index, ids):

    result = set()

    for token in row["name_tokens"]:

        if len(token) < 5:
            continue

        if token in STOPWORDS:
            continue

        prefix = token[:5]

        for num in row["numbers"]:

            bucket = index.get(
                (prefix, num),
                []
            )

            if len(bucket) <= 2000:
                result.update(ids[i] for i in bucket)

    return result


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, fn):

    candidates_total = 0
    retained = 0

    sizes = []

    for _, row in s1.iterrows():

        candidate_ids = fn(row)

        candidates_total += len(candidate_ids)

        sizes.append(len(candidate_ids))

        retained += len(
            truth.get(row["entity_id"], set())
            & candidate_ids
        )

    sizes.sort()

    recall = retained / total_true

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Candidate pairs:       {candidates_total:,}")
    print(f"True pairs retained:   {retained:,}")
    print(f"Candidate recall:      {recall:.4%}")
    print(
        f"Avg candidates / S1:   "
        f"{candidates_total / len(s1):.2f}"
    )
    print(
        f"P95 candidates / S1:   "
        f"{sizes[int(len(sizes)*0.95)]:,}"
    )
    print(
        f"Max candidates / S1:   "
        f"{max(sizes):,}"
    )

    return {
        "block": name,
        "candidates": candidates_total,
        "retained": retained,
        "recall": recall,
        "avg": candidates_total / len(s1),
    }


results = []


# ============================================================
# F — ORIGINAL NAME TOKEN + NUMBER
# ============================================================

results.append(
    evaluate(
        "F — Name token + address number",
        lambda row:
            candidates_name_number(
                row, s2_nn, s2_ids
            )
            |
            candidates_name_number(
                row, s3_nn, s3_ids
            )
    )
)


# ============================================================
# G — TRANSLITERATED TOKEN + NUMBER
# ============================================================

results.append(
    evaluate(
        "G — Transliteration token + address number",
        lambda row:
            candidates_trans_number(
                row, s2_tn, s2_ids
            )
            |
            candidates_trans_number(
                row, s3_tn, s3_ids
            )
    )
)


# ============================================================
# H — NAME TOKEN + ADDRESS TOKEN
# ============================================================

results.append(
    evaluate(
        "H — Name token + address token",
        lambda row:
            candidates_name_addr_token(
                row, s2_nt, s2_ids
            )
            |
            candidates_name_addr_token(
                row, s3_nt, s3_ids
            )
    )
)


# ============================================================
# I — TRANSLITERATED NAME TOKEN + ADDRESS TOKEN
# ============================================================

results.append(
    evaluate(
        "I — Transliteration token + address token",
        lambda row:
            candidates_trans_addr_token(
                row, s2_tt, s2_ids
            )
            |
            candidates_trans_addr_token(
                row, s3_tt, s3_ids
            )
    )
)


# ============================================================
# J — NAME PREFIX + NUMBER
# ============================================================

results.append(
    evaluate(
        "J — Name prefix + address number",
        lambda row:
            candidates_prefix_number(
                row, s2_pn, s2_ids
            )
            |
            candidates_prefix_number(
                row, s3_pn, s3_ids
            )
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\n\n")
print("=" * 90)
print("COMPOUND BLOCKING SUMMARY")
print("=" * 90)

summary = pd.DataFrame(results)

print(summary.to_string(index=False))

print("\nDONE")