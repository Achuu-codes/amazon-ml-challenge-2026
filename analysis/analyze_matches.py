import pandas as pd
import re

BASE = "dataset/train"

S1_FILE = f"{BASE}/train_source1.tsv"
S2_FILE = f"{BASE}/train_source2.tsv"
S3_FILE = f"{BASE}/train_source3.tsv"
GT_FILE = f"{BASE}/train_ground_truth.tsv"


def numbers(text):
    return set(re.findall(r"\d+", str(text)))


def tokens(text):
    return set(
        re.findall(
            r"[A-Za-z0-9]+",
            str(text).lower()
        )
    )


def load(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    ).set_index("entity_id")


print("Loading...")
s1 = load(S1_FILE)
s2 = load(S2_FILE)
s3 = load(S3_FILE)

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# Sample 20,000 S1 entities
gt = gt.sample(
    min(20000, len(gt)),
    random_state=42
)

results = []

for _, row in gt.iterrows():

    sid = row["source1_entity_id"]

    if sid not in s1.index:
        continue

    a = s1.loc[sid]

    for tid in row["matched_entity_ids"].split(","):

        tid = tid.strip()

        if tid.startswith("S2-"):
            source = s2
            src = "S2"
        elif tid.startswith("S3-"):
            source = s3
            src = "S3"
        else:
            continue

        if tid not in source.index:
            continue

        b = source.loc[tid]

        a_addr = a["business_address"]
        b_addr = b["business_address"]

        a_tokens = tokens(a_addr)
        b_tokens = tokens(b_addr)

        a_nums = numbers(a_addr)
        b_nums = numbers(b_addr)

        results.append({
            "source": src,

            "address_token_overlap":
                len(a_tokens & b_tokens),

            "address_token_jaccard":
                len(a_tokens & b_tokens) /
                len(a_tokens | b_tokens)
                if a_tokens | b_tokens else 0,

            "shared_numbers":
                len(a_nums & b_nums),

            "number_match":
                bool(a_nums & b_nums),

            "same_country":
                a["country"] == b["country"],

            "s1_address_tokens":
                len(a_tokens),

            "target_address_tokens":
                len(b_tokens),
        })


df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("ADDRESS COMPONENT ANALYSIS")
print("=" * 70)

for src in ["S2", "S3"]:

    d = df[df.source == src]

    print(f"\n--- {src} ---")
    print(f"True pairs: {len(d):,}")

    print(
        "At least one shared number: "
        f"{d.number_match.mean()*100:.2f}%"
    )

    print(
        "Mean shared numeric tokens: "
        f"{d.shared_numbers.mean():.3f}"
    )

    print(
        "Mean address token overlap: "
        f"{d.address_token_overlap.mean():.3f}"
    )

    print(
        "Median address Jaccard: "
        f"{d.address_token_jaccard.median():.3f}"
    )

    print(
        "Country agrees: "
        f"{d.same_country.mean()*100:.2f}%"
    )

print("\nDONE")