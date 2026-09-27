import pandas as pd
import re
from unidecode import unidecode
from rapidfuzz.fuzz import ratio


BASE = "dataset/train"

S1_FILE = f"{BASE}/train_source1.tsv"
S2_FILE = f"{BASE}/train_source2.tsv"
S3_FILE = f"{BASE}/train_source3.tsv"
GT_FILE = f"{BASE}/train_ground_truth.tsv"


def transliterate(text):
    return re.sub(
        r"\s+",
        " ",
        unidecode(str(text)).lower()
    ).strip()


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

        a_name = transliterate(a["business_name"])
        b_name = transliterate(b["business_name"])

        sim = ratio(a_name, b_name)

        results.append({
            "source": src,
            "similarity": sim,
            "exact": a_name == b_name
        })


df = pd.DataFrame(results)

print()
print("=" * 70)
print("TRANSLITERATION ANALYSIS")
print("=" * 70)

for src in ["S2", "S3"]:

    d = df[df.source == src]

    print(f"\n--- {src} ---")

    print(f"True pairs: {len(d):,}")

    print(
        f"Transliterated exact: "
        f"{d.exact.mean() * 100:.2f}%"
    )

    print("\nSimilarity:")

    print(
        d.similarity.describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95
            ]
        ).round(2)
    )

    for threshold in [70, 80, 85, 90, 95]:

        recall = (
            d.similarity >= threshold
        ).mean()

        print(
            f"Similarity >= {threshold}: "
            f"{recall * 100:.2f}%"
        )


print("\nDONE")