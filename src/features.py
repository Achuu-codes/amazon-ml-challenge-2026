"""Phase 4 — pairwise feature extraction.

Takes the candidate_pairs table from ``src.blocking`` (s1_id, candidate_id),
merges in the normalized text for both sides, and computes RapidFuzz +
set-overlap features per pair, streaming chunks to Parquet so peak memory
stays bounded regardless of total candidate volume.

Deliberately single-process: an earlier multiprocessing version (5 worker
processes, each holding its own copy of the side tables) hit a system-wide
memory/swap crisis on this machine (unrelated stray processes ate most of
the 16GB RAM) and hung in a way multiprocessing.Pool couldn't recover from.
Single-process is slower (~40-90k pairs/s here) but has one predictable,
bounded memory footprint (~2GB) and no IPC failure modes. Blocking caps were
also tightened (see src.blocking) specifically to keep total volume — and
therefore this step's wall-clock time — manageable under that constraint.

Entry point:

    python3 -m src.features --split train
    python3 -m src.features --split test
"""

import argparse
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from src import config

CHUNK_ROWS = 3_000_000
SIDE_COLS = ["entity_id", "country", "name_norm", "addr_norm", "name_translit"]


def _load_side_tables(split):
    frames = {}
    for source in ["s1", "s2", "s3"]:
        path = config.processed_path(split, source)
        df = pd.read_parquet(path, columns=SIDE_COLS)
        frames[source] = df
    s1_df = frames["s1"].rename(columns={c: f"{c}_a" for c in SIDE_COLS if c != "entity_id"})
    s2_df = frames["s2"].rename(columns={c: f"{c}_b" for c in SIDE_COLS if c != "entity_id"})
    s3_df = frames["s3"].rename(columns={c: f"{c}_b" for c in SIDE_COLS if c != "entity_id"})
    return s1_df, s2_df, s3_df


def _merge_chunk(chunk, s1_df, s2_df, s3_df):
    merged = chunk.merge(s1_df, left_on="s1_id", right_on="entity_id", how="left")
    is_s2 = merged["candidate_id"].str.startswith("S2")
    m2 = merged[is_s2].merge(s2_df, left_on="candidate_id", right_on="entity_id", how="left")
    m3 = merged[~is_s2].merge(s3_df, left_on="candidate_id", right_on="entity_id", how="left")
    merged = pd.concat([m2, m3], ignore_index=True)
    for col in SIDE_COLS:
        if col == "entity_id":
            continue
        merged[f"{col}_a"] = merged[f"{col}_a"].fillna("")
        merged[f"{col}_b"] = merged[f"{col}_b"].fillna("")
    return merged


def _numbers(s):
    return {t for t in s.split(" ") if t.isdigit()}


def compute_features(merged: pd.DataFrame) -> pd.DataFrame:
    n = len(merged)

    a_name = merged["name_norm_a"].tolist()
    b_name = merged["name_norm_b"].tolist()
    a_translit = merged["name_translit_a"].tolist()
    b_translit = merged["name_translit_b"].tolist()
    a_addr = merged["addr_norm_a"].tolist()
    b_addr = merged["addr_norm_b"].tolist()
    a_country = merged["country_a"].tolist()
    b_country = merged["country_b"].tolist()

    name_ratio = np.empty(n, dtype=np.float32)
    name_token_sort = np.empty(n, dtype=np.float32)
    name_token_set = np.empty(n, dtype=np.float32)
    translit_ratio = np.empty(n, dtype=np.float32)
    addr_ratio = np.empty(n, dtype=np.float32)
    addr_token_set = np.empty(n, dtype=np.float32)

    name_exact = np.empty(n, dtype=np.int8)
    translit_exact = np.empty(n, dtype=np.int8)
    addr_exact = np.empty(n, dtype=np.int8)

    name_jaccard = np.empty(n, dtype=np.float32)
    name_common = np.empty(n, dtype=np.int16)
    addr_jaccard = np.empty(n, dtype=np.float32)
    addr_common = np.empty(n, dtype=np.int16)
    shared_num_count = np.empty(n, dtype=np.int16)

    name_len_diff = np.empty(n, dtype=np.int16)
    name_len_ratio = np.empty(n, dtype=np.float32)
    addr_len_diff = np.empty(n, dtype=np.int16)
    addr_len_ratio = np.empty(n, dtype=np.float32)

    country_equal = np.empty(n, dtype=np.int8)
    s1_name_missing = np.empty(n, dtype=np.int8)
    cand_name_missing = np.empty(n, dtype=np.int8)
    s1_addr_missing = np.empty(n, dtype=np.int8)
    cand_addr_missing = np.empty(n, dtype=np.int8)

    for i in range(n):
        an, bn = a_name[i], b_name[i]
        at, bt = a_translit[i], b_translit[i]
        aa, ba = a_addr[i], b_addr[i]

        if an and bn:
            name_ratio[i] = fuzz.ratio(an, bn)
            name_token_sort[i] = fuzz.token_sort_ratio(an, bn)
            name_token_set[i] = fuzz.token_set_ratio(an, bn)
        else:
            name_ratio[i] = name_token_sort[i] = name_token_set[i] = 0.0

        translit_ratio[i] = fuzz.ratio(at, bt) if (at and bt) else 0.0

        if aa and ba:
            addr_ratio[i] = fuzz.ratio(aa, ba)
            addr_token_set[i] = fuzz.token_set_ratio(aa, ba)
        else:
            addr_ratio[i] = addr_token_set[i] = 0.0

        name_exact[i] = int(bool(an) and an == bn)
        translit_exact[i] = int(bool(at) and at == bt)
        addr_exact[i] = int(bool(aa) and aa == ba)

        a_ntok, b_ntok = set(an.split(" ")) if an else set(), set(bn.split(" ")) if bn else set()
        u = a_ntok | b_ntok
        name_jaccard[i] = len(a_ntok & b_ntok) / len(u) if u else 0.0
        name_common[i] = len(a_ntok & b_ntok)

        a_atok, b_atok = set(aa.split(" ")) if aa else set(), set(ba.split(" ")) if ba else set()
        u2 = a_atok | b_atok
        addr_jaccard[i] = len(a_atok & b_atok) / len(u2) if u2 else 0.0
        addr_common[i] = len(a_atok & b_atok)

        shared_num_count[i] = len(_numbers(aa) & _numbers(ba))

        name_len_diff[i] = abs(len(an) - len(bn))
        mx = max(len(an), len(bn))
        name_len_ratio[i] = min(len(an), len(bn)) / mx if mx else 0.0

        addr_len_diff[i] = abs(len(aa) - len(ba))
        mx2 = max(len(aa), len(ba))
        addr_len_ratio[i] = min(len(aa), len(ba)) / mx2 if mx2 else 0.0

        country_equal[i] = int(bool(a_country[i]) and a_country[i] == b_country[i])
        s1_name_missing[i] = int(not an)
        cand_name_missing[i] = int(not bn)
        s1_addr_missing[i] = int(not aa)
        cand_addr_missing[i] = int(not ba)

    out = pd.DataFrame({
        "s1_id": merged["s1_id"].values,
        "candidate_id": merged["candidate_id"].values,
        "name_ratio": name_ratio,
        "name_token_sort_ratio": name_token_sort,
        "name_token_set_ratio": name_token_set,
        "name_translit_ratio": translit_ratio,
        "addr_ratio": addr_ratio,
        "addr_token_set_ratio": addr_token_set,
        "name_exact": name_exact,
        "name_translit_exact": translit_exact,
        "addr_exact": addr_exact,
        "name_token_jaccard": name_jaccard,
        "name_common_tokens": name_common,
        "addr_token_jaccard": addr_jaccard,
        "addr_common_tokens": addr_common,
        "shared_numeric_count": shared_num_count,
        "shared_numeric_bool": (shared_num_count > 0).astype(np.int8),
        "country_equal": country_equal,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_len_diff": addr_len_diff,
        "addr_len_ratio": addr_len_ratio,
        "s1_name_missing": s1_name_missing,
        "cand_name_missing": cand_name_missing,
        "s1_addr_missing": s1_addr_missing,
        "cand_addr_missing": cand_addr_missing,
        "is_source3": merged["candidate_id"].str.startswith("S3").astype(np.int8).values,
    })
    return out


def build_features(split, candidates_path=None, label=None):
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")

    cand_path = str(candidates_path or (config.PROCESSED / f"{split}_candidates.parquet"))
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{cand_path}')").fetchone()[0]
    tag = label or split
    print(f"[features] {tag}: {n_total:,} candidate pairs")

    s1_df, s2_df, s3_df = _load_side_tables(split)
    print(f"[features] side tables loaded ({time.time()-t0:.1f}s)")

    dest = config.FEATURES / f"{tag}_features.parquet"
    writer = None
    offset = 0
    chunk_idx = 0

    while offset < n_total:
        chunk = con.execute(f"""
            SELECT s1_id, candidate_id FROM read_parquet('{cand_path}')
            LIMIT {CHUNK_ROWS} OFFSET {offset}
        """).df()
        if chunk.empty:
            break

        merged = _merge_chunk(chunk, s1_df, s2_df, s3_df)
        result = compute_features(merged)

        table = pa.Table.from_pandas(result, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dest, table.schema)
        writer.write_table(table)

        offset += len(chunk)
        chunk_idx += 1
        elapsed = time.time() - t0
        rate = offset / elapsed if elapsed else 0
        eta = (n_total - offset) / rate if rate else 0
        print(f"  chunk {chunk_idx}: {offset:,}/{n_total:,} rows  "
              f"({elapsed:.0f}s elapsed, {rate:,.0f} pairs/s, eta {eta:.0f}s)")

    if writer is not None:
        writer.close()
    print(f"[features] done -> {dest} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "test"])
    args = parser.parse_args()
    build_features(args.split)
