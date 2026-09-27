"""Phase 1 — data loading + normalization.

Reads a raw source TSV once, builds every text representation we need
downstream (normalized, transliterated, tokenized), and writes a single
Parquet file per (split, source). Everything after this module reads the
Parquet files, never the raw TSVs, so normalization happens exactly once.

Run directly to (re)build all six processed tables:

    python3 -m src.normalize
"""

import re
import sys
import time

import pandas as pd
from unidecode import unidecode

from src import config

_PUNCT_RE = re.compile(r"[^0-9a-zA-ZÀ-ɏऀ-෿ ]")
_SPACE_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"^[0-9]+$")


def _norm_text(s: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace.

    Keeps Unicode letters (Latin-extended + the Devanagari/Indic block range
    covering Hindi/Gujarati/Tamil/Kannada scripts seen in the data) so this
    representation never destroys non-Latin script information.
    """
    s = _PUNCT_RE.sub(" ", s.lower())
    return _SPACE_RE.sub(" ", s).strip()


def _translit_norm(s: str) -> str:
    """ASCII transliteration, then the same normalization."""
    return _norm_text(unidecode(s))


def _map_unique(series: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` only to the unique values of ``series``, then map back.

    Business names/addresses repeat across duplicate-ish records; this avoids
    re-running unidecode/regex on the same string twice.
    """
    uniques = series.unique()
    mapping = {v: fn(v) for v in uniques}
    return series.map(mapping)


def _tokens(series: pd.Series, min_len: int) -> pd.Series:
    def split(s):
        return [t for t in s.split(" ") if len(t) >= min_len]

    return _map_unique(series, split)


def _numeric_tokens(series: pd.Series) -> pd.Series:
    def nums(s):
        return [t for t in s.split(" ") if _NUM_RE.match(t)]

    return _map_unique(series, nums)


def build_table(raw_path, min_token_len=config.MIN_TOKEN_LEN) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(
        raw_path, sep="\t", dtype=str, keep_default_na=False
    )
    for col in ("business_name", "business_address", "country"):
        if col not in df.columns:
            df[col] = ""
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("").str.strip()

    out = pd.DataFrame({"entity_id": df["entity_id"], "country": df["country"]})

    out["name_norm"] = _map_unique(df["business_name"], _norm_text)
    out["addr_norm"] = _map_unique(df["business_address"], _norm_text)
    out["name_translit"] = _map_unique(df["business_name"], _translit_norm)
    out["addr_translit"] = _map_unique(df["business_address"], _translit_norm)

    out["name_tokens"] = _tokens(out["name_norm"], min_token_len)
    out["addr_tokens"] = _tokens(out["addr_norm"], min_token_len)
    out["name_translit_tokens"] = _tokens(out["name_translit"], min_token_len)
    out["addr_numbers"] = _numeric_tokens(out["addr_norm"])

    print(f"  built {raw_path.name}: {len(out):,} rows in {time.time()-t0:.1f}s")
    return out


def build_and_save(split: str, source: str):
    raw_path = config.RAW_SOURCES[split][source]
    out = build_table(raw_path)
    dest = config.processed_path(split, source)
    out.to_parquet(dest, index=False)
    print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def main():
    only = sys.argv[1:] or None
    for split, sources in config.RAW_SOURCES.items():
        for source in sources:
            if source == "gt":
                continue
            key = f"{split}_{source}"
            if only and key not in only:
                continue
            print(f"[normalize] {key}")
            build_and_save(split, source)


if __name__ == "__main__":
    main()
