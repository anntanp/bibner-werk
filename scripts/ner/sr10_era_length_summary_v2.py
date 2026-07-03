# Purpose:      v2 of sr10_era_length_summary.py — era-bucket (pre-1700/1700-1800/
#               19th-c/modern/unknown) short/medium/long title-length summary,
#               computed directly from the reprocessed parquet corpus (9.21M rows)
#               instead of the retired pkl (4.48M rows, spaCy tokens).
#               Not a duplicate of sr10_analyse_title_lengths.py: that script's saved
#               JSON (title-length-analysis_words.json) buckets by 25-year period and
#               explicitly excludes pre-1500 records from the saved "bucketed" field
#               (only prints their count to console, never persists it), and does not
#               break out the no-year ("unknown") stratum's length distribution at all.
#               Aggregating that JSON into era buckets would therefore silently
#               undercount pre-1700 (the paper's hardest/most-cited stratum) and could
#               not produce a usable "unknown" row. This script re-derives era totals
#               from per-record data so both gaps are closed.
#               Default mode uses 'all_tokens' (xlm-roberta-large BPE), thresholds
#               recalibrated to the v2 quartiles (p25=8, p75=27; see
#               sr10_de-titles-distribution.md §7.2), not the stale v1 thresholds (4/14).
#               --word-tokens mode measures whitespace word count instead (mirrors
#               sr10_analyse_title_lengths.py --word-tokens: short_max=3, medium_max=12),
#               which is tokenizer-agnostic and comparable across the pkl/parquet/cc2llm
#               corpus generations — the preferred metric for citing in the paper.
# Usage:        python scripts/ner/sr10_era_length_summary_v2.py
#               python scripts/ner/sr10_era_length_summary_v2.py --short-max 4 --medium-max 14 --tag legacy14
#               python scripts/ner/sr10_era_length_summary_v2.py --word-tokens --tag words
# Inputs:       gemea/data/processed/de_titles_tokenized.parquet (passed via --data;
#               stays in gemea per bibner-werk conventions, not copied into this repo)
# Outputs:      data/processed/ner/sr10_era_length_summary_v2_<tag>.csv
# Dependencies: pandas, pyarrow
# Assumptions:  'all_tokens' = xlm-roberta-large SentencePiece BPE piece count (default mode).
#               Word mode: whitespace split on lowercased 'title', same as
#               sr10_analyse_title_lengths.py --word-tokens.
#               'dates' column is a string year (e.g. '1931') or NaN.
#               Era boundaries: pre-1700 (<1700), 1700-1800 (1700-1799),
#               19th-c (1800-1899), modern (>=1900), unknown (no year, no title-regex match).
#               No lower year bound — pre-1500 records ARE included in pre-1700
#               (unlike sr10_analyse_title_lengths.py's saved JSON).

import re
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = Path(
    "/Users/mta/Documents/claude/gemea/data/processed/de_titles_tokenized.parquet"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "ner"

WORD_SHORT_MAX = 3
WORD_MEDIUM_MAX = 12

YEAR_RE = re.compile(r"\b(?:1[4-9]\d{2}|20[012]\d)\b")

ERA_ORDER = ["pre-1700", "1700-1800", "19th-c", "modern", "unknown"]


def year_from_title(title: str):
    m = list(YEAR_RE.finditer(str(title)))
    return int(m[-1].group()) if m else None


def assign_era(year):
    if year is None or pd.isna(year):
        return "unknown"
    if year < 1700:
        return "pre-1700"
    if year < 1800:
        return "1700-1800"
    if year < 1900:
        return "19th-c"
    return "modern"


def median(series):
    s = series.sort_values()
    return int(s.iloc[len(s) // 2]) if len(s) else None


def main(data_path: Path, output_path: Path, short_max: int, medium_max: int, word_tokens: bool) -> None:
    columns = ["title", "dates"] if word_tokens else ["title", "dates", "all_tokens"]
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path, columns=columns)
    print(f"Shape: {df.shape}")

    if word_tokens:
        print("Computing whitespace word counts ...")
        df["length"] = df["title"].str.lower().str.split().str.len()
    else:
        df["length"] = df["all_tokens"]

    def resolve_year(row):
        if pd.notna(row["dates"]) and str(row["dates"]).strip():
            try:
                y = int(str(row["dates"]).strip()[:4])
                if 1400 <= y <= 2029:
                    return y
            except ValueError:
                pass
        return year_from_title(row["title"])

    print("Resolving years ...")
    df["year"] = df.apply(resolve_year, axis=1)
    df["era"] = df["year"].apply(assign_era)

    rows = []
    for era in ERA_ORDER:
        sub = df[df["era"] == era]["length"]
        n = len(sub)
        short = (sub <= short_max).sum()
        medium = ((sub > short_max) & (sub <= medium_max)).sum()
        long = (sub > medium_max).sum()
        rows.append({
            "era": era,
            "records": n,
            "short_n": int(short),
            "medium_n": int(medium),
            "long_n": int(long),
            "short_pct": round(100 * short / n, 1) if n else None,
            "medium_pct": round(100 * medium / n, 1) if n else None,
            "long_pct": round(100 * long / n, 1) if n else None,
            "median_length": median(sub),
        })

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    unit = "words" if word_tokens else "tokens"
    print(f"\n{'Era':<12}  {'Records':>9}  {'Short%':>7}  {'Medium%':>8}  {'Long%':>6}  {'Median (' + unit + ')':>14}")
    print("-" * 70)
    for r in rows:
        if r["records"] == 0:
            print(f"{r['era']:<12}  {r['records']:>9,}  {'—':>7}  {'—':>8}  {'—':>6}  {'—':>14}")
        else:
            print(f"{r['era']:<12}  {r['records']:>9,}  {r['short_pct']:>6.1f}%  "
                  f"{r['medium_pct']:>7.1f}%  {r['long_pct']:>5.1f}%  {r['median_length']:>14}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--word-tokens", action="store_true",
                         help="Use whitespace word count instead of BPE all_tokens")
    parser.add_argument("--short-max", type=int, default=None)
    parser.add_argument("--medium-max", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.word_tokens:
        short_max = args.short_max if args.short_max is not None else WORD_SHORT_MAX
        medium_max = args.medium_max if args.medium_max is not None else WORD_MEDIUM_MAX
        tag = args.tag or "words"
    else:
        short_max = args.short_max if args.short_max is not None else 8
        medium_max = args.medium_max if args.medium_max is not None else 27
        tag = args.tag or "p25p75"

    output = args.output or (OUTPUT_DIR / f"sr10_era_length_summary_v2_{tag}.csv")
    main(args.data, output, short_max, medium_max, args.word_tokens)
