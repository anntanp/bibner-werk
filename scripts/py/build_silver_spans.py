#!/usr/bin/env python3
# Purpose:      Stage 5 of the NER silver-dataset pipeline. Extract char-offset
#               NER spans (TITLE, OTHER_TITLE, PERSON) from ISBD field-flag
#               ratings; sub-classify PERSON via f_resp_* (ADR-06); recompute
#               tier-2 membership under the ADR-06 f_resp_person criterion
#               (the ratings CSV's silver_tier column predates that amendment);
#               era-join against the tokenized parquet; keep corrected tier-2
#               in full and an era-stratified tier-1 subsample; exclude all
#               gold record IDs.
# Usage:        python3 scripts/py/build_silver_spans.py --ratings PATH
#                   --parquet PATH [--gold-sample PATH] [--output PATH]
#                   [--stats-output PATH] [--join-failures-output PATH]
#                   [--tier1-sample-size N]
#                   [--stratify-mode proportional|balanced] [--seed N]
# Inputs:       sr01_isbd_field_ratings.csv, de_titles_tokenized.parquet,
#               sr08_gold_sample.csv
# Outputs:      data/processed/silver_spans.jsonl, data/processed/sr_silver_spans_stats.csv,
#               data/processed/sr_silver_spans_join_failures.csv (only written if non-empty)
# Dependencies: pandas, pyarrow
# Assumptions:  obj_id join ratings <-> parquet verified 1:1 as of 2026-07-06
#               (9,213,339 rows both sides, identical obj_id sets both
#               directions). An unmatched obj_id is not fatal: resolve_year()
#               falls back to a title-regex year match when the parquet
#               'dates' column is unavailable, so the record still gets
#               processed and era-bucketed normally; it is additionally
#               logged to --join-failures-output with enough context (title,
#               tier, key flags, resolved era) to investigate later, as a
#               guard against future drift between the two upstream scripts
#               (sr01_rate_isbd_fields.py, tokenize_de_titles.py), not because
#               a gap is currently expected. Era assignment logic replicated from
#               scripts/ner/sr10_era_length_summary_v2.py (sr-series convention:
#               no cross-script imports). silver_tier is recomputed locally, not
#               read from the ratings CSV (that column predates ADR-06). PERSON
#               spans restricted to f_resp_person=1, computed here for the first
#               time and validated only against heuristic-tier SoRs (see
#               notes/fiz/bibner-werk/A2-silver-spans-script-plan.md §8.2).

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_SAMPLE_DEFAULT = ROOT / "data" / "annotation" / "sr08_gold_sample.csv"
OUTPUT_DEFAULT = ROOT / "data" / "processed" / "silver_spans.jsonl"
STATS_OUTPUT_DEFAULT = ROOT / "data" / "processed" / "sr_silver_spans_stats.csv"
JOIN_FAILURES_OUTPUT_DEFAULT = ROOT / "data" / "processed" / "sr_silver_spans_join_failures.csv"

TIER2_BASELINE = 4_613  # pre-ADR-06 count (ADR-04), for comparison only

# ---------------------------------------------------------------------------
# ISBD boundary markers
# ---------------------------------------------------------------------------

AREA_SEP = ". - "
OTHER_TITLE_MARK = " :"
PERSON_MARK = " /"
SUB_DELIM_RE = re.compile(r";")

# ---------------------------------------------------------------------------
# f_resp_* sub-classifier (ADR-06)
# ---------------------------------------------------------------------------
#
# PERSON vs. ORG is decided by flair/ner-german-large (FLERT: Schweter & Akbik,
# 2020 — document-level contextual embeddings over XLM-R), not a keyword list.
# The original keyword-only design (RESP_ORG_RE) measured only 41% PERSON
# precision on the SR-04 100-record sample (verify_resp_classifier.py) —
# open-vocabulary person/institution name detection has no fixed keyword list
# to match against. EDITOR/FAMILY are role qualifiers, not NER categories
# FLERT tags, so those still use keyword regexes, and are checked *before*
# the FLERT PER/ORG tags: an editor's own name still gets tagged PER, but the
# correct f_resp_* class is f_resp_editor, not f_resp_person. See
# docs/bibner-adr.md D-04 "Implementation update (2026-07-06)".

RESP_EDITOR_RE = re.compile(
    # (Hg.) is handled as a separate alternative, not inside the \b...\b group:
    # a leading \b can never match right before '(' when the preceding
    # character is whitespace — both sides of that position are non-word
    # characters, so no \w/\W transition exists there (confirmed via
    # verify_resp_classifier.py: "Laubach Jakob (Hg.)" silently fell through
    # to f_resp_person until this was split out).
    r"(?i)\b(?:Hrsg\.?|herausgegeben|bearb\w*|edited by|zusammengestellt)\b|\(Hg\.\)"
)
RESP_FAMILY_RE = re.compile(r"(?i)\b(?:Familie|Nachlass)\b")

# Non-SoR false positives (ADR-06 background, SR-04's 41% finding): bare series/
# volume letters or numerals ("/ K", "/ Bd. 3"), roman numerals, single-token
# fragments. Cheap pre-filter to skip FLERT entirely for obvious non-names.
RESP_NON_SOR_RE = re.compile(r"^\s*(?:[IVXLCDM]{1,4}\.?|\d{1,4}\.?|[A-Za-zÄÖÜäöü]\.?)\s*$")

_TAGGER = None  # lazy-loaded singleton; loading flair/ner-german-large takes real time

# ---------------------------------------------------------------------------
# Era bucketing — replicated from scripts/ner/sr10_era_length_summary_v2.py
# (sr-series convention: no cross-script imports).
# ---------------------------------------------------------------------------

YEAR_RE = re.compile(r"\b(?:1[4-9]\d{2}|20[012]\d)\b")


def year_from_title(title: str) -> int | None:
    """Return the last 4-digit year matched in the title string, if any."""
    matches = list(YEAR_RE.finditer(str(title)))
    return int(matches[-1].group()) if matches else None


def resolve_year(dates_value: object, title: str) -> int | None:
    """Prefer the parquet's 'dates' column; fall back to a title-regex year match."""
    if pd.notna(dates_value) and str(dates_value).strip():
        try:
            year = int(str(dates_value).strip()[:4])
            if 1400 <= year <= 2029:
                return year
        except ValueError:
            pass
    return year_from_title(title)


def assign_era(year: int | None) -> str:
    """Bucket a resolved year into pre-1700 / 1700-1800 / 19th-c / modern / unknown.

    Checks both None and NaN: a Series built from a mix of int and None
    (as `merged["year"]` is, via resolve_year()) gets silently upcast by
    pandas to float64 with NaN standing in for the missing values — `NaN is
    None` is False, so a None-only check here would fall through the
    numeric comparisons below (all of which are False against NaN) and
    misreport every undated record as "modern" instead of "unknown". This
    exact bug was live in this file until caught via the era-join-failures
    log (see A2-silver-spans-script-plan.md §2.1) — the original source,
    sr10_era_length_summary_v2.py:63, already guards against both.
    """
    if year is None or pd.isna(year):
        return "unknown"
    if year < 1700:
        return "pre-1700"
    if year < 1800:
        return "1700-1800"
    if year < 1900:
        return "19th-c"
    return "modern"


# ---------------------------------------------------------------------------
# f_resp_* classification
# ---------------------------------------------------------------------------


def extract_sor(title: str) -> str:
    """Return the text after the first ' /' in the title, or '' if absent."""
    idx = title.find(PERSON_MARK)
    return title[idx + len(PERSON_MARK):].strip() if idx != -1 else ""


def get_tagger():
    """Lazily load the FLERT German NER tagger (flair/ner-german-large), once."""
    global _TAGGER
    if _TAGGER is None:
        from flair.models import SequenceTagger
        logging.info("Loading flair/ner-german-large ...")
        _TAGGER = SequenceTagger.load("flair/ner-german-large")
    return _TAGGER


def resp_entity_labels(sor_texts: list[str], tagger: object) -> list[set[str]]:
    """Batch-tag SoR texts; return the set of NER tag values found in each
    (e.g. {'PER'}, {'ORG'}, {'PER', 'ORG'}, or set() if neither is found)."""
    from flair.data import Sentence
    sentences = [Sentence(text) for text in sor_texts]
    if sentences:
        tagger.predict(sentences, mini_batch_size=64)
    return [
        {span.get_label("ner").value for span in sentence.get_spans("ner")}
        for sentence in sentences
    ]


def classify_resp_batch(sor_texts: list[str], tagger: object | None = None) -> list[str]:
    """Sub-classify a batch of statement-of-responsibility segments (ADR-06).

    EDITOR/FAMILY keyword checks run first (role qualifiers, not NER
    categories); PERSON vs. ORG is then decided by FLERT tags for whatever
    remains. Records that get neither an EDITOR/FAMILY keyword match nor a
    PER/ORG tag stay f_resp_other (the safe default — see module header).
    """
    if tagger is None:
        tagger = get_tagger()

    classes = ["f_resp_other"] * len(sor_texts)
    ner_pending: list[int] = []
    for i, text in enumerate(sor_texts):
        stripped = text.strip()
        if not stripped or RESP_NON_SOR_RE.match(stripped):
            continue
        if RESP_EDITOR_RE.search(text):
            classes[i] = "f_resp_editor"
            continue
        if RESP_FAMILY_RE.search(text):
            classes[i] = "f_resp_family"
            continue
        ner_pending.append(i)

    if ner_pending:
        label_sets = resp_entity_labels([sor_texts[i] for i in ner_pending], tagger)
        for i, labels in zip(ner_pending, label_sets):
            if "ORG" in labels:
                classes[i] = "f_resp_org"
            elif "PER" in labels:
                classes[i] = "f_resp_person"
    return classes


def classify_resp(sor_text: str, tagger: object | None = None) -> str:
    """Single-text convenience wrapper around classify_resp_batch (see there)."""
    return classify_resp_batch([sor_text], tagger=tagger)[0]


RESP_CLASSES = (
    "f_resp_person",
    "f_resp_org",
    "f_resp_family",
    "f_resp_editor",
    "f_resp_other",
)


def compute_resp_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add f_resp_* columns, computed only for records where f_person=1.

    Restricted to the f_person=1 subset (~34K of 9.21M rows) rather than the
    full corpus — the SoR classification (keyword + FLERT) is undefined
    without a SoR to classify, and batching only this subset through FLERT
    avoids running the tagger over 9.21M rows where the result is always 0.
    """
    df = df.copy()
    for col in RESP_CLASSES:
        df[col] = 0

    person_mask = df["f_person"] == 1
    sor_texts = df.loc[person_mask, "title"].apply(extract_sor).tolist()
    classes = pd.Series(classify_resp_batch(sor_texts), index=df.index[person_mask])
    for cls in RESP_CLASSES:
        df.loc[person_mask, cls] = (classes == cls).astype(int)
    return df


# ---------------------------------------------------------------------------
# Tier recomputation (ADR-06 amends ADR-04's tier-2 criterion)
# ---------------------------------------------------------------------------


def to_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV-loaded column to bool regardless of how pandas inferred it."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def recompute_silver_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute silver_tier locally rather than trust the ratings CSV's column.

    sr01_rate_isbd_fields.py's tier-2 test uses f_person, not f_resp_person —
    it predates ADR-06's amendment. Tier-1's formula is left unchanged; ADR-06
    only mandates tightening tier-2.
    """
    df = df.copy()
    df["has_dot_dash"] = to_bool(df["has_dot_dash"])

    manifestation = df[["f_edition", "f_place", "f_publisher", "f_year", "f_series"]].any(axis=1)
    tier2 = df["has_dot_dash"] & (df["f_resp_person"] == 1) & manifestation
    tier1 = ~tier2 & ((df["n_fields"] >= 3) | ((df["f_person"] == 1) & (df["f_year"] == 1)))

    df["silver_tier"] = 0
    df.loc[tier1, "silver_tier"] = 1
    df.loc[tier2, "silver_tier"] = 2
    return df


# ---------------------------------------------------------------------------
# Era join
# ---------------------------------------------------------------------------


def write_join_failures(failures: pd.DataFrame, path: Path) -> None:
    """Log every ratings/parquet join failure with enough context to investigate later.

    Not fatal (see era_join) — these records still get processed normally via
    the title-regex year fallback. This file exists so a future drift between
    sr01_rate_isbd_fields.py and tokenize_de_titles.py's outputs (the two
    files were verified 1:1 as of 2026-07-06, see module header) is visible
    and re-examinable without re-running the full pipeline, not so it blocks
    anything.
    """
    cols = [c for c in ("obj_id", "title", "silver_tier", "has_dot_dash",
                        "f_person", "f_resp_person", "n_fields", "year", "era")
            if c in failures.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    failures[cols].assign(reason="no matching obj_id in tokenized parquet").to_csv(path, index=False)
    logging.warning("Era join: %d records had no parquet match — logged to %s", len(failures), path)


def era_join(df: pd.DataFrame, parquet_path: Path, failures_path: Path) -> pd.DataFrame:
    """Left-join df against the tokenized parquet's 'dates' column, by obj_id.

    Verified 1:1 as of 2026-07-06 (see module header), but an unmatched
    obj_id is not treated as fatal: resolve_year() falls back to a
    title-regex year match when the parquet's 'dates' column is unavailable
    (dates=NaN just skips straight to that path), so the record can still be
    era-bucketed and processed normally. Every unmatched obj_id is still
    logged to failures_path (write_join_failures) with enough detail to
    investigate later, rather than silently disappearing.
    """
    logging.info("Loading dates from %s", parquet_path)
    dates_df = pd.read_parquet(parquet_path, columns=["obj_id", "dates"])
    merged = df.merge(dates_df, on="obj_id", how="left", indicator=True)

    unmatched = merged["_merge"] == "left_only"
    logging.info("Era join: %d/%d obj_ids unmatched", int(unmatched.sum()), len(merged))

    merged["year"] = merged.apply(lambda r: resolve_year(r["dates"], r["title"]), axis=1)
    merged["era"] = merged["year"].apply(assign_era)

    if unmatched.any():
        write_join_failures(merged[unmatched], failures_path)

    return merged.drop(columns=["_merge", "dates", "year"])


# ---------------------------------------------------------------------------
# Span extraction
# ---------------------------------------------------------------------------


def title_area_end(text: str, has_dot_dash: bool) -> int:
    """End offset (exclusive) of the ISBD title area: up to '. - ' if structural."""
    if not has_dot_dash:
        return len(text)
    pos = text.find(AREA_SEP)
    return pos if pos != -1 else len(text)


LEADING_STRIP_CHARS = " \t/:;,"  # stray delimiter fragments left by doubled marks (see below)


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip whitespace/trailing '.' and leading stray delimiter characters.

    Doubled delimiters in the source text (musical-manuscript '//' notation,
    or ' : /' sequences) leave one extra delimiter character as a leading
    artifact after the normal single-mark skip in extract_spans() — found via
    a full-corpus spot-check after the first production run (150/91,791
    spans, 0.16%, e.g. PERSON span '/ Cantilena in C...' or OTHER_TITLE span
    '/, Splanchnologische Tabellen'). Broadly stripping leading whitespace/
    slash/colon/semicolon/comma is safe: none of TITLE/OTHER_TITLE/PERSON text
    legitimately starts with a bare separator character.
    """
    span_text = text[start:end]
    left_trimmed = span_text.lstrip(LEADING_STRIP_CHARS)
    new_start = start + (len(span_text) - len(left_trimmed))
    trimmed = left_trimmed.rstrip()
    if trimmed.endswith("."):
        trimmed = trimmed[:-1].rstrip()
    return new_start, new_start + len(trimmed)


def extract_spans(
    text: str, has_dot_dash: bool, f_other_title: bool, f_resp_person: bool
) -> list[tuple[int, int, str]]:
    """Derive TITLE / OTHER_TITLE / PERSON char-offset spans from ISBD boundary markers.

    Boundary markers are only searched within the title area for structural
    records (has_dot_dash=True), matching how f_other_title/f_person were
    computed by sr01_rate_isbd_fields.py's rate(). Boundaries are walked in
    sorted position order rather than assuming ' :' precedes ' /' — real
    records occasionally have the reverse order, and a fixed-order
    implementation produces overlapping OTHER_TITLE/PERSON spans in that case
    (caught during smoke-testing via the Layer-1 overlap check, plan §8.1).
    """
    area_end = title_area_end(text, has_dot_dash)
    area = text[:area_end]

    other_pos = area.find(OTHER_TITLE_MARK) if f_other_title else -1
    person_pos = area.find(PERSON_MARK) if f_resp_person else -1

    boundaries = sorted(
        (pos, mark_len, label)
        for pos, mark_len, label in (
            (other_pos, len(OTHER_TITLE_MARK), "OTHER_TITLE"),
            (person_pos, len(PERSON_MARK), "PERSON"),
        )
        if pos != -1
    )

    title_end = boundaries[0][0] if boundaries else area_end
    spans = [(0, title_end, "TITLE")]

    for i, (pos, mark_len, label) in enumerate(boundaries):
        start = pos + mark_len
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else area_end
        if label == "PERSON":
            sub_match = SUB_DELIM_RE.search(area, start, end)
            if sub_match:
                end = sub_match.start()
        spans.append((start, end, label))

    trimmed = [(*trim_span(text, s, e), label) for s, e, label in spans]
    return [(s, e, label) for s, e, label in trimmed if e > s]


# ---------------------------------------------------------------------------
# Tier-1 era-stratified subsampling
# ---------------------------------------------------------------------------


def sample_tier1(df: pd.DataFrame, sample_size: int, mode: str, seed: int) -> pd.DataFrame:
    """Draw an era-stratified subsample of eligible tier-1 records."""
    if len(df) <= sample_size:
        logging.info("Tier-1 pool (%d) <= sample size (%d); keeping all", len(df), sample_size)
        return df

    era_counts = df["era"].value_counts()
    if mode == "balanced":
        per_era = sample_size // len(era_counts)
        targets = {era: min(per_era, n) for era, n in era_counts.items()}
    else:
        targets = {era: round(sample_size * n / len(df)) for era, n in era_counts.items()}

    sampled = [
        group.sample(n=min(targets[era], len(group)), random_state=seed)
        for era, group in df.groupby("era")
    ]
    return pd.concat(sampled, ignore_index=True)


# ---------------------------------------------------------------------------
# Record assembly and output
# ---------------------------------------------------------------------------


def build_records(df: pd.DataFrame) -> list[dict]:
    """Assemble Doccano-schema records: {"id", "text", "label": [[start,end,LABEL],...],
    "silver_tier", "era"}.

    silver_tier (1 or 2) and era are included so downstream consumers (e.g. A5's
    fine-tuning scripts, which need both for era x silver_tier-stratified splits)
    can use them directly without re-deriving — df already carries both columns
    from recompute_silver_tier()/era_join(), they just weren't being written out
    before.
    """
    records = []
    for row in df.itertuples():
        spans = extract_spans(
            row.title, bool(row.has_dot_dash), bool(row.f_other_title), bool(row.f_resp_person)
        )
        records.append({
            "id": row.obj_id,
            "text": row.title,
            "label": [[s, e, label] for s, e, label in spans],
            "silver_tier": int(row.silver_tier),
            "era": row.era,
        })
    return records


def load_gold_ids(path: Path) -> set[str]:
    """Load the DDB obj_id set to exclude from silver training data.

    sr08_gold_sample.csv's identifier column is 'obj_id' (the corpus-wide DDB
    ID, matching the ratings CSV) — not 'id' (Doccano's own sequential
    1-395 identifier, a different namespace used only in the annotator
    export JSONL files, data/inputs.md). Confirmed directly against the real
    file (2026-07-06): the smoke-test fixtures used a synthetic CSV with a
    column literally named 'id', which masked this mismatch entirely.
    """
    gold = pd.read_csv(path, dtype=str)
    return set(gold["obj_id"])


def verify_records(records: list[dict], gold_ids: set[str]) -> None:
    """Layer-1 mechanical checks (plan §8.1). Fails loudly before anything is written."""
    seen_ids: set[str] = set()
    for rec in records:
        if rec["id"] in seen_ids:
            raise ValueError(f"Duplicate id in output: {rec['id']}")
        seen_ids.add(rec["id"])

        if rec["id"] in gold_ids:
            raise ValueError(f"Gold id leaked into silver output: {rec['id']}")

        labels = {label for _, _, label in rec["label"]}
        if "TITLE" not in labels:
            raise ValueError(f"Record {rec['id']} has no TITLE span")

        spans_sorted = sorted(rec["label"], key=lambda s: s[0])
        for i in range(len(spans_sorted) - 1):
            if spans_sorted[i][1] > spans_sorted[i + 1][0]:
                raise ValueError(f"Overlapping spans in record {rec['id']}: {rec['label']}")

        for start, end, _ in rec["label"]:
            if not (0 <= start < end <= len(rec["text"])):
                raise ValueError(f"Invalid span bounds in record {rec['id']}: {(start, end)}")


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("Wrote %d records -> %s", len(records), path)


def write_stats(df: pd.DataFrame, records: list[dict], path: Path) -> None:
    """Write tier x era x label prevalence counts, for citing in 40-silverandgold.tex §4.1."""
    label_by_id = {rec["id"]: {label for _, _, label in rec["label"]} for rec in records}
    rows = []
    for tier, era_group in df.groupby("silver_tier"):
        for era, era_records in era_group.groupby("era"):
            labels_in_stratum = [label_by_id[obj_id] for obj_id in era_records["obj_id"]]
            for label in ("TITLE", "OTHER_TITLE", "PERSON"):
                count = sum(1 for labels in labels_in_stratum if label in labels)
                rows.append({
                    "silver_tier": tier, "era": era, "label": label,
                    "n_records": len(labels_in_stratum), "n_with_label": count,
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logging.info("Wrote stats -> %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Stage 5: build silver NER spans from ISBD field-flag ratings."
    )
    parser.add_argument("--ratings", type=Path, required=True,
                        help="Path to sr01_isbd_field_ratings.csv")
    parser.add_argument("--parquet", type=Path, required=True,
                        help="Path to de_titles_tokenized.parquet (for era join)")
    parser.add_argument("--gold-sample", type=Path, default=GOLD_SAMPLE_DEFAULT,
                        help="Path to sr08_gold_sample.csv (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT,
                        help="Output JSONL path (default: %(default)s)")
    parser.add_argument("--stats-output", type=Path, default=STATS_OUTPUT_DEFAULT,
                        help="Output stats CSV path (default: %(default)s)")
    parser.add_argument("--join-failures-output", type=Path, default=JOIN_FAILURES_OUTPUT_DEFAULT,
                        help="Era-join failures CSV path (default: %(default)s)")
    parser.add_argument("--tier1-sample-size", type=int, default=50_000,
                        help="Max tier-1 records to keep (default: %(default)s)")
    parser.add_argument("--stratify-mode", choices=["proportional", "balanced"],
                        default="proportional", help="Tier-1 era sampling mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.info("Loading ratings from %s", args.ratings)
    ratings = pd.read_csv(args.ratings, low_memory=False)

    logging.info("Computing f_resp_* sub-classification (ADR-06) ...")
    ratings = compute_resp_flags(ratings)

    logging.info("Recomputing silver_tier locally (ADR-06 tier-2 tightening) ...")
    ratings = recompute_silver_tier(ratings)

    tier2_n = int((ratings["silver_tier"] == 2).sum())
    logging.info("Recomputed tier-2: %d records (pre-ADR-06 baseline: %d)", tier2_n, TIER2_BASELINE)
    if tier2_n > TIER2_BASELINE:
        raise RuntimeError(
            f"Recomputed tier-2 ({tier2_n}) is larger than the pre-ADR-06 baseline "
            f"({TIER2_BASELINE}) — the tightened criterion should only ever shrink tier-2."
        )

    eligible = ratings[ratings["silver_tier"].isin([1, 2])].copy()
    logging.info("Eligible tier-1/2 pool: %d records", len(eligible))

    gold_ids = load_gold_ids(args.gold_sample)
    eligible = eligible[~eligible["obj_id"].isin(gold_ids)].copy()
    logging.info("After gold exclusion: %d records", len(eligible))

    eligible = era_join(eligible, args.parquet, args.join_failures_output)

    tier2_df = eligible[eligible["silver_tier"] == 2]
    tier1_df = eligible[eligible["silver_tier"] == 1]
    tier1_sampled = sample_tier1(tier1_df, args.tier1_sample_size, args.stratify_mode, args.seed)
    final_df = pd.concat([tier2_df, tier1_sampled], ignore_index=True)
    logging.info(
        "Final selection: %d tier-2 + %d tier-1 = %d records",
        len(tier2_df), len(tier1_sampled), len(final_df),
    )

    records = build_records(final_df)
    verify_records(records, gold_ids)
    logging.info("Layer-1 verification passed (%d records)", len(records))

    write_jsonl(records, args.output)
    write_stats(final_df, records, args.stats_output)


if __name__ == "__main__":
    main()
