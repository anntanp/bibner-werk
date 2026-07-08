#!/usr/bin/env python3
# Purpose:      Build a Doccano import file for the blind anchoring-bias check
#               (notes/fiz/bibner-werk/adj-llm-justification.md §2/§3): a stratified
#               random sample of tier-3 docs, re-presented with the same
#               per-annotator disagreement diff as the original adjudication pass
#               but WITHOUT the LLM's suggested label, confidence, or rationale.
# Usage:        python scripts/ner/sr08_build_blind_sample.py \
#                   --sheet data/adjudication/sr08_adjudication_sheet_export-20260623-1110.jsonl \
#                   --acceptance-csv data/adjudication/sr08_llm_acceptance_export-20260623-1110.csv \
#                   --htype-csv ~/Documents/claude/gemea/data/schema/ddbedm-htype.csv \
#                   --n-per-stratum 6 --seed 42
# Inputs:       --sheet: tier-3 canonical sheet (doc_id, text, conflict_type,
#                   annotations, comments, llm_confidence)
#               --acceptance-csv: sr08_compute_llm_acceptance.py's per-doc output
#                   (used only for contested_ref/precedent_ref, to exclude docs
#                   already discussed by name in sr08_adjudication-contested-tokens.md /
#                   adj-precedents.md — those are already anchored in the PI's
#                   written notes, independent of any LLM-suggestion effect)
#               --sample-csv (htype/url meta; row index+1 == doc_id; default:
#                   data/annotation/sr08_gold_sample.csv)
#               --htype-csv (htype code/label lookup; lives in gemea, not this repo,
#                   so no hardcoded default — pass explicitly, e.g.
#                   ~/Documents/claude/gemea/data/schema/ddbedm-htype.csv)
# Outputs:      data/adjudication/sr08_blind_sample_import_<export>.jsonl — Doccano
#                   import file: {text, label: [], Comments: [neutral marker line,
#                   then one line per reviewing annotator], meta: {htype, url}}.
#                   Row order is shuffled (seeded) — not doc_id order, not the
#                   original adjudication pass's order.
#               data/adjudication/sr08_blind_sample_key_<export>.csv — reconciliation
#                   key (NOT for Doccano import): blind_order, doc_id, conflict_type,
#                   llm_confidence, contested_ref, precedent_ref. Needed after the PI
#                   completes the blind review to join each blind decision back to its
#                   doc_id and original (LLM-assisted) decision.
# Dependencies: pandas (only for the two lookup CSVs, matching sr08_add_meta_to_export.py)
# Assumptions:  Doccano row order == import order for a fresh project, so the
#               reconciliation key's blind_order (0-indexed) is the only way to map a
#               completed blind-review export back to doc_id — Doccano does not echo
#               back any field from this script's output other than text/label/Comments/meta.
#               "Reviewing" = non-empty spans or a comment, same definition used
#               throughout the sr08 series (sr08_compute_iaa.py's get_labels()).
#               Comments carry the same per-annotator diff format as
#               sr08_adjudication_to_doccano.py (duplicated here, not imported —
#               no sr08 script imports another), but sr08_llm_acceptance_<export>.csv
#               must exist first (run sr08_compute_llm_acceptance.py before this script).

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]

ANNOTATOR_ORDER = ["frankp", "gundi", "maria"]
CONFLICT_ORDER = ["scope", "boundary", "three_way"]
NEUTRAL_REASON_LINE = "Blind adjudication check -- annotator disagreement below, no suggestion shown."

MONOGRAPH_FALLBACK = "htype_021 (Monografie)"


# ── Comments diff (duplicated from sr08_adjudication_to_doccano.py — no sr08
#    script imports another) ──────────────────────────────────────────────────

def build_annotator_line(annotator: str, text: str, spans: list, comment: str | None) -> str | None:
    reviewing = bool(spans) or bool(comment)
    if not reviewing:
        return None
    if not spans:
        return f"{annotator}: O[Grund: '{comment}']"
    parts = [f"{label}['{text[start:end]}']" for start, end, label in spans]
    if comment:
        parts.append(f"COMMENT['{comment}']")
    return f"{annotator}: " + ", ".join(parts)


def build_comments(text: str, annotations: dict, comments: dict) -> list:
    out = [NEUTRAL_REASON_LINE]
    for annotator in ANNOTATOR_ORDER:
        spans = annotations.get(annotator, [])
        comment = comments.get(annotator)
        line = build_annotator_line(annotator, text, spans, comment)
        if line is not None:
            out.append(line)
    return out


# ── Meta (htype + DDB URL), same lookup as sr08_add_meta_to_export.py ────────

def build_htype_lookup(path: Path) -> dict:
    df = pd.read_csv(path)
    return {row["label_de"]: f"{row['htype_code']} ({row['label_de']})"
            for _, row in df.iterrows()}


def derive_htype(type_str, lookup: dict):
    if not isinstance(type_str, str):
        return None
    last = type_str.split("|")[-1].strip()
    if last == "Monograph":
        return MONOGRAPH_FALLBACK
    return lookup.get(last)


def build_meta_by_doc_id(sample_csv: Path, htype_csv: Path) -> dict[int, dict]:
    """doc_id (1-indexed) -> {htype, url}; row index + 1 == doc_id in sample_csv."""
    lookup = build_htype_lookup(htype_csv)
    sample_df = pd.read_csv(sample_csv).reset_index(drop=True)
    return {
        i + 1: {"htype": derive_htype(row["dc_type"], lookup), "url": row["ddb_link"]}
        for i, row in sample_df.iterrows()
    }


# ── Sampling ──────────────────────────────────────────────────────────────────

def load_sheet(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_excluded_doc_ids(acceptance_csv: Path) -> set[int]:
    """doc_ids with a contested:N or precedent:N cross-ref — already discussed by
    name in written notes, so excluded from the blind sample (see header)."""
    excluded = set()
    with open(acceptance_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["contested_ref"] or row["precedent_ref"]:
                excluded.add(int(row["doc_id"]))
    return excluded


def stratified_sample(rows: list[dict], excluded: set[int], n_per_stratum: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    eligible = [r for r in rows if r["doc_id"] not in excluded]
    by_stratum: dict[str, list[dict]] = {c: [] for c in CONFLICT_ORDER}
    for r in eligible:
        by_stratum.setdefault(r["conflict_type"], []).append(r)

    sample: list[dict] = []
    for c in CONFLICT_ORDER:
        pool = by_stratum.get(c, [])
        k = min(n_per_stratum, len(pool))
        sample.extend(rng.sample(pool, k))

    rng.shuffle(sample)
    return sample


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Doccano import file for the blind anchoring-bias check"
    )
    parser.add_argument("--sheet", required=True, type=Path,
                         help="Tier-3 canonical sheet (sr08_adjudication_sheet_<export>.jsonl)")
    parser.add_argument("--acceptance-csv", required=True, type=Path,
                         help="sr08_compute_llm_acceptance.py per-doc output (for contested/precedent exclusion)")
    parser.add_argument("--n-per-stratum", type=int, default=6,
                         help="Docs sampled per conflict_type stratum (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--sample-csv", type=Path, default=BASE / "data/annotation/sr08_gold_sample.csv",
                         help="Gold sample CSV for htype/url meta (default: data/annotation/sr08_gold_sample.csv)")
    parser.add_argument("--htype-csv", required=True, type=Path,
                         help="htype code/label lookup CSV (lives in gemea, e.g. "
                              "~/Documents/claude/gemea/data/schema/ddbedm-htype.csv)")
    parser.add_argument("--output-dir", type=Path, default=BASE / "data/adjudication",
                         help="Output directory (default: data/adjudication/)")
    args = parser.parse_args()

    rows = load_sheet(args.sheet)
    excluded = load_excluded_doc_ids(args.acceptance_csv)
    sample = stratified_sample(rows, excluded, args.n_per_stratum, args.seed)
    meta_by_doc_id = build_meta_by_doc_id(args.sample_csv, args.htype_csv)

    m = re.search(r"sr08_adjudication_sheet_(.+)\.jsonl$", args.sheet.name)
    export_stem = m.group(1) if m else args.sheet.stem

    import_path = args.output_dir / f"sr08_blind_sample_import_{export_stem}.jsonl"
    key_path = args.output_dir / f"sr08_blind_sample_key_{export_stem}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(import_path, "w", encoding="utf-8") as f_import, \
            open(key_path, "w", newline="", encoding="utf-8") as f_key:
        key_writer = csv.DictWriter(
            f_key, fieldnames=["blind_order", "doc_id", "conflict_type", "llm_confidence"]
        )
        key_writer.writeheader()

        for blind_order, rec in enumerate(sample):
            doc_id = rec["doc_id"]
            text = rec["text"]
            comments = build_comments(text, rec["annotations"], rec["comments"])
            out = {
                "text": text,
                "label": [],
                "Comments": comments,
                "meta": meta_by_doc_id.get(doc_id, {"htype": None, "url": None}),
            }
            f_import.write(json.dumps(out, ensure_ascii=False) + "\n")
            key_writer.writerow({
                "blind_order": blind_order,
                "doc_id": doc_id,
                "conflict_type": rec["conflict_type"],
                "llm_confidence": rec["llm_confidence"],
            })

    by_stratum_count: dict[str, int] = {}
    for rec in sample:
        by_stratum_count[rec["conflict_type"]] = by_stratum_count.get(rec["conflict_type"], 0) + 1

    print(f"Doccano import ({len(sample)} docs, seed={args.seed}) -> {import_path}")
    print(f"Reconciliation key -> {key_path}")
    print(f"Excluded (contested/precedent refs): {len(excluded)} docs")
    print("Sample composition:", by_stratum_count)


if __name__ == "__main__":
    main()
