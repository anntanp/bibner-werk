#!/usr/bin/env python3
# Purpose:      Read-only scoping analysis: how many of gold's 130 PERSON spans carry an
#               honorific/titulature tail beyond a bare personal name, and how long is
#               that tail typically? Does NOT modify data/gold/gold_reconciled.jsonl --
#               produces a candidate-review CSV only, for a human to inspect and decide
#               trim points on. Precedes any decision to re-annotate gold (see
#               B6-gliner-eval.md §6, "it doesn't make sense to include the honorifics").
# Usage:        python3 scripts/py/scope_person_honorific_review.py
# Inputs:       data/processed/b6_person_error_analysis.csv (fn+tp rows together already
#                   cover all 130 gold PERSON spans -- folds partition the full 395-record
#                   gold set, so no direct read of data/gold/ is needed for this scoping pass)
# Outputs:      data/processed/person_honorific_scope.csv (doc_id, span_text, span_len,
#                   has_honorific_marker, suggested_core_name -- a candidate, not a decision)
# Dependencies: none (stdlib only)
# Assumptions:  "has_honorific_marker" is a keyword heuristic (titles, offices, dominion
#                   language), not a verified judgment -- every flagged row still needs
#                   human review before any gold file is touched.
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = ROOT / "data" / "processed" / "b6_person_error_analysis.csv"
OUTPUT = ROOT / "data" / "processed" / "person_honorific_scope.csv"

# German honorific / office / dominion-listing markers -- a span containing any of these
# is a candidate for carrying titulature beyond a bare personal name.
HONORIFIC_KEYWORDS = [
    "von Gottes Gnaden", "König", "Königin", "Kaiser", "Kaiserin", "Herzog", "Hertzog",
    "Landgraf", "Landgrauen", "Grauen", "Graf ", "Churfürst", "Kurfürst", "Bischof",
    "Erzbischof", "Prälat", "Präsident", "Profess", "Superint", "Past.", "Pastor",
    "Consistorii", "Geheimen", "Justitz-Rath", "Hofrat", "Rektor", "Dekan", "Ritter",
    "Beschützer", "Groß-Hertzog", "Burggrauen", "Assess.", "Inspector", "Lehrer",
]
HONORIFIC_RE = re.compile("|".join(re.escape(kw) for kw in HONORIFIC_KEYWORDS), re.IGNORECASE)

CORE_NAME_LEN_THRESHOLD = 25  # spans at or under this length are treated as already core-name-like


def has_honorific_marker(span_text: str) -> bool:
    return bool(HONORIFIC_RE.search(span_text))


def suggested_core_name(span_text: str) -> str:
    match = HONORIFIC_RE.search(span_text)
    if match:
        return span_text[: match.start()].rstrip(" ,;:")
    return span_text


def main() -> None:
    rows = list(csv.DictReader(open(INPUT, encoding="utf-8")))
    gold_person = {(r["doc_id"], r["span_text"]) for r in rows if r["outcome"] in ("fn", "tp")}

    out_rows = []
    for doc_id, span_text in sorted(gold_person):
        span_len = len(span_text)
        flagged = has_honorific_marker(span_text)
        out_rows.append({
            "doc_id": doc_id,
            "span_text": span_text,
            "span_len": span_len,
            "has_honorific_marker": flagged,
            "suggested_core_name": suggested_core_name(span_text) if flagged else span_text,
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "span_text", "span_len",
                                                "has_honorific_marker", "suggested_core_name"])
        writer.writeheader()
        writer.writerows(out_rows)

    total = len(out_rows)
    flagged_n = sum(1 for r in out_rows if r["has_honorific_marker"])
    long_n = sum(1 for r in out_rows if r["span_len"] > CORE_NAME_LEN_THRESHOLD)
    long_and_flagged = sum(1 for r in out_rows if r["span_len"] > CORE_NAME_LEN_THRESHOLD and r["has_honorific_marker"])
    long_not_flagged = [r for r in out_rows if r["span_len"] > CORE_NAME_LEN_THRESHOLD and not r["has_honorific_marker"]]

    print(f"Total gold PERSON spans: {total}")
    print(f"Flagged by honorific keyword: {flagged_n} ({flagged_n/total*100:.0f}%)")
    print(f"Longer than {CORE_NAME_LEN_THRESHOLD} chars: {long_n} ({long_n/total*100:.0f}%)")
    print(f"Both long and flagged: {long_and_flagged}")
    print(f"Long but NOT keyword-flagged (needs manual look): {len(long_not_flagged)}")
    for r in long_not_flagged[:15]:
        print(f"  [{r['doc_id']}] ({r['span_len']} chars) {r['span_text']!r}")
    print(f"\nWrote candidate review file -> {OUTPUT}")


if __name__ == "__main__":
    main()
