#!/usr/bin/env python3
# Purpose:      Classify each BibNER-Werk gold document into an adjudication tier
#               (1 auto-accept, 2 majority-auto, 3 PI-adjudicated), per
#               notes/fiz/bibner-werk/adjudication-plan.md §1-§2.1
# Usage:        python scripts/ner/sr08_classify_disagreements.py \
#                   --export-dir data/annotation/doccano/export-20260623-1110
# Inputs:       *.jsonl files in --export-dir (skips dummy.doccano.jsonl, ann.jsonl,
#               harald.jsonl, *.meta.jsonl, sr08_*)
# Outputs:      data/processed/ner/sr08_disagreement_classification_<export>.csv
#               (doc_id, tier, conflict_type, annotators_reviewed)
#               stdout: tier counts, cross-checked against adjudication-plan.md §1
# Dependencies: none (stdlib only)
# Assumptions:  reviewed = label != [] OR Comments != []; scope conflict = any
#               reviewing annotator's Comments mention Kapitel/Abschnitt/Aufsatz.
#               Classification logic is self-contained (not imported from
#               sr08_compute_iaa.py) — no sr08 script imports another; see
#               adjudication-plan.md §2.1 for the deviation from the original plan text.

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

SKIP_NAMES    = {"dummy.doccano.jsonl", "ann.jsonl", "harald.jsonl"}
SKIP_SUFFIXES = (".meta.jsonl",)
SKIP_PREFIXES = ("sr08_",)

SCOPE_KEYWORDS = ("kapitel", "abschnitt", "aufsatz")

# adjudication-plan.md §1 estimate, computed against export-20260623-1110 —
# printed alongside the real counts as a sanity check, not enforced.
PLAN_ESTIMATE = {
    "tier1": 42,
    "tier2": 50,
    "scope": 142,
    "boundary": 138,
    "three_way": 23,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def is_reviewed(record: dict) -> bool:
    return bool(record["label"]) or bool(record.get("Comments"))


def is_scope_comment(record: dict) -> bool:
    for c in record.get("Comments", []):
        text = (c.get("text", "") if isinstance(c, dict) else str(c)).lower()
        if any(k in text for k in SCOPE_KEYWORDS):
            return True
    return False


def spans_by_label(record: dict) -> dict[str, set[tuple[int, int]]]:
    """Group spans by label type; identical (start, end) pairs collapse to one,
    matching sr08_compute_iaa.py's dedup-by-type rule. Empty dict = O."""
    d: defaultdict[str, set[tuple[int, int]]] = defaultdict(set)
    for start, end, lbl in record["label"]:
        d[lbl].add((start, end))
    return dict(d)


def load_all_records(export_dir: Path) -> dict[str, dict[int, dict]]:
    """Return {annotator: {doc_id: record}} for every record, reviewed or not."""
    data: dict[str, dict[int, dict]] = {}
    for path in sorted(export_dir.glob("*.jsonl")):
        name = path.name
        if (name in SKIP_NAMES
                or any(name.endswith(s) for s in SKIP_SUFFIXES)
                or any(name.startswith(p) for p in SKIP_PREFIXES)):
            continue
        records: dict[int, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["id"]] = rec
        data[path.stem] = records
    return data


# ── Classification ────────────────────────────────────────────────────────────

def classify_doc(reviewers: dict[str, dict]) -> tuple[int, str | None]:
    """Classify one document into (tier, conflict_type) per adjudication-plan.md §1.

    reviewers: {annotator: record}, only annotators who reviewed this doc.
    conflict_type is None for tier 1/2; "scope" | "boundary" | "three_way" |
    "insufficient_coverage" for tier 3. Scope takes precedence over an otherwise
    clean majority match — every scope-flagged doc goes to adjudication, per the
    plan's "case-by-case, not a blanket rule" decision (§0).
    """
    if len(reviewers) < 2:
        return 3, "insufficient_coverage"

    labelsets = {a: frozenset(spans_by_label(r).keys()) for a, r in reviewers.items()}
    scope_flag = any(is_scope_comment(r) for r in reviewers.values())

    if len(reviewers) == 2:
        (a1, r1), (a2, r2) = reviewers.items()
        if labelsets[a1] != labelsets[a2]:
            return (3, "scope") if scope_flag else (3, "three_way")
        if scope_flag:
            return 3, "scope"
        return (1, None) if spans_by_label(r1) == spans_by_label(r2) else (3, "boundary")

    # len(reviewers) == 3
    vals = list(labelsets.values())
    if vals[0] == vals[1] == vals[2]:
        sb = [spans_by_label(r) for r in reviewers.values()]
        exact = sb[0] == sb[1] == sb[2]
        return (1, None) if exact else (3, "boundary")

    counts = Counter(vals)
    top_labelset, top_n = counts.most_common(1)[0]
    if top_n != 2:
        return (3, "scope") if scope_flag else (3, "three_way")
    if scope_flag:
        return 3, "scope"

    majority = [a for a, ls in labelsets.items() if ls == top_labelset]
    sb = [spans_by_label(reviewers[a]) for a in majority]
    return (2, None) if sb[0] == sb[1] else (3, "boundary")


# ── Output ────────────────────────────────────────────────────────────────────

def write_classification_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["doc_id", "tier", "conflict_type", "annotators_reviewed"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Classification -> {out_path}")


def print_summary(rows: list[dict]) -> None:
    counts: Counter = Counter()
    for r in rows:
        counts["tier1" if r["tier"] == 1 else "tier2" if r["tier"] == 2 else r["conflict_type"]] += 1

    print("\n-- Tier counts ----------------------------------------------------")
    print(f"  {'Bucket':<25} {'n docs':>7}  {'plan estimate':>13}")
    print(f"  {'-'*25} {'-'*7}  {'-'*13}")
    for key, label in [
        ("tier1", "1 - auto-accept"),
        ("tier2", "2 - majority-auto"),
        ("scope", "3 - scope"),
        ("boundary", "3 - boundary"),
        ("three_way", "3 - three-way"),
        ("insufficient_coverage", "3 - insufficient coverage"),
    ]:
        n = counts.get(key, 0)
        plan = PLAN_ESTIMATE.get(key)
        plan_str = str(plan) if plan is not None else "-"
        flag = "" if plan is None or plan == n else "  *"
        print(f"  {label:<25} {n:>7}  {plan_str:>13}{flag}")

    total = len(rows)
    tier3 = sum(v for k, v in counts.items() if k not in ("tier1", "tier2"))
    print(f"\n  Total: {total}  (tier 3 = {tier3}, {tier3/total:.0%})")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify BibNER-Werk gold documents into adjudication tiers"
    )
    parser.add_argument(
        "--export-dir", required=True, type=Path,
        help="Path to Doccano export directory (read-only)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BASE / "data/processed/ner",
        help="Output directory (default: data/processed/ner/)"
    )
    args = parser.parse_args()

    export_dir: Path = args.export_dir
    if not export_dir.is_dir():
        sys.exit(f"Not a directory: {export_dir}")

    export_stem = export_dir.name

    print(f"Loading from: {export_dir}")
    data = load_all_records(export_dir)
    if len(data) < 2:
        sys.exit(f"Need >= 2 annotators; found {len(data)}: {list(data)}")
    print(f"Annotators:   {sorted(data)}")

    all_doc_ids = sorted(set.union(*(set(records) for records in data.values())))

    rows: list[dict] = []
    for doc_id in all_doc_ids:
        reviewers = {
            a: data[a][doc_id]
            for a in data
            if doc_id in data[a] and is_reviewed(data[a][doc_id])
        }
        tier, conflict_type = classify_doc(reviewers)
        rows.append({
            "doc_id": doc_id,
            "tier": tier,
            "conflict_type": conflict_type or "",
            "annotators_reviewed": ",".join(sorted(reviewers)),
        })

    out_path = args.output_dir / f"sr08_disagreement_classification_{export_stem}.csv"
    write_classification_csv(rows, out_path)
    print_summary(rows)


if __name__ == "__main__":
    main()
