#!/usr/bin/env python3
# Purpose:      Compute the PI's acceptance/override rate for LLM-drafted tier-3
#               adjudication suggestions (sr08 gold reconciliation), per
#               notes/fiz/bibner-werk/adj-llm-justification.md §3.1.
# Usage:        python scripts/ner/sr08_compute_llm_acceptance.py \
#                   --sheet data/adjudication/sr08_adjudication_sheet_export-20260623-1110.jsonl \
#                   --adjudicated data/annotation/doccano/adjucated-20260707-1557.jsonl
# Inputs:       --sheet: tier-3 canonical sheet (doc_id, conflict_type, llm_suggested_label,
#                   llm_confidence), written by sr08_draft_adjudication.py /
#                   sr08_ingest_llm_drafts.py.
#               --adjudicated: the PI's completed Doccano export (id, label, Comments) —
#                   the only place the PI's final decision lives. sr08_ingest_adjudication.py
#                   (adjudication-plan.md §2.6) was never run, so `pi_decision` in --sheet
#                   is null for all 303 docs and is not used here.
# Outputs:      data/adjudication/sr08_llm_acceptance_<export>.csv — one row per tier-3
#                   doc: doc_id, conflict_type, llm_confidence, outcome, contested_ref,
#                   precedent_ref, pi_comment
#               data/adjudication/sr08_llm_acceptance_summary_<export>.csv — one row per
#                   stratum (overall / each conflict_type / each llm_confidence): dimension,
#                   value, accepted_verbatim, accepted_boundary_edit, overridden, total,
#                   override_pct — backs the two summary tables reported in the paper/notes
#               stdout: same summary, printed as cross-tabs
# Dependencies: none (stdlib only)
# Assumptions:  Outcome is decided by comparing spans directly, never by presence/absence
#               of a PI Comment — only 13/303 tier-3 docs carry a non-empty Comments entry,
#               so most PI edits (including full overrides) leave no comment at all.
#               Comments are read only for the optional contested-tokens/precedent
#               cross-reference (`"contested:N"` / `"precedent:N"`, N = section number in
#               sr08_adjudication-contested-tokens.md / adj-precedents.md §2).
#               Three-way outcome classification:
#                 - accepted_verbatim: final (start, end, label) span set is identical to
#                   the LLM's suggestion.
#                 - accepted_boundary_edit: same label composition (same labels, same
#                   counts per label) but at least one span's offsets differ — the PI
#                   agreed with the LLM's call and only adjusted a boundary.
#                 - overridden: label composition differs (a label type was added,
#                   removed, or its count changed) — the PI disagreed with the LLM's
#                   substantive call, e.g. a scope decision (Kapitel -> O or vice versa).
#               Every tier-3 doc_id in --sheet must appear in --adjudicated; missing
#               doc_ids are a fail-loud error, not a silent skip.

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

Span = tuple[int, int, str]

CONTESTED_RE = re.compile(r"contested:(\d+)")
PRECEDENT_RE = re.compile(r"precedent:(\d+)")

OUTCOME_ORDER = ["accepted_verbatim", "accepted_boundary_edit", "overridden"]
CONFIDENCE_ORDER = ["high", "medium", "low"]
CONFLICT_ORDER = ["scope", "boundary", "three_way", "insufficient_coverage"]


def to_spans(raw: list[list]) -> frozenset[Span]:
    return frozenset((start, end, label) for start, end, label in raw)


def label_composition(raw: list[list]) -> Counter[str]:
    return Counter(label for _, _, label in raw)


def classify_outcome(llm_label: list[list], pi_label: list[list]) -> str:
    if to_spans(llm_label) == to_spans(pi_label):
        return "accepted_verbatim"
    if label_composition(llm_label) == label_composition(pi_label):
        return "accepted_boundary_edit"
    return "overridden"


def extract_refs(comments: list) -> tuple[str, str, str]:
    """Return (contested_ref, precedent_ref, raw_comment_text) from a Doccano
    Comments list. Comments entries are plain strings in the adjudicated export
    (not the {text, ...} dict shape used in the raw per-annotator exports)."""
    texts = [c if isinstance(c, str) else str(c.get("text", "")) for c in comments]
    joined = "; ".join(texts)
    contested = ",".join(CONTESTED_RE.findall(joined))
    precedent = ",".join(PRECEDENT_RE.findall(joined))
    return contested, precedent, joined


def load_sheet(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_adjudicated(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def build_summary_rows(rows: list[dict], dimension: str, key: str | None, order: list[str]) -> list[dict]:
    """Aggregate `rows` by `key` (or as a single "all" bucket if key is None)
    into one summary row per stratum, in `order` (any strata not in `order`
    are appended after, in first-seen order)."""
    counts: dict[str, Counter[str]] = {}
    for r in rows:
        value = r[key] if key is not None else "all"
        counts.setdefault(value, Counter())[r["outcome"]] += 1

    summary: list[dict] = []
    seen_extra = [v for v in counts if v not in order]
    for value in [*order, *seen_extra]:
        if value not in counts:
            continue
        c = counts[value]
        total = sum(c.values())
        overridden = c.get("overridden", 0)
        summary.append({
            "dimension": dimension,
            "value": value,
            "accepted_verbatim": c.get("accepted_verbatim", 0),
            "accepted_boundary_edit": c.get("accepted_boundary_edit", 0),
            "overridden": overridden,
            "total": total,
            "override_pct": round(100 * overridden / total, 1) if total else 0.0,
        })
    return summary


def print_crosstab(summary_rows: list[dict], key_label: str) -> None:
    header = f"  {key_label:<20}" + "".join(f"{o:>24}" for o in OUTCOME_ORDER) + f"{'total':>8}"
    print(header)
    print(f"  {'-'*20}" + "".join(f"{'-'*24}" for _ in OUTCOME_ORDER) + f"{'-'*8}")
    for r in summary_rows:
        cells = (f"{r['accepted_verbatim']:>24}{r['accepted_boundary_edit']:>24}"
                 f"{r['overridden']:>24}")
        print(f"  {r['value']:<20}{cells}{r['total']:>8}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PI acceptance/override rate for LLM-drafted tier-3 adjudication suggestions"
    )
    parser.add_argument(
        "--sheet", required=True, type=Path,
        help="Tier-3 canonical sheet (sr08_adjudication_sheet_<export>.jsonl)"
    )
    parser.add_argument(
        "--adjudicated", required=True, type=Path,
        help="PI's completed Doccano export (adjucated-<date>.jsonl)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BASE / "data/adjudication",
        help="Output directory (default: data/adjudication/)"
    )
    args = parser.parse_args()

    sheet_rows = load_sheet(args.sheet)
    final_by_id = load_adjudicated(args.adjudicated)

    m = re.search(r"sr08_adjudication_sheet_(.+)\.jsonl$", args.sheet.name)
    export_stem = m.group(1) if m else args.sheet.stem

    missing = [r["doc_id"] for r in sheet_rows if r["doc_id"] not in final_by_id]
    if missing:
        sys.exit(f"{len(missing)} tier-3 doc_id(s) missing from --adjudicated: {missing}")

    out_rows: list[dict] = []
    for r in sheet_rows:
        doc_id = r["doc_id"]
        final = final_by_id[doc_id]
        outcome = classify_outcome(r["llm_suggested_label"], final["label"])
        contested_ref, precedent_ref, pi_comment = extract_refs(final.get("Comments", []))
        out_rows.append({
            "doc_id": doc_id,
            "conflict_type": r["conflict_type"],
            "llm_confidence": r["llm_confidence"],
            "outcome": outcome,
            "contested_ref": contested_ref,
            "precedent_ref": precedent_ref,
            "pi_comment": pi_comment,
        })

    out_path = args.output_dir / f"sr08_llm_acceptance_{export_stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["doc_id", "conflict_type", "llm_confidence", "outcome",
              "contested_ref", "precedent_ref", "pi_comment"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Per-doc outcomes -> {out_path}\n")

    overall_summary = build_summary_rows(out_rows, "overall", None, ["all"])
    conflict_summary = build_summary_rows(out_rows, "conflict_type", "conflict_type", CONFLICT_ORDER)
    confidence_summary = build_summary_rows(out_rows, "llm_confidence", "llm_confidence", CONFIDENCE_ORDER)

    summary_path = args.output_dir / f"sr08_llm_acceptance_summary_{export_stem}.csv"
    summary_fields = ["dimension", "value", "accepted_verbatim", "accepted_boundary_edit",
                       "overridden", "total", "override_pct"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows([*overall_summary, *conflict_summary, *confidence_summary])
    print(f"Summary tables -> {summary_path}\n")

    print("-- Overall ----------------------------------------------------------")
    total = overall_summary[0]["total"]
    for o in OUTCOME_ORDER:
        n = overall_summary[0][o]
        print(f"  {o:<25} {n:>4}  ({n/total:.1%})")
    print(f"  {'total':<25} {total:>4}\n")

    print("-- By conflict_type ---------------------------------------------------")
    print_crosstab(conflict_summary, "conflict_type")

    print("-- By llm_confidence ----------------------------------------------------")
    print_crosstab(confidence_summary, "llm_confidence")

    with_refs = [r for r in out_rows if r["contested_ref"] or r["precedent_ref"]]
    print(f"-- Docs with contested/precedent cross-refs: {len(with_refs)}/{total} --")
    for r in with_refs:
        print(f"  doc {r['doc_id']:>4}  outcome={r['outcome']:<22}"
              f"  contested={r['contested_ref'] or '-':<4}  precedent={r['precedent_ref'] or '-'}")


if __name__ == "__main__":
    main()
