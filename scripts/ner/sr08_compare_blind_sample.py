#!/usr/bin/env python3
# Purpose:      Compare the PI's blind re-adjudication (no LLM suggestion shown) against
#               the original LLM-assisted decision on the same docs, to check for an
#               anchoring effect — notes/fiz/bibner-werk/adj-llm-justification.md §2/§3.
# Usage:        python scripts/ner/sr08_compare_blind_sample.py \
#                   --blind-export data/annotation/doccano/adjudiced-blind-20260707-1914.jsonl \
#                   --key data/adjudication/sr08_blind_sample_key_export-20260623-1110.csv \
#                   --sheet data/adjudication/sr08_adjudication_sheet_export-20260623-1110.jsonl \
#                   --adjudicated data/annotation/doccano/adjucated-20260707-1557.jsonl
# Inputs:       --blind-export: Doccano export of the blind re-adjudication project
#                   (id, text, meta, Comments, label) — built by
#                   sr08_build_blind_sample.py, completed manually by the PI.
#               --key: sr08_build_blind_sample.py's reconciliation key (blind_order,
#                   doc_id, conflict_type, llm_confidence).
#               --sheet: tier-3 canonical sheet (doc_id, text, llm_suggested_label) —
#                   used to match blind-export records to doc_id by exact text (Doccano's
#                   own `id` is a fresh project-internal counter, not the original doc_id),
#                   and to compare the blind decision against the LLM's original suggestion.
#               --adjudicated: the PI's original, LLM-assisted final decision
#                   (data/annotation/doccano/adjucated-<date>.jsonl) — what the blind
#                   decision is actually being checked against.
# Outputs:      data/adjudication/sr08_blind_comparison_<export>.csv — one row per
#                   sampled doc: doc_id, conflict_type, llm_confidence, vs_original
#                   (match_verbatim/match_boundary_edit/mismatch), vs_llm_suggestion
#                   (same three-way scheme, blind decision vs. the LLM's suggestion itself)
#               stdout: agreement rate (blind vs. original) with a Wilson 95% CI, overall
#                   and per conflict_type stratum — flagged as underpowered given n=18 (6/stratum)
# Dependencies: none (stdlib only)
# Assumptions:  Blind-export records are matched to doc_id by exact text match against
#               --sheet (restricted to the doc_ids in --key) — fails loudly on any
#               unmatched or duplicate-text record rather than guessing from row order.
#               Outcome classification reuses the same three-way scheme as
#               sr08_compute_llm_acceptance.py's classify_outcome() (duplicated here, not
#               imported — no sr08 script imports another): match_verbatim (identical span
#               set), match_boundary_edit (same label composition, offsets differ),
#               mismatch (label composition differs).
#               Wilson score interval (z=1.96) used for the agreement-rate CI rather than
#               a normal approximation, since n is small per stratum (as low as 6).

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

Span = tuple[int, int, str]
CONFLICT_ORDER = ["scope", "boundary", "three_way"]


def to_spans(raw: list[list]) -> frozenset[Span]:
    return frozenset((start, end, label) for start, end, label in raw)


def label_composition(raw: list[list]) -> Counter[str]:
    return Counter(label for _, _, label in raw)


def classify_outcome(label_a: list[list], label_b: list[list]) -> str:
    if to_spans(label_a) == to_spans(label_b):
        return "match_verbatim"
    if label_composition(label_a) == label_composition(label_b):
        return "match_boundary_edit"
    return "mismatch"


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion x/n."""
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return ((center - adj) / denom, (center + adj) / denom)


def load_jsonl_by_key(path: Path, key: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)[key]: json.loads(line) for line in f}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare blind re-adjudication against the original LLM-assisted decision"
    )
    parser.add_argument("--blind-export", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--sheet", required=True, type=Path)
    parser.add_argument("--adjudicated", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=BASE / "data/adjudication")
    args = parser.parse_args()

    with open(args.blind_export, encoding="utf-8") as f:
        blind_records = [json.loads(line) for line in f]

    with open(args.key, encoding="utf-8") as f:
        key_rows = {int(row["doc_id"]): row for row in csv.DictReader(f)}

    sheet_by_id = load_jsonl_by_key(args.sheet, "doc_id")
    text_to_doc_id = {sheet_by_id[doc_id]["text"]: doc_id for doc_id in key_rows}
    if len(text_to_doc_id) != len(key_rows):
        sys.exit("Duplicate text among sampled docs -- text-based matching is ambiguous")

    adjudicated_by_id = load_jsonl_by_key(args.adjudicated, "id")

    unmatched = [r["text"][:60] for r in blind_records if r["text"] not in text_to_doc_id]
    if unmatched:
        sys.exit(f"{len(unmatched)} blind-export record(s) not found in --sheet: {unmatched}")
    if len(blind_records) != len(key_rows):
        sys.exit(f"Blind export has {len(blind_records)} records, key has {len(key_rows)} -- expected equal")

    out_rows: list[dict] = []
    for rec in blind_records:
        doc_id = text_to_doc_id[rec["text"]]
        key_row = key_rows[doc_id]
        blind_label = rec["label"]
        original_label = adjudicated_by_id[doc_id]["label"]
        llm_suggested = sheet_by_id[doc_id]["llm_suggested_label"]

        out_rows.append({
            "doc_id": doc_id,
            "conflict_type": key_row["conflict_type"],
            "llm_confidence": key_row["llm_confidence"],
            "vs_original": classify_outcome(blind_label, original_label),
            "vs_llm_suggestion": classify_outcome(blind_label, llm_suggested),
        })

    m = re.search(r"sr08_blind_sample_key_(.+)\.csv$", args.key.name)
    export_stem = m.group(1) if m else args.key.stem

    out_path = args.output_dir / f"sr08_blind_comparison_{export_stem}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["doc_id", "conflict_type", "llm_confidence", "vs_original", "vs_llm_suggestion"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Per-doc comparison -> {out_path}\n")

    def report(rows: list[dict], label: str) -> None:
        n = len(rows)
        if n == 0:
            return
        agree = sum(1 for r in rows if r["vs_original"] in ("match_verbatim", "match_boundary_edit"))
        lo, hi = wilson_ci(agree, n)
        print(f"  {label:<12} agree {agree}/{n} ({agree/n:.1%})  "
              f"Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"    outcome breakdown: {dict(Counter(r['vs_original'] for r in rows))}")

    print("-- Blind vs. original (LLM-assisted) decision, agreement rate -----------")
    report(out_rows, "overall")
    for c in CONFLICT_ORDER:
        report([r for r in out_rows if r["conflict_type"] == c], c)
    print("\n  NOTE: n as low as 6 per stratum -- these CIs are wide; treat as directional,")
    print("  not a precise estimate. See adj-llm-justification.md for the intra-rater-drift")
    print("  confound this check cannot separate out from anchoring.\n")

    print("-- Blind vs. the LLM's original suggestion (independent-agreement check) --")
    llm_agree = sum(1 for r in out_rows if r["vs_llm_suggestion"] in ("match_verbatim", "match_boundary_edit"))
    print(f"  {llm_agree}/{len(out_rows)} ({llm_agree/len(out_rows):.1%}) -- how often blind PI judgment "
          f"matches the LLM's suggestion with no anchoring possible")


if __name__ == "__main__":
    main()
