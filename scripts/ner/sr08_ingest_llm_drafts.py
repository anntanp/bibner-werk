#!/usr/bin/env python3
# Purpose:      Ingest manually-pasted LLM adjudication drafts (batch_NN.json) into the
#               canonical tier-3 sheet, per adjudication-plan.md §2.3
# Usage:        python scripts/ner/sr08_ingest_llm_drafts.py \
#                   --export-dir data/annotation/doccano/export-20260623-1110 \
#                   --responses-dir data/annotation/adjudication_responses/
# Inputs:       data/adjudication/sr08_adjudication_sheet_<export>.jsonl (skeleton, §2.2)
#               data/annotation/adjudication_responses/*.json (PI's pasted LLM responses)
# Outputs:      data/adjudication/sr08_adjudication_sheet_<export>.jsonl (overwritten in
#               place with llm_suggested_label/llm_rationale/llm_rationale_de/llm_confidence
#               filled)
#               stdout: docs filled this run, total filled, still-missing doc_ids
# Dependencies: none (stdlib only)
# Assumptions:  Runs incrementally — not every tier-3 doc needs a response yet. Hard
#               errors only for correctness problems (duplicate doc_id across responses,
#               a response doc_id not in the tier-3 sheet, malformed response fields);
#               missing/not-yet-drafted docs are reported, not an error. This is a
#               deliberate deviation from adjudication-plan.md §2.3, which specified
#               "fail loudly" on any missing doc_id — too strict for incremental batches.
#               llm_rationale_de (German rationale, for the Doccano Comments block built by
#               sr08_adjudication_to_doccano.py §2.4) is optional — defaults to None if a
#               response entry doesn't have it yet, so older batch files don't break.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

VALID_CONFIDENCE = {"high", "medium", "low"}
REQUIRED_FIELDS  = {"doc_id", "llm_suggested_label", "llm_rationale", "llm_confidence"}


def load_sheet(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[rec["doc_id"]] = rec
    return records


def load_responses(responses_dir: Path) -> dict[int, dict]:
    """Return {doc_id: response} across all *.json files, erroring on duplicates
    or malformed entries."""
    seen: dict[int, dict] = {}
    seen_in: dict[int, str] = {}
    for path in sorted(responses_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        if not isinstance(batch, list):
            sys.exit(f"{path}: expected a JSON array, got {type(batch).__name__}")
        for entry in batch:
            missing = REQUIRED_FIELDS - entry.keys()
            if missing:
                sys.exit(f"{path}: doc_id={entry.get('doc_id')} missing fields: {sorted(missing)}")
            if entry["llm_confidence"] not in VALID_CONFIDENCE:
                sys.exit(
                    f"{path}: doc_id={entry['doc_id']} invalid llm_confidence "
                    f"{entry['llm_confidence']!r} (must be high/medium/low)"
                )
            doc_id = entry["doc_id"]
            if doc_id in seen:
                sys.exit(
                    f"doc_id={doc_id} appears in both {seen_in[doc_id]} and {path} — "
                    "duplicate/overlapping batch response"
                )
            seen[doc_id] = entry
            seen_in[doc_id] = str(path)
    return seen


def write_sheet(records: dict[int, dict], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for doc_id in sorted(records):
            f.write(json.dumps(records[doc_id], ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill manually-pasted LLM adjudication drafts into the canonical tier-3 sheet"
    )
    parser.add_argument("--export-dir", required=True, type=Path, help="Doccano export directory")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "data/adjudication",
        help="Where sr08_adjudication_sheet_<export>.jsonl lives (default: data/adjudication/)"
    )
    parser.add_argument(
        "--responses-dir", type=Path,
        default=BASE / "data/annotation/adjudication_responses",
        help="Directory of batch_NN.json response files "
             "(default: data/annotation/adjudication_responses/)"
    )
    args = parser.parse_args()

    export_stem = args.export_dir.name
    sheet_path = args.output_dir / f"sr08_adjudication_sheet_{export_stem}.jsonl"
    if not sheet_path.is_file():
        sys.exit(
            f"Skeleton sheet not found: {sheet_path}\n"
            f"Run sr08_draft_adjudication.py --export-dir {args.export_dir} first."
        )
    if not args.responses_dir.is_dir():
        sys.exit(f"Not a directory: {args.responses_dir}")

    sheet = load_sheet(sheet_path)
    responses = load_responses(args.responses_dir)

    unexpected = set(responses) - set(sheet)
    if unexpected:
        sys.exit(f"Response doc_ids not in the tier-3 sheet: {sorted(unexpected)}")

    filled_this_run = 0
    for doc_id, entry in responses.items():
        rec = sheet[doc_id]
        already_filled = rec["llm_suggested_label"] is not None
        rec["llm_suggested_label"] = entry["llm_suggested_label"]
        rec["llm_rationale"]       = entry["llm_rationale"]
        rec["llm_rationale_de"]    = entry.get("llm_rationale_de")
        rec["llm_confidence"]      = entry["llm_confidence"]
        if not already_filled:
            filled_this_run += 1

    write_sheet(sheet, sheet_path)

    total_filled = sum(1 for r in sheet.values() if r["llm_suggested_label"] is not None)
    missing = [doc_id for doc_id, r in sheet.items() if r["llm_suggested_label"] is None]

    print(f"Sheet:           {sheet_path}")
    print(f"Filled this run: {filled_this_run}")
    print(f"Total filled:    {total_filled} / {len(sheet)}")
    if missing:
        preview = sorted(missing)[:20]
        tail = " ..." if len(missing) > 20 else ""
        print(f"Still missing:   {len(missing)} doc_ids (not an error — run more batches)")
        print(f"  {preview}{tail}")
    else:
        print("All tier-3 docs have LLM drafts.")


if __name__ == "__main__":
    main()
