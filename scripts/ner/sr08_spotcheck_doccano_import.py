#!/usr/bin/env python3
"""Spot-check sr08_adjudication_to_doccano.py's output: print one EN+DE
record per tier (1, 2, 3) for manual eyeballing.

Usage:
    python3 sr08_spotcheck_doccano_import.py \
        --classification-csv data/adjudication/sr08_disagreement_classification_export-20260623-1110.csv \
        --output-en data/adjudication/sr08_adjudication_import_en_export-20260623-1110.jsonl \
        --output-de data/adjudication/sr08_adjudication_import_de_export-20260623-1110.jsonl

Inputs: the classification CSV (doc_id, tier, ...) and the EN/DE Doccano
    import JSONLs produced by sr08_adjudication_to_doccano.py.
Outputs: stdout only (one EN + one DE record per tier, pretty-printed).
Dependencies: stdlib only.
Assumptions: both JSONL files are in the same doc_id order as the
    classification CSV sorted by doc_id (ascending, contiguous from 1).
"""
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-csv", required=True)
    parser.add_argument("--output-en", required=True)
    parser.add_argument("--output-de", required=True)
    args = parser.parse_args()

    rows = sorted(
        ({**r, "doc_id": int(r["doc_id"])} for r in csv.DictReader(open(args.classification_csv, encoding="utf-8"))),
        key=lambda r: r["doc_id"],
    )
    en_lines = Path(args.output_en).read_text(encoding="utf-8").splitlines()
    de_lines = Path(args.output_de).read_text(encoding="utf-8").splitlines()

    seen_tiers = set()
    for i, row in enumerate(rows):
        tier = row["tier"]
        if tier in seen_tiers:
            continue
        seen_tiers.add(tier)
        print(f"=== tier {tier}, doc_id {row['doc_id']} (row {i}) ===")
        print("EN:", json.dumps(json.loads(en_lines[i]), ensure_ascii=False, indent=2))
        print("DE:", json.dumps(json.loads(de_lines[i]), ensure_ascii=False, indent=2))
        print()
        if seen_tiers == {"1", "2", "3"}:
            break


if __name__ == "__main__":
    main()
