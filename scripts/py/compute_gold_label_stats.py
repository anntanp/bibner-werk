#!/usr/bin/env python3
"""Compute per-label entity instance counts and document-level prevalence
for the gold NER set, including the share of documents with no labels at
all (O-only — the record was judged not to be an independent Work).

Usage:
    python3 compute_gold_label_stats.py \
        --gold-reconciled data/gold/gold_reconciled.jsonl \
        --output data/gold/gold_label_stats.csv

Inputs:
    --gold-reconciled: data/gold/gold_reconciled.jsonl, one JSON
        record per line with "id" and "label" (list of [start, end, type]
        char-offset spans; empty list means the document is O-only).
Outputs: a CSV with one row per label type (instance_count = total spans
    of that type; docs_with_label = documents containing >=1 span of that
    type; doc_prevalence_pct = docs_with_label / total_docs), plus a final
    O_ONLY row (docs with zero spans of any type) and a TOTAL row.
Dependencies: none beyond the standard library.
Assumptions: gold_reconciled.jsonl's "label" spans are already adjudicated
    and final (see docs/adjudication-method.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_label_stats(gold_reconciled_path: Path) -> tuple[Counter[str], Counter[str], int, int]:
    """Return (instance_count, docs_with_label, o_only_docs, total_docs)."""
    instance_count: Counter[str] = Counter()
    docs_with_label: Counter[str] = Counter()
    o_only_docs = 0
    total_docs = 0

    with open(gold_reconciled_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total_docs += 1
            spans = record["label"]
            if not spans:
                o_only_docs += 1
                continue
            types_in_doc = {span[2] for span in spans}
            for label in types_in_doc:
                docs_with_label[label] += 1
            for span in spans:
                instance_count[span[2]] += 1

    return instance_count, docs_with_label, o_only_docs, total_docs


def write_csv(
    output_path: Path,
    instance_count: Counter[str],
    docs_with_label: Counter[str],
    o_only_docs: int,
    total_docs: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "instance_count", "docs_with_label", "total_docs", "doc_prevalence_pct"])
        for label in sorted(instance_count):
            pct = round(100 * docs_with_label[label] / total_docs, 1)
            writer.writerow([label, instance_count[label], docs_with_label[label], total_docs, pct])
        o_only_pct = round(100 * o_only_docs / total_docs, 1)
        writer.writerow(["O_ONLY", 0, o_only_docs, total_docs, o_only_pct])
        writer.writerow(["TOTAL", sum(instance_count.values()), total_docs, total_docs, 100.0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-reconciled", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    instance_count, docs_with_label, o_only_docs, total_docs = compute_label_stats(args.gold_reconciled)
    write_csv(args.output, instance_count, docs_with_label, o_only_docs, total_docs)
    logger.info("%d total docs, %d O-only -> %s", total_docs, o_only_docs, args.output)


if __name__ == "__main__":
    main()
