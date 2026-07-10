#!/usr/bin/env python3
# Purpose:      Quantify how many silver training records exceed GLiNER's
#               model.config.max_len (word-token truncation, default 384 for
#               urchade/gliner_multi-v2.1) and how many NER entity spans get
#               silently dropped as a result. Triggered by the
#               "Sentence of length N has been truncated to 384" warning seen
#               during a real train_gliner_silver.py run — the truncation
#               happens in gliner's own processor.py (self.config.max_len,
#               not something train_gliner_silver.py's CLI controls), and
#               prepare_span_labels() masks out any span whose end index
#               falls past the truncated length, so an entity past the cutoff
#               doesn't just lose text, it loses its label too.
# Usage:        python3 scripts/py/analyze_max_len_truncation.py
#                   [--silver-data PATH] [--max-len INT] [--output-dir PATH]
# Inputs:       data/silver/gliner_dataset_silver.jsonl (GLiNER schema:
#                   {"id","tokenized_text","ner","silver_tier","era"})
# Outputs:      data/processed/max_len_truncation_summary.csv (one row:
#                   overall record/entity truncation counts)
#               data/processed/max_len_truncation_by_label.csv (per-label
#                   entity drop counts, to check whether truncation
#                   disproportionately hits PERSON — already the minority
#                   class oversampled per A5-person-imbalance.md)
# Dependencies: none beyond the standard library
# Assumptions:  ner spans are [start, end, label] word-token indices with an
#               INCLUSIVE end (confirmed against gliner's
#               data_processing/processor.py prepare_span_labels():
#               valid_span_mask = spans_idx[:, 1] > num_tokens - 1). A span is
#               dropped by truncation to max_len if its end index >= max_len.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SILVER_DATA = ROOT / "data" / "silver" / "gliner_dataset_silver.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
DEFAULT_MAX_LEN = 384  # urchade/gliner_multi-v2.1 checkpoint config, confirmed via the training log


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-data", type=Path, default=DEFAULT_SILVER_DATA)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                         help="word-token cutoff to check against (default: 384, "
                              "urchade/gliner_multi-v2.1's config.max_len)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = read_jsonl(args.silver_data)

    total_records = len(records)
    records_truncated = 0
    total_entities = 0
    entities_dropped = 0
    by_label: dict[str, dict[str, int]] = {}

    for rec in records:
        num_tokens = len(rec["tokenized_text"])
        truncated = num_tokens > args.max_len
        if truncated:
            records_truncated += 1
        for start, end, label in rec["ner"]:
            total_entities += 1
            stats = by_label.setdefault(label, {"total": 0, "dropped": 0})
            stats["total"] += 1
            if truncated and end >= args.max_len:
                entities_dropped += 1
                stats["dropped"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "max_len_truncation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["max_len", "total_records", "records_truncated", "pct_records_truncated",
                          "total_entities", "entities_dropped", "pct_entities_dropped"])
        writer.writerow([
            args.max_len,
            total_records,
            records_truncated,
            round(100 * records_truncated / total_records, 3) if total_records else 0,
            total_entities,
            entities_dropped,
            round(100 * entities_dropped / total_entities, 3) if total_entities else 0,
        ])

    by_label_path = args.output_dir / "max_len_truncation_by_label.csv"
    with open(by_label_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "entities_total", "entities_dropped", "pct_dropped"])
        for label in sorted(by_label):
            stats = by_label[label]
            pct = round(100 * stats["dropped"] / stats["total"], 3) if stats["total"] else 0
            writer.writerow([label, stats["total"], stats["dropped"], pct])

    print(f"Wrote {summary_path}")
    print(f"Wrote {by_label_path}")
    print(f"{records_truncated}/{total_records} records exceed max_len={args.max_len} "
          f"({100 * records_truncated / total_records:.2f}%)" if total_records else "no records")
    print(f"{entities_dropped}/{total_entities} entity spans dropped "
          f"({100 * entities_dropped / total_entities:.2f}%)" if total_entities else "no entities")


if __name__ == "__main__":
    main()
