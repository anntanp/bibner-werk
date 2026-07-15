#!/usr/bin/env python3
# Purpose:      Quantify how many silver training records exceed
#               train_xlmr_silver.py's --max-length=256 subword-token cutoff,
#               and how many BIO entity spans get silently dropped as a
#               result. Unlike GLiNER's max_len=384 (a word-token count
#               checked by analyze_max_len_truncation.py), XLM-R's cutoff is
#               a subword-token count applied after xlm-roberta-large's own
#               SentencePiece tokenization (tokenize_and_align_labels() in
#               train_xlmr_silver.py, tokenizer(..., is_split_into_words=True,
#               truncation=True, max_length=256)), so the two limits are not
#               directly comparable and need a separate check against the
#               real tokenizer rather than a word-count proxy.
# Usage:        python3 scripts/py/analyze_xlmr_max_len_truncation.py
#                   [--bio-data PATH] [--model-name STR] [--max-length INT]
#                   [--output-dir PATH]
# Inputs:       data/silver/bio_dataset_silver.jsonl (BIO schema:
#                   {"id","tokens","tags", ...})
# Outputs:      data/processed/xlmr_max_len_truncation_summary.csv (one row:
#                   overall record/entity truncation counts)
#               data/processed/xlmr_max_len_truncation_by_label.csv (per-label
#                   entity drop counts, to check whether truncation
#                   disproportionately hits PERSON)
# Dependencies: transformers (AutoTokenizer for xlm-roberta-large; loaded from
#                   the local HF cache, no network access needed)
# Assumptions:  a BIO entity is counted as "dropped" if any of its word
#               indices fall at or past the last word index the truncated
#               subword sequence still covers (tokenizer.word_ids() on the
#               truncated encoding) -- partial truncation of a multi-word
#               entity still corrupts its BIO tag sequence, so it is counted
#               the same as full loss, mirroring analyze_max_len_truncation.py's
#               end >= max_len convention for GLiNER spans.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BIO_DATA = ROOT / "data" / "silver" / "bio_dataset_silver.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
DEFAULT_MODEL_NAME = "xlm-roberta-large"
DEFAULT_MAX_LENGTH = 256  # train_xlmr_silver.py's --max-length default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bio_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """Convert a BIO tag sequence to (start_word_idx, end_word_idx_inclusive, label) spans."""
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    label: str | None = None
    for i, tag in enumerate(tags + ["O"]):  # sentinel to close a trailing span
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1, label))  # type: ignore[arg-type]
            start, label = i, tag[2:]
        elif tag.startswith("I-") and start is not None and tag[2:] == label:
            continue
        else:
            if start is not None:
                spans.append((start, i - 1, label))  # type: ignore[arg-type]
            start, label = None, None
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bio-data", type=Path, default=DEFAULT_BIO_DATA)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # type: ignore[no-untyped-call]
    records = read_jsonl(args.bio_data)

    total_records = len(records)
    records_truncated = 0
    total_entities = 0
    entities_dropped = 0
    by_label: dict[str, dict[str, int]] = {}
    max_subword_len = 0

    for rec in records:
        tokens = rec["tokens"]
        tags = rec["tags"]

        full_len = len(tokenizer(tokens, is_split_into_words=True, truncation=False)["input_ids"])
        max_subword_len = max(max_subword_len, full_len)
        truncated = full_len > args.max_length

        if truncated:
            records_truncated += 1
            truncated_encoding = tokenizer(
                tokens, is_split_into_words=True, truncation=True, max_length=args.max_length
            )
            word_ids = [w for w in truncated_encoding.word_ids() if w is not None]
            last_surviving_word = max(word_ids) if word_ids else -1
        else:
            last_surviving_word = len(tokens) - 1

        for start, end, label in bio_spans(tags):
            total_entities += 1
            stats = by_label.setdefault(label, {"total": 0, "dropped": 0})
            stats["total"] += 1
            if end > last_surviving_word:
                entities_dropped += 1
                stats["dropped"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "xlmr_max_len_truncation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "max_length", "total_records", "records_truncated",
                          "pct_records_truncated", "max_subword_len", "total_entities",
                          "entities_dropped", "pct_entities_dropped"])
        writer.writerow([
            args.model_name,
            args.max_length,
            total_records,
            records_truncated,
            round(100 * records_truncated / total_records, 3) if total_records else 0,
            max_subword_len,
            total_entities,
            entities_dropped,
            round(100 * entities_dropped / total_entities, 3) if total_entities else 0,
        ])

    by_label_path = args.output_dir / "xlmr_max_len_truncation_by_label.csv"
    with open(by_label_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "entities_total", "entities_dropped", "pct_dropped"])
        for label in sorted(by_label):
            stats = by_label[label]
            pct = round(100 * stats["dropped"] / stats["total"], 3) if stats["total"] else 0
            writer.writerow([label, stats["total"], stats["dropped"], pct])

    print(f"Wrote {summary_path}")
    print(f"Wrote {by_label_path}")
    print(f"{records_truncated}/{total_records} records exceed max_length={args.max_length} subwords "
          f"({100 * records_truncated / total_records:.3f}%), max seen: {max_subword_len}"
          if total_records else "no records")
    print(f"{entities_dropped}/{total_entities} entity spans dropped "
          f"({100 * entities_dropped / total_entities:.3f}%)" if total_entities else "no entities")


if __name__ == "__main__":
    main()
