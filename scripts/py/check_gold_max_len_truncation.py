#!/usr/bin/env python3
# Purpose:      Quantify how many gold_reconciled.jsonl records exceed GLiNER's
#               model.config.max_len (word-token truncation, default 384 for
#               urchade/gliner_multi-v2.1), to check whether B3 zero-shot inference
#               (predict_entities, which tokenizes internally) risks silently
#               truncating gold titles. Companion to analyze_max_len_truncation.py,
#               which only covers pre-tokenized silver training data — gold uses a
#               different schema (Doccano char-offset spans, not word-token ner
#               indices) and was never checked. See
#               notes/fiz/bibner-werk/B3-gliner-zeroshot-script-plan.md §8.
# Usage:        python3 scripts/py/check_gold_max_len_truncation.py
#                   [--gold PATH] [--max-len INT] [--splitter-type STR] [--output PATH]
# Inputs:       data/gold/gold_reconciled.jsonl (Doccano schema: id, text, label)
# Outputs:      data/processed/gold_max_len_truncation_summary.csv (one row: max_len,
#                   splitter_type, total_records, records_truncated,
#                   pct_records_truncated, max_tokens, median_tokens)
# Dependencies: gliner (gliner.data_processing.tokenizer.WordsSplitter only — a local
#                   regex tokenizer, no model download / no network access required)
# Assumptions:  splitter_type="whitespace" matches prepare_gliner_dataset.py's default
#                   (--splitter-type), which is what GLiNER's own WordsSplitter uses
#                   internally during predict_entities' tokenization at inference time
#                   — this check approximates that path, it does not call the model
#                   itself. No entity-level drop count is computed here (unlike
#                   analyze_max_len_truncation.py): gold's `label` spans are Doccano
#                   char offsets, not word-token indices, and B3 is zero-shot inference,
#                   not training, so there is no "span masked out during training" to
#                   measure — only whether a record's token count risks truncation.
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from gliner.data_processing.tokenizer import WordsSplitter  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "gold_max_len_truncation_summary.csv"
DEFAULT_MAX_LEN = 384  # urchade/gliner_multi-v2.1 checkpoint config, same constant as analyze_max_len_truncation.py
DEFAULT_SPLITTER_TYPE = "whitespace"  # matches prepare_gliner_dataset.py's --splitter-type default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def count_word_tokens(text: str, splitter: WordsSplitter) -> int:
    return len(list(splitter(text)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                         help="word-token cutoff to check against (default: 384, "
                              "urchade/gliner_multi-v2.1's config.max_len)")
    parser.add_argument("--splitter-type", type=str, default=DEFAULT_SPLITTER_TYPE,
                         help="GLiNER WordsSplitter backend (default: %(default)s) — "
                              "must match prepare_gliner_dataset.py's default to be comparable")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = read_jsonl(args.gold)
    splitter = WordsSplitter(splitter_type=args.splitter_type)

    token_counts = [count_word_tokens(rec["text"], splitter) for rec in records]
    total_records = len(records)
    records_truncated = sum(1 for n in token_counts if n > args.max_len)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["max_len", "splitter_type", "total_records", "records_truncated",
                          "pct_records_truncated", "max_tokens", "median_tokens"])
        writer.writerow([
            args.max_len,
            args.splitter_type,
            total_records,
            records_truncated,
            round(100 * records_truncated / total_records, 3) if total_records else 0,
            max(token_counts) if token_counts else 0,
            round(statistics.median(token_counts), 1) if token_counts else 0,
        ])

    print(f"Wrote {args.output}")
    if total_records:
        print(f"{records_truncated}/{total_records} gold records exceed max_len={args.max_len} "
              f"({100 * records_truncated / total_records:.2f}%); "
              f"max_tokens={max(token_counts)}, median_tokens={statistics.median(token_counts)}")
    else:
        print("no records")


if __name__ == "__main__":
    main()
