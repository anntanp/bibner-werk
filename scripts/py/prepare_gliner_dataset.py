#!/usr/bin/env python3
# Purpose:      Convert Doccano-schema char-offset NER records (gold or silver)
#               into GLiNER's word-indexed training format, using GLiNER's own
#               WordsSplitter so tokenization matches exactly what the model
#               sees at train/inference time.
# Usage:        python3 scripts/py/prepare_gliner_dataset.py --input PATH
#                   [--output PATH] [--misaligned-output PATH]
#                   [--splitter-type whitespace]
# Inputs:       Doccano-schema JSONL: {"id"?, "text", "label": [[start,end,LABEL],...]}
#               e.g. data/processed/silver_spans.jsonl or a gold Doccano export
# Outputs:      data/processed/gliner_dataset.jsonl
#               (GLiNER schema: {"id", "tokenized_text": [...], "ner": [[start_word,end_word,LABEL],...]})
#               data/processed/gliner_dataset_misaligned.jsonl (only written if non-empty)
# Dependencies: gliner
# Assumptions:  Span boundaries are expected to fall exactly on WordsSplitter token
#               edges — verified against build_silver_spans.py's trim_span() output
#               and real gold-adjudication strings, including slash-glued Early
#               Modern German text with no surrounding whitespace (2026-07-07, see
#               notes/ner/ — WordsSplitter's `\w+(?:[-_]\w+)*|\S` regex splits
#               punctuation into its own token even when glued to a word). Records
#               with any misaligned span are excluded from the GLiNER output and
#               logged to --misaligned-output instead of failing the whole run —
#               same not-fatal-but-logged posture as build_silver_spans.py's
#               era_join(). GLiNER's "ner" end index is inclusive (a word index),
#               not a Python-slice exclusive offset — confirmed against
#               gliner/data_processing/processor.py's get_dict().

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterator

from gliner.data_processing.tokenizer import WordsSplitter

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DEFAULT = ROOT / "data" / "processed" / "gliner_dataset.jsonl"
MISALIGNED_OUTPUT_DEFAULT = ROOT / "data" / "processed" / "gliner_dataset_misaligned.jsonl"

CharSpan = tuple[int, int, str]


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Stream records from a JSONL file, one dict per line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def to_gliner_example(
    record_id: object, text: str, char_spans: list[CharSpan], splitter: WordsSplitter
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Convert one Doccano-schema record to GLiNER's word-indexed format.

    Args:
        record_id: Record identifier, carried through unchanged for traceability.
        text: The raw catalog title string.
        char_spans: Char-offset spans as (start, end, label) — end exclusive,
            matching Doccano's convention (same as build_silver_spans.py output).
        splitter: A GLiNER WordsSplitter instance, reused across calls.

    Returns:
        (example, misaligned_report): example is None if any span fails to
        align to a WordsSplitter token boundary, in which case
        misaligned_report describes the offending span(s); otherwise
        misaligned_report is None.
    """
    tokens = list(splitter(text))  # [(word, char_start, char_end), ...]
    words = [word for word, _, _ in tokens]
    start_to_idx = {start: i for i, (_, start, _) in enumerate(tokens)}
    end_to_idx = {end: i for i, (_, _, end) in enumerate(tokens)}

    ner: list[list[object]] = []
    bad_spans: list[list[object]] = []
    for c_start, c_end, label in char_spans:
        if c_start in start_to_idx and c_end in end_to_idx:
            ner.append([start_to_idx[c_start], end_to_idx[c_end], label])
        else:
            bad_spans.append([c_start, c_end, label])

    if bad_spans:
        return None, {"id": record_id, "text": text, "bad_spans": bad_spans}
    return {"id": record_id, "tokenized_text": words, "ner": ner}, None


def convert(
    input_path: Path, splitter_type: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str]]:
    """Convert every record in input_path to GLiNER format.

    Returns:
        (examples, misaligned, label_counts) — cleanly-converted examples,
        records excluded due to span misalignment, and a per-label span count
        over the converted examples only.
    """
    splitter = WordsSplitter(splitter_type=splitter_type)
    examples: list[dict[str, object]] = []
    misaligned: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()

    for i, rec in enumerate(read_jsonl(input_path)):
        record_id = rec.get("id", i)
        char_spans = [(s, e, label) for s, e, label in rec["label"]]  # type: ignore[union-attr]
        example, bad = to_gliner_example(record_id, rec["text"], char_spans, splitter)  # type: ignore[arg-type]
        if bad is not None:
            misaligned.append(bad)
            continue
        assert example is not None
        examples.append(example)
        label_counts.update(label for _, _, label in example["ner"])  # type: ignore[index]

    return examples, misaligned, label_counts


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    """Write records as one JSON object per line, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("Wrote %d records -> %s", len(records), path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert Doccano char-offset NER records to GLiNER's word-indexed training format."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Doccano-schema JSONL (e.g. data/processed/silver_spans.jsonl or a gold export)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT,
                        help="Output GLiNER-schema JSONL (default: %(default)s)")
    parser.add_argument("--misaligned-output", type=Path, default=MISALIGNED_OUTPUT_DEFAULT,
                        help="Report of records with unaligned spans, excluded from output (default: %(default)s)")
    parser.add_argument("--splitter-type", default="whitespace",
                        help="GLiNER WordsSplitter backend (default: %(default)s) — must match "
                             "the splitter used at inference time")
    args = parser.parse_args()

    logging.info("Loading %s", args.input)
    examples, misaligned, label_counts = convert(args.input, args.splitter_type)

    logging.info("Converted %d records (%d misaligned, excluded)", len(examples), len(misaligned))
    logging.info("Label counts: %s", dict(label_counts))

    write_jsonl(examples, args.output)
    if misaligned:
        write_jsonl(misaligned, args.misaligned_output)
        logging.warning("%d records had misaligned spans — see %s", len(misaligned), args.misaligned_output)


if __name__ == "__main__":
    main()
