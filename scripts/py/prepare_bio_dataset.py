#!/usr/bin/env python3
# Purpose:      Convert Doccano-schema char-offset NER records (silver spans) into
#               word-level BIO-tagged training examples for XLM-R fine-tuning
#               (train_xlmr_silver.py). Mirrors prepare_gliner_dataset.py's
#               char-offset -> word-index conversion and misalignment-logging
#               posture, but targets BIO tags instead of GLiNER's [start_word,
#               end_word, label] triples. Resolves the standalone-vs-inline open
#               item in notes/fiz/bibner-werk/A5-finetuning-script-plan.md §2/§9
#               in favor of a standalone script, for the same reuse/testability
#               reasons prepare_gliner_dataset.py was split out originally.
# Usage:        python3 scripts/py/prepare_bio_dataset.py --input PATH
#                   [--output PATH] [--misaligned-output PATH]
# Inputs:       Doccano-schema JSONL: {"id"?, "text", "label": [[start,end,LABEL],...],
#                   "silver_tier"?, "era"?} — e.g. data/silver/silver_spans.jsonl
# Outputs:      data/silver/bio_dataset_silver.jsonl
#                   ({"id", "tokens": [...], "tags": ["O","B-TITLE",...], "silver_tier"?, "era"?})
#               data/silver/bio_dataset_silver_misaligned.jsonl (only written if non-empty)
# Dependencies: none beyond the standard library — word splitting is a local regex,
#               not the XLM-R tokenizer (see Assumptions)
# Assumptions:  Word-level splitting, not the XLM-R subword tokenizer — the same
#               \w+(?:[-_]\w+)*|\S regex used by GLiNER's WordsSplitter (kept
#               consistent so both systems see the same "word" granularity for
#               span-alignment purposes), reimplemented locally here to avoid a
#               gliner dependency in an XLM-R-only script. Word -> subword
#               alignment for the actual XLM-R tokenizer happens later, in
#               train_xlmr_silver.py, via the standard "label only the first
#               subword" HF convention (tokenizer.word_ids()) — this script only
#               needs to reach word-level BIO tags, not subwords, since word-level
#               tags are tokenizer-agnostic (works for the xlm-roberta-base
#               fallback too, per A5-finetuning-script-plan.md §3b). Records with
#               any span that doesn't align to a word boundary are excluded from
#               the output and logged to --misaligned-output instead of failing
#               the whole run — same posture as prepare_gliner_dataset.py.
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, cast

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DEFAULT = ROOT / "data" / "silver" / "bio_dataset_silver.jsonl"
MISALIGNED_OUTPUT_DEFAULT = ROOT / "data" / "silver" / "bio_dataset_silver_misaligned.jsonl"

CharSpan = tuple[int, int, str]

# Matches GLiNER's WordsSplitter regex (see module Assumptions) — must stay in
# sync if that pattern ever changes, since span-alignment behavior depends on it.
WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S", re.UNICODE)

# Matches train_gliner_silver.py's LABELS / eval_ner_gold.py's LABELS; ADR D-10.
ENTITY_LABELS = {"TITLE", "OTHER_TITLE", "PERSON"}
BIO_TAGS = ["O", "B-TITLE", "I-TITLE", "B-OTHER_TITLE", "I-OTHER_TITLE", "B-PERSON", "I-PERSON"]

PASSTHROUGH_FIELDS = ("silver_tier", "era")


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Stream records from a JSONL file, one dict per line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def tokenize_words(text: str) -> list[tuple[str, int, int]]:
    """Split text into (word, char_start, char_end) triples via a whitespace/punctuation regex."""
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]


def to_bio_example(
    record_id: object,
    text: str,
    char_spans: list[CharSpan],
    passthrough: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Convert one Doccano-schema record to word-level BIO tags.

    Args:
        record_id: Record identifier, carried through unchanged for traceability.
        text: The raw catalog title string.
        char_spans: Char-offset spans as (start, end, label) — end exclusive,
            matching Doccano's convention (same as build_silver_spans.py output).
        passthrough: Fields to copy verbatim into the output (e.g. silver_tier,
            era). Keys with a None value are omitted rather than written as null.

    Returns:
        (example, misaligned_report): example is None if any span fails to
        align to a word boundary, in which case misaligned_report describes the
        offending span(s); otherwise misaligned_report is None. Mirrors
        prepare_gliner_dataset.py's to_gliner_example().
    """
    tokens = tokenize_words(text)
    words = [word for word, _, _ in tokens]
    start_to_idx = {start: i for i, (_, start, _) in enumerate(tokens)}
    end_to_idx = {end: i for i, (_, _, end) in enumerate(tokens)}

    tags = ["O"] * len(words)
    bad_spans: list[list[object]] = []
    for c_start, c_end, label in char_spans:
        if label not in ENTITY_LABELS:
            bad_spans.append([c_start, c_end, label])
            continue
        if c_start in start_to_idx and c_end in end_to_idx:
            w_start, w_end = start_to_idx[c_start], end_to_idx[c_end]
            tags[w_start] = f"B-{label}"
            for i in range(w_start + 1, w_end + 1):
                tags[i] = f"I-{label}"
        else:
            bad_spans.append([c_start, c_end, label])

    if bad_spans:
        return None, {"id": record_id, "text": text, "bad_spans": bad_spans}
    example: dict[str, object] = {"id": record_id, "tokens": words, "tags": tags}
    for key, value in (passthrough or {}).items():
        if value is not None:
            example[key] = value
    return example, None


def convert(input_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str]]:
    """Convert every record in input_path to word-level BIO format.

    Returns:
        (examples, misaligned, tag_counts) — cleanly-converted examples, records
        excluded due to span misalignment, and a per-tag count over the
        converted examples only (O included).
    """
    examples: list[dict[str, object]] = []
    misaligned: list[dict[str, object]] = []
    tag_counts: Counter[str] = Counter()

    for i, rec in enumerate(read_jsonl(input_path)):
        record_id = rec.get("id", i)
        raw_spans = cast("list[list[object]]", rec["label"])
        char_spans = [(cast(int, s), cast(int, e), cast(str, label)) for s, e, label in raw_spans]
        passthrough = {k: rec.get(k) for k in PASSTHROUGH_FIELDS}
        example, bad = to_bio_example(
            record_id, cast(str, rec["text"]), char_spans, passthrough
        )
        if bad is not None:
            misaligned.append(bad)
            continue
        assert example is not None
        examples.append(example)
        tag_counts.update(cast("list[str]", example["tags"]))

    return examples, misaligned, tag_counts


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
        description="Convert Doccano char-offset NER records to word-level BIO tags for XLM-R fine-tuning."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Doccano-schema JSONL, e.g. data/silver/silver_spans.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT,
                        help="Output BIO-schema JSONL (default: %(default)s)")
    parser.add_argument("--misaligned-output", type=Path, default=MISALIGNED_OUTPUT_DEFAULT,
                        help="Report of records with unaligned spans, excluded from output (default: %(default)s)")
    args = parser.parse_args()

    logging.info("Loading %s", args.input)
    examples, misaligned, tag_counts = convert(args.input)

    logging.info("Converted %d records (%d misaligned, excluded)", len(examples), len(misaligned))
    logging.info("Tag counts: %s", dict(tag_counts))

    write_jsonl(examples, args.output)
    if misaligned:
        write_jsonl(misaligned, args.misaligned_output)
        logging.warning("%d records had misaligned spans — see %s", len(misaligned), args.misaligned_output)


if __name__ == "__main__":
    main()
