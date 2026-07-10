#!/usr/bin/env python3
# Purpose:      One-off: write an AGENT-labeled copy of a Doccano-schema gold/silver
#               JSONL file, renaming the PERSON label to AGENT in span triples only.
#               The source file is never modified; "text"/"Comments" fields are
#               never touched, only the label string inside each "label"/"ner" triple.
#               See notes/fiz/bibner-werk/person-to-agent-rename-plan.md.
# Usage:        python3 scripts/py/rename_person_to_agent.py --input PATH --output PATH
# Inputs:       Doccano-schema JSONL: {"id", "text", "label": [[start,end,LABEL],...], ...}
#               or GLiNER-schema JSONL: {"id", "tokenized_text", "ner": [[start,end,LABEL],...]}
#               (label-array key is auto-detected: "label" or "ner")
# Outputs:      Same schema, same record order, PERSON -> AGENT in every span triple's
#               label field; all other fields byte-identical to the input.
# Dependencies: none beyond stdlib.
# Assumptions:  One JSON object per line. Every record has either a "label" or "ner"
#               key holding a list of [start, end, LABEL] triples. Fails loudly (does
#               not silently skip) if a record has neither key, or if post-rename
#               instance counts don't reconcile with pre-rename counts.
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

OLD_LABEL = "PERSON"
NEW_LABEL = "AGENT"


def rename_labels(path_in: Path, path_out: Path) -> tuple[int, int, int]:
    """Rewrite path_in to path_out with OLD_LABEL -> NEW_LABEL in span triples.

    Returns:
        (n_records, n_old_label_before, n_new_label_after) for the caller to
        assert n_old_label_before == n_new_label_after.
    """
    n_records = 0
    n_before = 0
    n_after = 0

    with open(path_in, encoding="utf-8") as f_in, open(path_out, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            n_records += 1

            if "label" in record:
                span_key = "label"
            elif "ner" in record:
                span_key = "ner"
            else:
                raise ValueError(f"{path_in}:{line_no} has neither 'label' nor 'ner' key: {record.keys()}")

            new_spans = []
            for start, end, label in record[span_key]:
                if label == OLD_LABEL:
                    n_before += 1
                    label = NEW_LABEL
                    n_after += 1
                new_spans.append([start, end, label])
            record[span_key] = new_spans

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    return n_records, n_before, n_after


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Write an AGENT-labeled copy of a gold/silver JSONL file (PERSON -> AGENT)."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists — refusing to overwrite; delete it first if intended.")

    n_records, n_before, n_after = rename_labels(args.input, args.output)

    if n_before != n_after:
        raise AssertionError(f"Label count mismatch: {n_before} {OLD_LABEL} before, {n_after} {NEW_LABEL} after")

    logging.info(
        "Wrote %d records -> %s (%d %s -> %s instances, source file untouched)",
        n_records, args.output, n_before, OLD_LABEL, NEW_LABEL,
    )


if __name__ == "__main__":
    main()
