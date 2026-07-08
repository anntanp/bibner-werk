#!/usr/bin/env python3
"""Parse notes/ner/sr08_adjudication-contested-tokens.md into a flat list
of (doc_id, obj_id, contested_token). The note is self-contained (obj_id
is written inline into each entry as part of the note-writing workflow —
see notes/fiz/bibner-werk/adjudication-plan.md §2.5.1), so this script
does no external file lookups.

Usage:
    python3 sr08_extract_contested_tokens.py \
        --note notes/ner/sr08_adjudication-contested-tokens.md \
        --output data/adjudication/sr08_contested_tokens.csv

Inputs: the contested-tokens note. Expected structure per case:
    "## N. <obj_id>"              (doc heading)
    "**doc_id**: `<doc_id>`"      (inline, written when the entry was added)
    "### N.x Contested token: \"TOKEN\" → ..."   (one or more per doc)
Outputs: data/adjudication/sr08_contested_tokens.csv
    (doc_id, obj_id, contested_token), and the same rows to stdout.
Dependencies: stdlib only.
Assumptions: doc heading matches r"^## \d+\.\s+(\S+)\s*$", capturing
    obj_id; doc_id line matches r"^\*\*doc_id\*\*:\s*`(\d+)`" and appears
    once per doc, before any of that doc's "Contested token:" subheadings;
    a doc with no doc_id line yet (entry not finished per the note's
    workflow) gets an empty doc_id and a stderr warning rather than
    failing the whole run. Docs with no "Contested token:" subheading are
    skipped.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

DOC_HEADING_RE = re.compile(r"^## \d+\.\s+(\S+)\s*$")
DOC_ID_RE = re.compile(r"^\*\*doc_id\*\*:\s*`(\d+)`")
TOKEN_HEADING_RE = re.compile(r'^### [\d.]+\s+Contested token:\s*"([^"]+)"')


def extract(note_path: Path) -> list[tuple[str, str, str]]:
    rows = []
    current_obj_id = None
    current_doc_id = ""
    for line in note_path.read_text(encoding="utf-8").splitlines():
        doc_match = DOC_HEADING_RE.match(line)
        if doc_match:
            current_obj_id = doc_match.group(1)
            current_doc_id = ""
            continue
        doc_id_match = DOC_ID_RE.match(line)
        if doc_id_match and current_obj_id is not None:
            current_doc_id = doc_id_match.group(1)
            continue
        token_match = TOKEN_HEADING_RE.match(line)
        if token_match and current_obj_id is not None:
            if not current_doc_id:
                print(f"warning: obj_id {current_obj_id} has no **doc_id** line yet", file=sys.stderr)
            rows.append((current_doc_id, current_obj_id, token_match.group(1)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = extract(Path(args.note))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "obj_id", "contested_token"])
        for doc_id, obj_id, token in rows:
            writer.writerow([doc_id, obj_id, token])
            print(f"{doc_id}\t{obj_id}\t{token}")

    print(f"\n{len(rows)} contested token(s) -> {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
