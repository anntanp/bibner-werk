#!/usr/bin/env python3
# Purpose:      Validate a drafted batch_NN.json adjudication response against its source
#               prompt file — the manual-verification step in adjudication-batch-workflow.md
#               §2.4, formalized as a script instead of repeated ad-hoc python3 -c snippets.
# Usage:        python scripts/ner/sr08_verify_adjudication_batch.py \
#                   --batch-prompt data/adjudication/sr08_adjudication_prompt_batch_08_export-20260623-1110.txt \
#                   --response data/annotation/adjudication_responses/batch_08.json
#               python scripts/ner/sr08_verify_adjudication_batch.py --batch-prompt ... --show
#                   (prints each doc's text + every annotator's spans/comments, no response needed)
# Inputs:       a sr08_adjudication_prompt_batch_NN_<export>.txt file (§2.2) and, unless --show,
#               a batch_NN.json response file matching the prompt's output schema (§2.2.1)
# Outputs:      stdout only — pass/fail per doc, first failure reason; exits non-zero on any failure
# Dependencies: none (stdlib only)
# Assumptions:  batch prompt files embed the documents JSON between the literal markers
#               "## Documents" and "## Output format", as written by sr08_draft_adjudication.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_batch_docs(prompt_path: Path) -> dict[int, dict]:
    raw = prompt_path.read_text(encoding="utf-8")
    try:
        start = raw.index("## Documents") + len("## Documents")
        end = raw.index("## Output format")
    except ValueError:
        sys.exit(f"{prompt_path}: missing '## Documents' / '## Output format' markers")
    docs = json.loads(raw[start:end].strip())
    return {d["doc_id"]: d for d in docs}


def show(docs: dict[int, dict]) -> None:
    for doc_id in sorted(docs):
        d = docs[doc_id]
        text = d["text"]
        print(f"=== doc {doc_id} ({d['conflict_type']}) len={len(text)} ===")
        print(repr(text))
        for annotator, v in d["annotations"].items():
            for s, e, label in v["spans"]:
                print(f"  {annotator:8} {label:12} [{s}:{e}] = {text[s:e]!r}")
            if v["comment"]:
                print(f"  {annotator:8} comment: {v['comment']!r}")
        print()


def verify(docs: dict[int, dict], response_path: Path) -> list[str]:
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(response, list):
        return [f"{response_path}: expected a JSON array, got {type(response).__name__}"]

    errors: list[str] = []
    seen: set[int] = set()

    for entry in response:
        doc_id = entry.get("doc_id")
        if doc_id not in docs:
            errors.append(f"doc_id={doc_id}: not present in {response_path.name if False else 'batch prompt'}")
            continue
        if doc_id in seen:
            errors.append(f"doc_id={doc_id}: appears more than once in the response")
        seen.add(doc_id)

        text = docs[doc_id]["text"]
        spans = sorted((s, e, label) for s, e, label in entry["llm_suggested_label"])

        for s, e, label in spans:
            seg = text[s:e]
            if seg != seg.strip():
                errors.append(f"doc_id={doc_id} {label}[{s}:{e}]: leading/trailing whitespace in {seg!r}")
            if not seg:
                errors.append(f"doc_id={doc_id} {label}[{s}:{e}]: empty span")

        for i in range(len(spans) - 1):
            if spans[i][1] > spans[i + 1][0]:
                errors.append(f"doc_id={doc_id}: overlap between {spans[i]} and {spans[i+1]}")

        if entry.get("llm_confidence") not in {"high", "medium", "low"}:
            errors.append(f"doc_id={doc_id}: invalid llm_confidence {entry.get('llm_confidence')!r}")

    missing = sorted(set(docs) - seen)
    if missing:
        errors.append(f"missing doc_ids from response: {missing}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a drafted adjudication batch response against its source prompt file"
    )
    parser.add_argument("--batch-prompt", required=True, type=Path)
    parser.add_argument("--response", type=Path, help="batch_NN.json response file (omit with --show)")
    parser.add_argument("--show", action="store_true", help="print text + annotator spans, don't verify")
    args = parser.parse_args()

    if not args.batch_prompt.is_file():
        sys.exit(f"Not a file: {args.batch_prompt}")

    docs = load_batch_docs(args.batch_prompt)

    if args.show:
        show(docs)
        return

    if not args.response:
        sys.exit("--response is required unless --show is given")
    if not args.response.is_file():
        sys.exit(f"Not a file: {args.response}")

    errors = verify(docs, args.response)
    if errors:
        print(f"FAIL — {len(errors)} issue(s) in {args.response.name}:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"PASS — {args.response.name}: {len(docs)} docs, all spans clean, no overlaps, no missing doc_ids")


if __name__ == "__main__":
    main()
