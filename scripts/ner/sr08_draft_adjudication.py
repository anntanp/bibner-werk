#!/usr/bin/env python3
# Purpose:      Batch tier-3 (adjudicate) documents into paste-able LLM prompts for
#               manual adjudication drafting, per adjudication-plan.md §2.2/§2.2.1
# Usage:        python scripts/ner/sr08_draft_adjudication.py \
#                   --export-dir data/annotation/doccano/export-20260623-1110
# Inputs:       data/processed/ner/sr08_disagreement_classification_<export>.csv
#               (from sr08_classify_disagreements.py — run that first)
#               *.jsonl files in --export-dir (skips dummy.doccano.jsonl, ann.jsonl,
#               harald.jsonl, *.meta.jsonl, sr08_*)
# Outputs:      data/processed/ner/sr08_adjudication_prompt_batch_NN_<export>.txt
#               (one per batch, ~10-15 tier-3 docs each — paste into a Claude chat)
#               data/processed/ner/sr08_adjudication_sheet_<export>.jsonl
#               (skeleton canonical sheet, one record per tier-3 doc, LLM/PI fields null)
#               stdout: batch count, doc count, output paths
# Dependencies: none (stdlib only)
# Assumptions:  no API access — this writes prompts for manual paste, it does not call
#               an LLM. sr08_ingest_llm_drafts.py (§2.3) reads the pasted responses back in.
#               Self-contained like every sr08 script — no cross-script imports.

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

SKIP_NAMES    = {"dummy.doccano.jsonl", "ann.jsonl", "harald.jsonl"}
SKIP_SUFFIXES = (".meta.jsonl",)
SKIP_PREFIXES = ("sr08_",)

BATCH_SIZE = 12  # within the plan's ~10-15 docs/prompt range (adjudication-plan.md §2.2)

DOCUMENTS_PLACEHOLDER = "%%DOCUMENTS_JSON%%"

PROMPT_TEMPLATE = """You are adjudicating disagreements between three human annotators (frankp, gundi, maria)
who independently labeled named-entity spans in German bibliographic title strings
(BibNER-Werk gold set, JCDL 2026). Draft a SUGGESTED resolution for each document below —
the PI makes the final call in Doccano, so flag uncertainty rather than forcing a
confident-sounding answer.

## Label definitions
- TITLE: main work title. Exactly one per record. If no subtitle separator is present,
  TITLE may extend through what would otherwise be a subtitle.
- OTHER_TITLE: subtitle/alternative title elaborating TITLE (after " : ", "Das ist:", "oder", ...).
- PERSON: named responsible person (author, editor, corporate body) + all credentials,
  degrees, and role phrases that form one naming unit with the name.
- PARALLEL_TITLE: title restated in a second language (after " = " or " / " + non-German text).
- TRANSLATOR: named translator, only with an explicit translation keyword
  ("übersetzt", "Übers.", "ins Deutsche übertragen").
- MEDIUM: musical performance instrumentation only ("für Klavier", "op. 12").

## Scope question (no blanket rule)
Some documents are chapters, sections, or articles (Kapitel/Abschnitt/Aufsatz) rather than
full works. gundi and maria have been marking many of these O ("kein Werk"); frankp has been
annotating them normally. There is no fixed rule for this batch — judge each string on its
own terms: does it name a genuinely titled sub-unit (annotate normally), or is it too
fragmentary/generic to support a TITLE span (return O)? Do not decide from the htype label
alone — read the actual title string.

## Span rules
- Offsets are character positions, 0-indexed, end-exclusive. text[start:end] must equal
  the labeled substring exactly.
- No overlapping or nested spans.
- Separating punctuation between spans (" / ", " : ", ", ") belongs to neither span.

## Conflict types (for context only — do not let this bias the answer)
- "scope": disagreement over whether this document should be annotated at all.
- "boundary": annotators agree an entity is present but disagree on exact offsets.
- "three_way": no majority — three different readings of the string.

## Documents
%%DOCUMENTS_JSON%%

## Output format
Return ONLY a JSON array, one object per doc_id, no other text:
[
  {
    "doc_id": 4,
    "llm_suggested_label": [[7, 38, "TITLE"]],
    "llm_rationale": "One sentence, naming the specific rule applied.",
    "llm_confidence": "high"
  }
]

Self-check before returning:
- Every doc_id from the input appears exactly once in the output.
- For every span: text[start:end] equals the labeled substring exactly.
- No overlapping spans within a document.
- llm_confidence is "high", "medium", or "low" — use "low" for anything genuinely
  ambiguous rather than guessing confidently.
"""


# ── Data loading ──────────────────────────────────────────────────────────────

def comment_text(record: dict) -> str | None:
    texts = [
        c.get("text", c) if isinstance(c, dict) else str(c)
        for c in record.get("Comments", [])
    ]
    texts = [t for t in texts if t]
    return "; ".join(texts) if texts else None


def load_all_records(export_dir: Path) -> dict[str, dict[int, dict]]:
    """Return {annotator: {doc_id: record}} for every record, reviewed or not."""
    data: dict[str, dict[int, dict]] = {}
    for path in sorted(export_dir.glob("*.jsonl")):
        name = path.name
        if (name in SKIP_NAMES
                or any(name.endswith(s) for s in SKIP_SUFFIXES)
                or any(name.startswith(p) for p in SKIP_PREFIXES)):
            continue
        records: dict[int, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["id"]] = rec
        data[path.stem] = records
    return data


def load_tier3_doc_ids(classification_csv: Path) -> list[tuple[int, str, list[str]]]:
    """Return [(doc_id, conflict_type, annotators_reviewed)] for tier-3 rows, sorted by doc_id."""
    rows = []
    with open(classification_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["tier"] != "3":
                continue
            annotators = [a for a in row["annotators_reviewed"].split(",") if a]
            rows.append((int(row["doc_id"]), row["conflict_type"], annotators))
    return sorted(rows)


# ── Doc assembly ──────────────────────────────────────────────────────────────

def build_doc_entries(
    tier3: list[tuple[int, str, list[str]]],
    data: dict[str, dict[int, dict]],
) -> tuple[list[dict], list[dict]]:
    """Build (prompt_docs, sheet_records) for every tier-3 doc.

    prompt_docs: shape for the LLM prompt's ## Documents section (§2.2.1) —
      nested {"spans": [...], "comment": ...} per reviewing annotator.
    sheet_records: shape for the canonical skeleton sheet (§2.2) —
      flat spans per annotator, separate comments dict, LLM/PI fields null.
    """
    all_annotators = sorted(data)
    prompt_docs: list[dict] = []
    sheet_records: list[dict] = []

    for doc_id, conflict_type, annotators in tier3:
        text_source = annotators[0] if annotators else all_annotators[0]
        text = data[text_source][doc_id]["text"]

        annotations_prompt: dict[str, dict] = {}
        annotations_sheet: dict[str, list] = {}
        comments_sheet: dict[str, str] = {}
        for a in annotators:
            rec = data[a][doc_id]
            annotations_prompt[a] = {"spans": rec["label"], "comment": comment_text(rec)}
            annotations_sheet[a] = rec["label"]
            c = comment_text(rec)
            if c:
                comments_sheet[a] = c

        prompt_docs.append({
            "doc_id": doc_id,
            "text": text,
            "conflict_type": conflict_type,
            "annotations": annotations_prompt,
        })
        sheet_records.append({
            "doc_id": doc_id,
            "text": text,
            "conflict_type": conflict_type,
            "annotations": annotations_sheet,
            "comments": comments_sheet,
            "llm_suggested_label": None,
            "llm_rationale": None,
            "llm_confidence": None,
            "pi_decision": None,
            "pi_notes": None,
        })

    return prompt_docs, sheet_records


# ── Output ────────────────────────────────────────────────────────────────────

def write_prompt_batches(
    prompt_docs: list[dict],
    batch_size: int,
    export_stem: str,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(prompt_docs), batch_size):
        batch = prompt_docs[i:i + batch_size]
        batch_num = i // batch_size + 1
        documents_json = json.dumps(batch, indent=2, ensure_ascii=False)
        prompt = PROMPT_TEMPLATE.replace(DOCUMENTS_PLACEHOLDER, documents_json)
        out_path = output_dir / f"sr08_adjudication_prompt_batch_{batch_num:02d}_{export_stem}.txt"
        out_path.write_text(prompt, encoding="utf-8")
        paths.append(out_path)
    return paths


def write_sheet_jsonl(sheet_records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in sheet_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Skeleton sheet -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch tier-3 BibNER-Werk gold documents into paste-able LLM adjudication prompts"
    )
    parser.add_argument(
        "--export-dir", required=True, type=Path,
        help="Path to Doccano export directory (read-only)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BASE / "data/processed/ner",
        help="Output directory (default: data/processed/ner/); also where the "
             "sr08_disagreement_classification_<export>.csv input is expected"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Tier-3 docs per prompt (default: {BATCH_SIZE})"
    )
    args = parser.parse_args()

    export_dir: Path = args.export_dir
    if not export_dir.is_dir():
        sys.exit(f"Not a directory: {export_dir}")

    export_stem = export_dir.name
    classification_csv = args.output_dir / f"sr08_disagreement_classification_{export_stem}.csv"
    if not classification_csv.is_file():
        sys.exit(
            f"Classification CSV not found: {classification_csv}\n"
            f"Run sr08_classify_disagreements.py --export-dir {export_dir} first."
        )

    print(f"Loading from:        {export_dir}")
    data = load_all_records(export_dir)
    print(f"Annotators:          {sorted(data)}")

    tier3 = load_tier3_doc_ids(classification_csv)
    print(f"Tier-3 docs:          {len(tier3)}")
    if not tier3:
        sys.exit("No tier-3 documents found — nothing to draft.")

    prompt_docs, sheet_records = build_doc_entries(tier3, data)

    batch_paths = write_prompt_batches(prompt_docs, args.batch_size, export_stem, args.output_dir)
    print(f"Prompt batches:       {len(batch_paths)} (batch size {args.batch_size})")
    for p in batch_paths:
        print(f"  -> {p}")

    sheet_path = args.output_dir / f"sr08_adjudication_sheet_{export_stem}.jsonl"
    write_sheet_jsonl(sheet_records, sheet_path)

    assert len(sheet_records) == len(tier3), "sheet record count must match tier-3 doc count"
    print(f"\nNext: paste each batch into a Claude chat, save the JSON response as "
          f"data/annotation/adjudication_responses/batch_NN.json, then run "
          f"sr08_ingest_llm_drafts.py.")


if __name__ == "__main__":
    main()
