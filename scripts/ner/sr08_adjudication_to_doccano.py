#!/usr/bin/env python3
"""Convert the full BibNER-Werk gold corpus (tier 1 auto-accept + tier 2
majority-auto + tier 3 PI-adjudicated) into two Doccano-importable JSONL
files (English and German) for the PI's visual review (§2.4 of
adjudication-plan.md).

Usage:
    python3 sr08_adjudication_to_doccano.py \
        --export-dir data/annotation/doccano/export-20260623-1110 \
        --classification-csv data/adjudication/sr08_disagreement_classification_export-20260623-1110.csv \
        --sheet data/adjudication/sr08_adjudication_sheet_export-20260623-1110.jsonl \
        --output-en data/adjudication/sr08_adjudication_import_en_export-20260623-1110.jsonl \
        --output-de data/adjudication/sr08_adjudication_import_de_export-20260623-1110.jsonl

Inputs:
    --export-dir: the original per-annotator Doccano export (frankp.jsonl,
        gundi.jsonl, maria.jsonl), used for text + raw spans/comments of
        tier 1/2 docs (all three files cover doc_id 1..395).
    --classification-csv: sr08_classify_disagreements.py's output
        (doc_id, tier, conflict_type, annotators_reviewed) for all 395 docs.
    --sheet: the completed tier-3 adjudication sheet (doc_id, text,
        annotations, comments, llm_suggested_label, llm_rationale,
        llm_rationale_de, llm_confidence) for the 303 tier-3 docs.
Outputs: two Doccano sequence-labeling import JSONLs, one row per doc_id,
    in ascending doc_id order (== the order of every file in --export-dir,
    since all three annotator files are already sorted 1..395 — i.e. the
    same order as the original annotation export). Per record:
    - text: the original document only (no diff, no offset shifting)
    - label: resolved gold spans (tier 1/2) or the LLM's suggested gold
      spans (tier 3), pre-filled
    - Comments: [reason line, then one line per reviewing annotator
      showing their actual labeled substrings (not offsets), or their
      O-reason if they declined; non-reviewing annotators (no spans and
      no comment) are omitted entirely]. The reason line is "100% Agreed"
      for tier 1 (unanimous, both EN and DE files); for tier 2 it names
      the majority annotators; for tier 3 it is
      "Reason: (Confidence: ...) {llm_rationale}" (EN) or
      "Grund: (Konfidenz: ...) {llm_rationale_de}" (DE). Per-annotator
      comments are verbatim (as originally written) in both files, not
      translated.
Dependencies: stdlib only.
Assumptions: an annotator is "reviewing" iff they have a non-empty span
    list or a comment (same definition as sr08_compute_iaa.py's
    get_labels: reviewed = label != [] OR Comments != []). Tier 2 always
    has exactly 3 reviewers with a 2-of-3 exact-span majority (per
    sr08_classify_disagreements.py's classify_doc); tier 1 has 2 or 3
    reviewers all matching exactly.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

ANNOTATOR_ORDER = ["frankp", "gundi", "maria"]

SKIP_NAMES = {"dummy.doccano.jsonl", "ann.jsonl", "harald.jsonl"}
SKIP_SUFFIXES = (".meta.jsonl",)
SKIP_PREFIXES = ("sr08_",)


# ── Shared helpers (duplicated from sr08_classify_disagreements.py /
#    sr08_draft_adjudication.py — no sr08 script imports another) ────────────

def comment_text(record: dict) -> str | None:
    texts = [
        c.get("text", c) if isinstance(c, dict) else str(c)
        for c in record.get("Comments", [])
    ]
    texts = [t for t in texts if t]
    return "; ".join(texts) if texts else None


def spans_by_label(record: dict) -> dict[str, set[tuple[int, int]]]:
    d: defaultdict[str, set[tuple[int, int]]] = defaultdict(set)
    for start, end, lbl in record["label"]:
        d[lbl].add((start, end))
    return dict(d)


def load_all_records(export_dir: Path) -> dict[str, dict[int, dict]]:
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


def build_annotator_line(annotator: str, text: str, spans: list, comment: str | None) -> str | None:
    reviewing = bool(spans) or bool(comment)
    if not reviewing:
        return None
    if not spans:
        return f"{annotator}: O[Grund: '{comment}']"
    parts = [f"{label}['{text[start:end]}']" for start, end, label in spans]
    if comment:
        parts.append(f"COMMENT['{comment}']")
    return f"{annotator}: " + ", ".join(parts)


def build_comments(text: str, annotations: dict, comments: dict, reason_line: str) -> list:
    out = [reason_line]
    for annotator in ANNOTATOR_ORDER:
        spans = annotations.get(annotator, [])
        comment = comments.get(annotator)
        line = build_annotator_line(annotator, text, spans, comment)
        if line is not None:
            out.append(line)
    return out


# ── Tier 1/2 resolution ───────────────────────────────────────────────────────

def resolve_tier12(doc_id: int, reviewers: list[str], data: dict[str, dict[int, dict]]) -> dict:
    """Return {text, label, annotations, comments, majority} for a tier 1/2 doc."""
    recs = {a: data[a][doc_id] for a in reviewers}
    text = recs[reviewers[0]]["text"]

    if len(reviewers) == 3:
        sb = {a: spans_by_label(recs[a]) for a in reviewers}
        pair = next(
            (x, y) for x, y in itertools.combinations(reviewers, 2) if sb[x] == sb[y]
        )
    else:
        pair = tuple(reviewers)

    resolved_label = recs[pair[0]]["label"]
    annotations = {a: recs[a]["label"] for a in reviewers}
    comments = {a: comment_text(recs[a]) for a in reviewers if comment_text(recs[a])}
    return {"text": text, "label": resolved_label, "annotations": annotations,
            "comments": comments, "majority": pair}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--classification-csv", required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--output-en", required=True)
    parser.add_argument("--output-de", required=True)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    data = load_all_records(export_dir)

    with open(args.classification_csv, encoding="utf-8") as f:
        classification = sorted(
            ({**row, "doc_id": int(row["doc_id"])} for row in csv.DictReader(f)),
            key=lambda r: r["doc_id"],
        )

    sheet_by_id: dict[int, dict] = {}
    with open(args.sheet, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sheet_by_id[rec["doc_id"]] = rec

    out_en_path = Path(args.output_en)
    out_de_path = Path(args.output_de)
    out_en_path.parent.mkdir(parents=True, exist_ok=True)
    out_de_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_en_path.open("w", encoding="utf-8") as fout_en, \
            out_de_path.open("w", encoding="utf-8") as fout_de:
        for row in classification:
            doc_id = row["doc_id"]
            tier = int(row["tier"])

            if tier == 3:
                rec = sheet_by_id[doc_id]
                if rec.get("llm_suggested_label") is None:
                    raise ValueError(f"doc_id {doc_id}: missing llm_suggested_label")
                if not rec.get("llm_rationale"):
                    raise ValueError(f"doc_id {doc_id}: missing llm_rationale")
                if not rec.get("llm_rationale_de"):
                    raise ValueError(f"doc_id {doc_id}: missing llm_rationale_de")
                if not rec.get("llm_confidence"):
                    raise ValueError(f"doc_id {doc_id}: missing llm_confidence")

                text = rec["text"]
                label = rec["llm_suggested_label"]
                confidence = rec["llm_confidence"]
                en_reason = f"Reason: (Confidence: {confidence}) {rec['llm_rationale']}"
                de_reason = f"Grund: (Konfidenz: {confidence}) {rec['llm_rationale_de']}"
                annotations, comments = rec["annotations"], rec["comments"]

            else:
                reviewers = [a for a in row["annotators_reviewed"].split(",") if a]
                resolved = resolve_tier12(doc_id, reviewers, data)
                text = resolved["text"]
                label = resolved["label"]
                annotations, comments = resolved["annotations"], resolved["comments"]

                if tier == 1:
                    en_reason = de_reason = "100% Agreed"
                else:
                    x, y = resolved["majority"]
                    en_reason = f"Reason: Tier 2 — majority agreement ({x}, {y} of {len(reviewers)}), no adjudication needed."
                    de_reason = f"Grund: Stufe 2 — mehrheitliche Übereinstimmung ({x}, {y} von {len(reviewers)}), keine Adjudikation erforderlich."

            comments_en = build_comments(text, annotations, comments, en_reason)
            comments_de = build_comments(text, annotations, comments, de_reason)

            out_en = {"text": text, "label": label, "Comments": comments_en}
            out_de = {"text": text, "label": label, "Comments": comments_de}

            fout_en.write(json.dumps(out_en, ensure_ascii=False) + "\n")
            fout_de.write(json.dumps(out_de, ensure_ascii=False) + "\n")
            written += 1

    print(f"{written} record(s) written to {out_en_path} and {out_de_path}")


if __name__ == "__main__":
    main()
