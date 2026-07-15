#!/usr/bin/env python3
# Purpose:      B6 follow-up error analysis — test the hypothesis that PERSON's low
#               gold-F1 (B6-gliner-eval.md, weighted F1 0.149; B6-xlmr-eval.md, weighted
#               F1 0.120) is concentrated on corporate-body/institutional
#               responsible-agent spans, not individual person names. Motivated by
#               silver's own construction: extract_spans() only ever tags a span PERSON
#               when compute_resp_flags() classified it f_resp_person=1 -- an
#               f_resp_org=1 classification (same SoR text, FLERT said organization)
#               gets NO span at all, so silver's training signal never teaches the
#               model "an institution name can fill this role." If gold's PERSON
#               annotations include institutional agents (plausible -- see
#               person-to-agent-rename-plan.md's "der verantwortliche Agent" framing),
#               this would show up as false negatives concentrated on org-shaped spans
#               specifically. --system (added 2026-07-15) parametrizes this across any
#               fine-tuned system sharing the predictions_{system}_fold{f}.jsonl schema
#               eval_ner_gold.py score consumes -- run once for gliner, once for xlmr,
#               same silver/gold pipeline so a shared script beats duplicating this
#               file for one filename difference.
# Usage:        python3 scripts/py/b6_person_error_analysis.py [--system {gliner,xlmr}]
#                   [--gold PATH] [--predictions-dir PATH] [--folds PATH] [--output PATH]
# Inputs:       data/gold/gold_reconciled.jsonl,
#                   data/eval/predictions_{system}_fold{0-9}.jsonl, data/eval/folds_k10.json
# Outputs:      data/processed/b6_person_error_analysis.csv (--system gliner, the
#                   original default, unchanged path for backward compatibility with
#                   B6-gliner-eval.md's existing reference) or
#                   data/processed/b6_person_error_analysis_{system}.csv otherwise
#                   (one row per gold/predicted PERSON span: doc_id, era, outcome,
#                   span_text, looks_org)
# Dependencies: none (stdlib only)
# Assumptions:  "looks_org" is a keyword heuristic (German institutional-noun list),
#                   not a verified classification -- reported as suggestive evidence,
#                   not proof. Reproduces eval_ner_gold.py's exact-span-match outcome
#                   logic for PERSON only, rather than importing it, since that script
#                   folds outcome computation inline into cmd_score() rather than
#                   exposing it as a reusable function.
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_PREDICTIONS_DIR = ROOT / "data" / "eval"
DEFAULT_FOLDS = ROOT / "data" / "eval" / "folds_k10.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "b6_person_error_analysis.csv"

# German institutional/organizational nouns -- a span containing any of these (case-insensitive,
# word-boundary match) is heuristically flagged as "looks like an organization," not a person name.
ORG_KEYWORDS = [
    "Gesellschaft", "Verein", "Verband", "Kommission", "Ausschuss", "Amt", "Behörde",
    "Ministerium", "Akademie", "Universität", "Institut", "Anstalt", "Kammer", "Kirche",
    "Kloster", "Kongress", "Partei", "Bund", "Kolleg", "Fakultät", "Senat", "Landtag",
    "Reichstag", "Stiftung", "Genossenschaft", "Innung", "Zunft", "Orden", "Redaktion",
    "Verlag", "Druckerei", "Buchhandlung", "Firma", "Compagnie", "Kompanie", "Synode",
    "Landesamt", "Ausschuß", "Konsistorium", "Magistrat", "Rathaus", "Gemeinde",
]
ORG_RE = re.compile(r"\b(?:" + "|".join(re.escape(kw) for kw in ORG_KEYWORDS) + r")\b", re.IGNORECASE)


def looks_org(span_text: str) -> bool:
    return bool(ORG_RE.search(span_text))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="B6 PERSON-label error analysis: person vs. org spans.")
    parser.add_argument("--system", type=str, default="gliner",
                        help="Matches predictions_{system}_fold{f}.jsonl's naming convention (default: gliner)")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: data/processed/b6_person_error_analysis.csv for --system gliner "
                             "(unchanged, matches the original default), "
                             "data/processed/b6_person_error_analysis_{system}.csv otherwise")
    args = parser.parse_args()
    output = args.output or (
        DEFAULT_OUTPUT if args.system == "gliner"
        else ROOT / "data" / "processed" / f"b6_person_error_analysis_{args.system}.csv"
    )

    gold_records = {rec["id"]: rec for rec in read_jsonl(args.gold)}
    folds = json.loads(args.folds.read_text())

    rows: list[dict[str, Any]] = []
    for fold_index in range(10):
        pred_path = args.predictions_dir / f"predictions_{args.system}_fold{fold_index}.jsonl"
        if not pred_path.is_file():
            continue
        predictions = {rec["id"]: rec["spans"] for rec in read_jsonl(pred_path)}

        doc_ids = [int(doc_id) for doc_id, info in folds.items() if info["fold"] == fold_index]
        for doc_id in doc_ids:
            gold_rec = gold_records[doc_id]
            text = gold_rec["text"]
            era = folds[str(doc_id)]["era"]
            gold_person = {tuple(s) for s in gold_rec["label"] if s[2] == "PERSON"}
            pred_person = {tuple(s) for s in predictions.get(doc_id, []) if s[2] == "PERSON"}

            for start, end, label in gold_person | pred_person:
                span_text = text[start:end]
                if (start, end, label) in gold_person and (start, end, label) in pred_person:
                    outcome = "tp"
                elif (start, end, label) in pred_person:
                    outcome = "fp"
                else:
                    outcome = "fn"
                rows.append({
                    "doc_id": doc_id, "fold": fold_index, "era": era, "outcome": outcome,
                    "span_text": span_text, "looks_org": looks_org(span_text),
                })

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "fold", "era", "outcome", "span_text", "looks_org"])
        writer.writeheader()
        writer.writerows(rows)

    for outcome in ("fn", "fp", "tp"):
        subset = [r for r in rows if r["outcome"] == outcome]
        org_count = sum(1 for r in subset if r["looks_org"])
        print(f"{outcome.upper()}: {len(subset)} spans, {org_count} ({org_count/len(subset)*100:.0f}%) look org-like" if subset else f"{outcome.upper()}: 0 spans")

    print(f"\nWrote per-span detail -> {output}")
    fn_org = [r for r in rows if r["outcome"] == "fn" and r["looks_org"]]
    if fn_org:
        print("\nExample org-like false negatives (missed gold PERSON spans):")
        for r in fn_org[:10]:
            print(f"  [{r['era']}] {r['span_text']!r}")


if __name__ == "__main__":
    main()
