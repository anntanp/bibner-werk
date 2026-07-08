#!/usr/bin/env python3
"""Merge translated llm_rationale_de values back into adjudication batch
files, in place, preserving key order (doc_id, llm_suggested_label,
llm_rationale, llm_rationale_de, llm_confidence).

Usage:
    python3 sr08_apply_rationale_translations.py \
        --responses-dir data/annotation/adjudication_responses \
        --translations data/adjudication/sr08_rationales_to_translate.json

Inputs: the JSON produced by sr08_extract_rationales_for_translation.py,
    with llm_rationale_de filled in for every entry.
Outputs: batch_NN.json files overwritten in place with llm_rationale_de
    inserted.
Dependencies: stdlib only.
Assumptions: every entry in the translations file has a non-null
    llm_rationale_de; fails loudly otherwise so a partial translation
    pass can't silently corrupt the batch files.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", required=True)
    parser.add_argument("--translations", required=True)
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    translations = json.loads(Path(args.translations).read_text(encoding="utf-8"))

    missing = [t["doc_id"] for t in translations if not t.get("llm_rationale_de")]
    if missing:
        raise ValueError(f"{len(missing)} entries still missing llm_rationale_de: {missing}")

    by_batch = defaultdict(dict)
    for t in translations:
        by_batch[t["batch"]][t["doc_id"]] = t["llm_rationale_de"]

    for batch_num, doc_map in sorted(by_batch.items()):
        batch_path = responses_dir / f"batch_{batch_num:02d}.json"
        records = json.loads(batch_path.read_text(encoding="utf-8"))

        updated = 0
        for rec in records:
            doc_id = rec["doc_id"]
            if doc_id not in doc_map:
                continue
            new_rec = {
                "doc_id": rec["doc_id"],
                "llm_suggested_label": rec["llm_suggested_label"],
                "llm_rationale": rec["llm_rationale"],
                "llm_rationale_de": doc_map[doc_id],
                "llm_confidence": rec["llm_confidence"],
            }
            rec.clear()
            rec.update(new_rec)
            updated += 1

        batch_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"batch_{batch_num:02d}.json: {updated} record(s) updated")


if __name__ == "__main__":
    main()
