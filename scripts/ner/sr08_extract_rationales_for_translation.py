#!/usr/bin/env python3
"""Extract llm_rationale texts (with doc_id) from adjudication batch files
that don't yet have llm_rationale_de, into one consolidated JSON for
bulk translation.

Usage:
    python3 sr08_extract_rationales_for_translation.py \
        --responses-dir data/annotation/adjudication_responses \
        --start 11 --end 26 \
        --output data/adjudication/sr08_rationales_to_translate.json

Inputs: batch_NN.json files (list of records with doc_id, llm_rationale, ...)
Outputs: JSON list of {batch, doc_id, llm_rationale, llm_rationale_de: null},
         in batch/doc_id order, ready for a translator to fill in
         llm_rationale_de in place.
Dependencies: stdlib only.
Assumptions: batch files are zero-padded two-digit numbered
    (batch_01.json .. batch_26.json); only records missing
    llm_rationale_de are included.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    to_translate = []
    for batch_num in range(args.start, args.end + 1):
        batch_path = responses_dir / f"batch_{batch_num:02d}.json"
        if not batch_path.exists():
            raise FileNotFoundError(f"missing {batch_path}")
        records = json.loads(batch_path.read_text(encoding="utf-8"))
        for rec in records:
            if rec.get("llm_rationale_de") is not None:
                continue
            to_translate.append({
                "batch": batch_num,
                "doc_id": rec["doc_id"],
                "llm_rationale": rec["llm_rationale"],
                "llm_rationale_de": None,
            })

    out_path.write_text(
        json.dumps(to_translate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{len(to_translate)} rationales (batches {args.start}-{args.end}) written to {out_path}")


if __name__ == "__main__":
    main()
