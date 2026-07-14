#!/usr/bin/env python3
# Purpose:      B3 — generate GLiNER zero-shot predictions on the full reconciled gold set,
#               in the schema eval_ner_gold.py's `score` subcommand expects. One inference
#               pass covers all folds (see notes/fiz/bibner-werk/B3-gliner-zeroshot-script-plan.md
#               §4 — a zero-shot model has no per-fold state, so predictions don't depend on
#               fold assignment; folds only matter downstream, at scoring time).
# Usage:        python3 scripts/py/b3_generate_gliner_zeroshot_predictions.py
#                   [--model STR] [--gold PATH] [--threshold FLOAT] [--output PATH]
# Inputs:       data/gold/gold_reconciled.jsonl
# Outputs:      data/eval/predictions_gliner_zero.jsonl
#                   ({"id": doc_id, "spans": [[start, end, label], ...]})
# Dependencies: gliner
# Assumptions:  Labels TITLE/OTHER_TITLE/PERSON, bare strings (no rich descriptions — that's
#               B4's few-shot job), matching b6_generate_gliner_predictions.py's LABELS exactly
#               (ADR D-10, no pooling). threshold=0.5 is GLiNER's own default, not tuned for
#               this task — same "flagged default" posture as B6 and the training scripts, but
#               a sharper risk here since B3's F1 is the paper's headline zero-shot-failure
#               evidence (see plan §8's threshold-calibration open item). Requires network
#               access to HuggingFace Hub (or a local cache) to load --model; not runnable in
#               a network-isolated sandbox — see plan §6.
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL = "urchade/gliner_multi-v2.1"
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "eval" / "predictions_gliner_zero.jsonl"

LABELS = ["TITLE", "OTHER_TITLE", "PERSON"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def predict_spans(model: Any, text: str, threshold: float) -> list[list[Any]]:
    """Run GLiNER zero-shot prediction, return [[start, end, label], ...] in this project's span-triple schema."""
    entities = model.predict_entities(text, LABELS, threshold=threshold)
    return [[e["start"], e["end"], e["label"]] for e in entities]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate GLiNER zero-shot predictions on gold, for eval_ner_gold.py score."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace Hub model id or local checkpoint path.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--threshold", type=float, default=0.5, help="GLiNER's own default, not tuned")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.info("Loading gold from %s", args.gold)
    gold_records = read_jsonl(args.gold)
    logging.info("%d gold records to predict on.", len(gold_records))

    logging.info("Loading GLiNER zero-shot model %s", args.model)
    from gliner import GLiNER  # type: ignore[import-untyped]
    model = GLiNER.from_pretrained(args.model)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in gold_records:
            spans = predict_spans(model, rec["text"], args.threshold)
            f.write(json.dumps({"id": rec["id"], "spans": spans}, ensure_ascii=False) + "\n")

    logging.info("Wrote %d predictions -> %s", len(gold_records), args.output)


if __name__ == "__main__":
    main()
