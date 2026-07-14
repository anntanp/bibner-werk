#!/usr/bin/env python3
# Purpose:      B6 — generate GLiNER fine-tuned-checkpoint predictions on gold, in the
#               schema eval_ner_gold.py's `score` subcommand expects. Written to get a
#               real signal on true accuracy (against human-adjudicated gold) before
#               committing more GPU-hours to the data-scaling sweep — silver's known
#               labeling bias (Pattern A, A5-finetuning-xlmr-spotcheck.md §4) means
#               silver-dev eval_loss alone can't answer "is silver good enough,"
#               only gold-eval can. See A5-finetuning-scaling-sweep-plan.md.
# Usage:        python3 scripts/py/b6_generate_gliner_predictions.py --fold-index 0
#                   [--checkpoint-dir PATH] [--gold PATH] [--folds PATH]
#                   [--threshold FLOAT] [--output PATH]
#               Run once per fold (0..9) against models/gliner_fold{i}/, matching the
#               1:1 (training-replicate, gold-fold) pairing eval_ner_gold.py's score/
#               aggregate subcommands assume.
# Inputs:       models/gliner_fold{fold-index}/ (GLiNER checkpoint, via train_gliner_silver.py)
#               data/gold/gold_reconciled.jsonl, data/eval/folds_k10.json
# Outputs:      data/eval/predictions_gliner_fold{fold-index}.jsonl
#                   ({"id": doc_id, "spans": [[start,end,label], ...]})
# Dependencies: gliner
# Assumptions:  Labels TITLE/OTHER_TITLE/PERSON, matching train_gliner_silver.py's LABELS
#               exactly (ADR D-10, no pooling). threshold=0.5 is GLiNER's own default,
#               not tuned for this task — same "flagged default" posture as the
#               training scripts' hyperparameters.
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_FOLDS = ROOT / "data" / "eval" / "folds_k10.json"
DEFAULT_CHECKPOINT_ROOT = ROOT / "models"

LABELS = ["TITLE", "OTHER_TITLE", "PERSON"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def select_fold_records(
    gold_records: list[dict[str, Any]], folds: dict[str, Any], fold_index: int
) -> list[dict[str, Any]]:
    """Return gold records whose folds.json assignment matches fold_index."""
    fold_ids = {int(doc_id) for doc_id, info in folds.items() if info["fold"] == fold_index}
    selected = [rec for rec in gold_records if rec["id"] in fold_ids]
    if not selected:
        raise ValueError(f"No gold records assigned to fold {fold_index} in folds file.")
    return selected


def predict_spans(model: Any, text: str, threshold: float) -> list[list[Any]]:
    """Run GLiNER prediction, return [[start, end, label], ...] in this project's span-triple schema."""
    entities = model.predict_entities(text, LABELS, threshold=threshold)
    return [[e["start"], e["end"], e["label"]] for e in entities]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate GLiNER fine-tuned predictions on gold, for eval_ner_gold.py score."
    )
    parser.add_argument("--fold-index", type=int, required=True, help="0..9, matches models/gliner_fold{i}/")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Default: models/gliner_fold{fold-index}/")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--threshold", type=float, default=0.5, help="GLiNER's own default, not tuned")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: data/eval/predictions_gliner_fold{fold-index}.jsonl")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or (DEFAULT_CHECKPOINT_ROOT / f"gliner_fold{args.fold_index}")
    output = args.output or (ROOT / "data" / "eval" / f"predictions_gliner_fold{args.fold_index}.jsonl")

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_dir} — run this on the server where "
            "train_gliner_silver.py's output actually lives, not this machine."
        )

    logging.info("Loading gold from %s", args.gold)
    gold_records = read_jsonl(args.gold)
    folds = json.loads(args.folds.read_text())
    fold_records = select_fold_records(gold_records, folds, args.fold_index)
    logging.info("Fold %d: %d gold records to predict on.", args.fold_index, len(fold_records))

    logging.info("Loading GLiNER checkpoint from %s", checkpoint_dir)
    from gliner import GLiNER  # type: ignore[import-untyped]
    model = GLiNER.from_pretrained(str(checkpoint_dir))
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for rec in fold_records:
            spans = predict_spans(model, rec["text"], args.threshold)
            f.write(json.dumps({"id": rec["id"], "spans": spans}, ensure_ascii=False) + "\n")

    logging.info("Wrote %d predictions -> %s", len(fold_records), output)


if __name__ == "__main__":
    main()
