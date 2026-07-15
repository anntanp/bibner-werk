#!/usr/bin/env python3
"""Rank CV folds by overall micro-F1 against gold, to pick a checkpoint for release.

Usage: python3 scripts/py/rank_folds_by_f1.py
Inputs: data/eval/eval_fold{0..9}_{gliner,xlmr}.json (per-fold, per-label/era tp/fp/fn
    from eval_ner_gold.py's `score` subcommand)
Outputs: data/processed/fold_ranking.csv (one row per fold x system, micro-F1 over the
    three core labels TITLE/OTHER_TITLE/PERSON, all eras pooled)
Dependencies: none beyond stdlib
Assumptions: core labels are TITLE, OTHER_TITLE, PERSON (matches B6-xlmr-eval.md /
    B6-gliner-eval.md scope); PARALLEL_TITLE/TRANSLATOR are out of scope and never
    appear in these per-fold files.
"""
import csv
import json
from pathlib import Path

EVAL_DIR = Path("data/eval")
OUT_PATH = Path("data/processed/fold_ranking.csv")
SYSTEMS = ["gliner", "xlmr"]
N_FOLDS = 10


def main() -> None:
    rows = []
    for system in SYSTEMS:
        for fold in range(N_FOLDS):
            path = EVAL_DIR / f"eval_fold{fold}_{system}.json"
            data = json.loads(path.read_text())
            tp = sum(r["tp"] for r in data["results"])
            fp = sum(r["fp"] for r in data["results"])
            fn = sum(r["fn"] for r in data["results"])
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            rows.append(
                {
                    "system": system,
                    "fold": fold,
                    "n_records": data["n_records"],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "micro_f1": round(f1, 4),
                }
            )

    rows.sort(key=lambda r: (r["system"], -r["micro_f1"]))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for system in SYSTEMS:
        best = max((r for r in rows if r["system"] == system), key=lambda r: r["micro_f1"])
        print(f"best {system} fold: fold{best['fold']} (micro-F1={best['micro_f1']})")


if __name__ == "__main__":
    main()
