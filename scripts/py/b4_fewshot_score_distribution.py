#!/usr/bin/env python3
# Purpose:      B4 follow-up -- quantify the raw (pre-threshold) score collapse across the full
#               395-record gold set, extending b4_diagnose_fewshot_threshold.py's single-record
#               check to a real distributional result. For each k and each gold record, records
#               GLiNER's top raw score (threshold=0.0) per label, then aggregates
#               mean/median/stdev/min/max per (k, label) plus an OVERALL pseudo-label (the max
#               across the three real labels per record). This is the evidence base for
#               60-results.tex's Few-Shot Results claim that scores fall further as k grows --
#               previously supported by one example record only.
# Usage:        python3 scripts/py/b4_fewshot_score_distribution.py
#                   [--examples-file PATH] [--k-values 1,3,5] [--model STR] [--gold PATH]
#                   [--silver-spans PATH] [--examples-seed INT] [--output PATH]
#                   [--summary-output PATH]
#               --examples-file should point at data/eval/fewshot_examples_pool.json (the
#               hand-curated pool actually cited in 60-results.tex); omitting it falls back to a
#               fresh auto-draw, which will not match the reported result.
# Inputs:       data/gold/gold_reconciled.jsonl, an example pool (curated file via
#                   --examples-file, or a fresh tier-1 silver draw otherwise)
# Outputs:      data/processed/b4_fewshot_threshold_scores.csv
#                   (long format: id, k, label, top_score -- one row per record x k x label)
#               data/processed/b4_fewshot_threshold_summary.csv
#                   (k, label, n, mean, median, stdev, min, max -- the citable aggregate table)
# Dependencies: gliner
# Assumptions:  Reuses BASE_DESCRIPTIONS/LABELS/build_label_descriptions/load_examples_pool/
#               draw_examples/read_jsonl directly from b4_generate_gliner_fewshot_predictions.py
#               (cross-script import, same posture as b4_diagnose_fewshot_threshold.py's reuse).
#               threshold=0.0 throughout, to see every candidate GLiNER proposes, not the
#               reporting threshold=0.5 used elsewhere. A label's top_score is 0.0 if GLiNER
#               proposes no candidate for it in that record at all, not distinguished from a
#               candidate that scored exactly 0.0 -- b4_diagnose_fewshot_threshold.py already
#               ruled out "no candidate at all" as the dominant pattern, so this script only
#               needs score magnitude, not candidate presence. Requires network access to
#               HuggingFace Hub (or a local cache) to load --model; not runnable in a
#               network-isolated sandbox -- run on teach03.
from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_generate_gliner_fewshot_predictions import (  # noqa: E402
    BASE_DESCRIPTIONS,
    DEFAULT_EXAMPLES_SEED,
    DEFAULT_GOLD,
    DEFAULT_MODEL,
    DEFAULT_SILVER_SPANS,
    LABELS,
    build_label_descriptions,
    draw_examples,
    load_examples_pool,
    read_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "b4_fewshot_threshold_scores.csv"
DEFAULT_SUMMARY_OUTPUT = ROOT / "data" / "processed" / "b4_fewshot_threshold_summary.csv"

ScoreRow = tuple[int, int, str, float]  # (doc_id, k, label, top_score)


def top_scores_per_record(model: Any, text: str, label_descriptions: dict[str, str]) -> dict[str, float]:
    """Return each bare label's top raw GLiNER score (threshold=0.0) for one record.

    0.0 for a label means GLiNER proposed no candidate span for it in this record; see the
    header's Assumptions note on why that isn't distinguished from a genuinely low-scoring
    candidate here.
    """
    enriched_labels = list(label_descriptions.values())
    reverse_map = {description: label for label, description in label_descriptions.items()}
    entities = model.predict_entities(text, enriched_labels, threshold=0.0)
    top = dict.fromkeys(label_descriptions, 0.0)
    for entity in entities:
        label = reverse_map[entity["label"]]
        top[label] = max(top[label], entity["score"])
    return top


def compute_summary(rows: list[ScoreRow], k_values: list[int]) -> list[tuple[Any, ...]]:
    """Aggregate per-record top scores into mean/median/stdev/min/max per (k, label).

    Adds an OVERALL pseudo-label per k: the max score across the three real labels, per record,
    then aggregated the same way -- answers "how confident is GLiNER's single best guess per
    record," not just per-label confidence.
    """
    by_key: dict[tuple[int, str], list[float]] = defaultdict(list)
    by_record: dict[tuple[int, int], list[float]] = defaultdict(list)
    for doc_id, k, label, score in rows:
        by_key[(k, label)].append(score)
        by_record[(doc_id, k)].append(score)

    summary = []
    for k in k_values:
        for label in LABELS:
            scores = by_key[(k, label)]
            summary.append(_summary_row(k, label, scores))
        overall_scores = [max(scores) for (_, key_k), scores in by_record.items() if key_k == k]
        summary.append(_summary_row(k, "OVERALL", overall_scores))
    return summary


def _summary_row(k: int, label: str, scores: list[float]) -> tuple[Any, ...]:
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return (
        k, label, len(scores),
        round(statistics.mean(scores), 4), round(statistics.median(scores), 4),
        round(stdev, 4), round(min(scores), 4), round(max(scores), 4),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Quantify GLiNER few-shot's raw-score collapse across the full gold set."
    )
    parser.add_argument("--k-values", type=str, default="1,3,5",
                        help="Comma-separated k values to check, e.g. '1,3,5'.")
    parser.add_argument("--examples-file", type=Path, default=None,
                        help="Hand-curated pool (recommended: data/eval/fewshot_examples_pool.json, "
                             "matching what 60-results.tex reports). Omitting this falls back to a "
                             "fresh auto-draw, which will not match the reported result.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--silver-spans", type=Path, default=DEFAULT_SILVER_SPANS)
    parser.add_argument("--examples-seed", type=int, default=DEFAULT_EXAMPLES_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    args = parser.parse_args()

    k_values = sorted(int(k) for k in args.k_values.split(","))
    max_k = max(k_values)

    if args.examples_file:
        logging.info("Loading hand-curated examples from %s", args.examples_file)
        examples = load_examples_pool(args.examples_file, LABELS, max_k)
    else:
        logging.warning("No --examples-file given -- auto-drawing, will NOT match 60-results.tex")
        examples = draw_examples(args.silver_spans, args.examples_seed, LABELS, n=max_k)

    gold_records = read_jsonl(args.gold)
    logging.info("%d gold records, k=%s", len(gold_records), k_values)

    logging.info("Loading GLiNER model %s", args.model)
    from gliner import GLiNER  # type: ignore[import-untyped]
    model = GLiNER.from_pretrained(args.model)
    model.eval()

    rows: list[ScoreRow] = []
    for k in k_values:
        descriptions = build_label_descriptions(BASE_DESCRIPTIONS, examples, k)
        for i, record in enumerate(gold_records):
            top = top_scores_per_record(model, record["text"], descriptions)
            rows.extend((record["id"], k, label, score) for label, score in top.items())
            if (i + 1) % 50 == 0:
                logging.info("k=%d: %d/%d records done", k, i + 1, len(gold_records))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "k", "label", "top_score"])
        writer.writerows(rows)
    logging.info("Wrote %d rows -> %s", len(rows), args.output)

    summary = compute_summary(rows, k_values)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "label", "n", "mean", "median", "stdev", "min", "max"])
        writer.writerows(summary)
    logging.info("Wrote summary -> %s", args.summary_output)


if __name__ == "__main__":
    main()
