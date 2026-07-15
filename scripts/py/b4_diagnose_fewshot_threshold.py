#!/usr/bin/env python3
# Purpose:      B4 diagnostic -- investigate the 2026-07-15 finding that few-shot GLiNER
#               predictions collapse toward zero as k grows (k=1: PERSON only; k=3/k=5:
#               nothing at all, any label, any record -- B4-gliner-fewshot-workflow.md §2).
#               Reruns model.predict_entities at threshold=0.0 on a handful of gold records
#               for each k, printing label description lengths and top raw (pre-threshold)
#               scores, to distinguish two hypotheses (script plan §4.4): (1) calibration
#               collapse -- real but low scores exist below 0.5, still a candidate at 0.0;
#               or (2) candidate generation itself breaks on long descriptions -- nothing
#               scored, at any threshold.
# Usage:        python3 scripts/py/b4_diagnose_fewshot_threshold.py
#                   [--k-values 1,3,5] [--n-records N] [--top-n N] [--model STR]
#                   [--gold PATH] [--silver-spans PATH] [--examples-seed INT]
# Inputs:       data/gold/gold_reconciled.jsonl, data/silver/silver_spans.jsonl
# Outputs:      stdout only -- a diagnostic printout, not a data artifact
# Dependencies: gliner
# Assumptions:  Reuses draw_examples/build_label_descriptions/BASE_DESCRIPTIONS/LABELS/
#               read_jsonl directly from b4_generate_gliner_fewshot_predictions.py
#               (cross-script import, same posture as compare_gliner_zero_vs_finetuned.py's
#               reuse of b6_summarize_gliner_eval.py) so this diagnostic can't silently
#               drift out of sync with the actual generation script. Requires network
#               access to HuggingFace Hub (or a local cache) to load --model; not runnable
#               in a network-isolated sandbox.
from __future__ import annotations

import argparse
import sys
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
    read_jsonl,
)

DEFAULT_TOP_N = 5


def diagnose_record(
    model: Any,
    text: str,
    k_values: list[int],
    examples: dict[str, list[str]],
    top_n: int,
) -> None:
    """Print label descriptions and top raw (threshold=0.0) scores for one gold record, per k."""
    print(f"text: {text!r}")
    for k in k_values:
        descriptions = build_label_descriptions(BASE_DESCRIPTIONS, examples, k)
        enriched_labels = list(descriptions.values())
        print(f"--- k={k} ---")
        for label, description in descriptions.items():
            print(f"  {label} ({len(description)} chars): {description!r}")

        entities = model.predict_entities(text, enriched_labels, threshold=0.0)
        top_entities = sorted(entities, key=lambda e: -e["score"])[:top_n]
        print("  top raw scores (threshold=0.0):")
        for entity in top_entities:
            span_text = text[entity["start"]:entity["end"]]
            print(f"    {entity['score']:.4f}  {entity['label'][:60]}  -> {span_text!r}")
        if not entities:
            print("    NO CANDIDATES AT ALL, even at threshold=0.0")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose GLiNER few-shot prediction collapse: raw scores at threshold=0.0."
    )
    parser.add_argument("--k-values", type=str, default="1,3,5",
                        help="Comma-separated k values to check, e.g. '1,3,5'.")
    parser.add_argument("--n-records", type=int, default=1,
                        help="Number of gold records to check, taken from the start of the file.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Number of top-scoring entities to print per k per record.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace Hub model id or local checkpoint path.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--silver-spans", type=Path, default=DEFAULT_SILVER_SPANS)
    parser.add_argument("--examples-seed", type=int, default=DEFAULT_EXAMPLES_SEED)
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    print(f"Drawing examples from {args.silver_spans} (seed={args.examples_seed})")
    examples = draw_examples(args.silver_spans, args.examples_seed, LABELS)
    for label, spans in examples.items():
        print(f"  {label}: {len(spans)} examples drawn")

    print(f"Loading GLiNER model {args.model}")
    from gliner import GLiNER  # type: ignore[import-untyped]
    model = GLiNER.from_pretrained(args.model)
    model.eval()

    gold_records = read_jsonl(args.gold)[: args.n_records]
    print(f"Diagnosing {len(gold_records)} gold record(s), k={k_values}\n")

    for record in gold_records:
        print(f"=== record id={record['id']} ===")
        diagnose_record(model, record["text"], k_values, examples, args.top_n)


if __name__ == "__main__":
    main()
