#!/usr/bin/env python3
# Purpose:      B4 — generate GLiNER few-shot predictions on the full reconciled gold set, at a
#               given k-shot value, in the schema eval_ner_gold.py's `score` subcommand expects.
#               One inference pass per k covers all folds (same fold-independence reasoning as
#               B3 -- see notes/fiz/bibner-werk/B4-gliner-fewshot-script-plan.md §4.1). Label
#               descriptions are enriched with k example spans per label, nested across k so
#               k=1 subset of k=3 subset of k=5 (§4.3); GLiNER echoes back whichever label
#               string it was given, so returned entity labels are remapped to the bare
#               TITLE/OTHER_TITLE/PERSON form before writing predictions (§4.2 -- do not skip
#               this, it is silent if wrong).
# Usage:        python3 scripts/py/b4_generate_gliner_fewshot_predictions.py --k-shot K
#                   [--model STR] [--gold PATH] [--silver-spans PATH] [--examples-seed INT]
#                   [--threshold FLOAT] [--output PATH] [--examples-output PATH]
#               K must be one of 1, 3, 5 -- run once per value (see plan §5 for a loop example).
# Inputs:       data/gold/gold_reconciled.jsonl, data/silver/silver_spans.jsonl
# Outputs:      data/eval/predictions_gliner_fewshot_k{K}.jsonl
#                   ({"id": doc_id, "spans": [[start, end, bare_label], ...]})
#               data/eval/fewshot_examples_k{K}.json (the actual drawn example strings, for the
#                   paper's §5 few-shot description to cite verbatim)
# Dependencies: gliner
# Assumptions:  Labels TITLE/OTHER_TITLE/PERSON (ADR D-10, no pooling); no true k=0 pass (plan
#               §4.1 -- 50-setup.tex's systems table treats zero-shot and few-shot as separate
#               rows, not one ablation curve). Examples drawn once at n=5 from silver_tier == 1
#               spans (length-filtered 3-80 chars, deduplicated, fixed seed) then nested down for
#               k=1/k=3 (§4.3) -- isolates the effect of adding examples from the effect of which
#               examples were chosen. threshold=0.5 not tuned (§4.4). Requires network access to
#               HuggingFace Hub (or a local cache) to load --model; not runnable in a
#               network-isolated sandbox -- see plan §6.
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL = "urchade/gliner_multi-v2.1"
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_SILVER_SPANS = ROOT / "data" / "silver" / "silver_spans.jsonl"
DEFAULT_EXAMPLES_SEED = 42
DEFAULT_THRESHOLD = 0.5

MAX_EXAMPLES_DRAWN = 5  # largest k this script supports; k=1/k=3 nest inside this draw
MIN_EXAMPLE_LEN = 3  # shorter spans are likely truncation artifacts
MAX_EXAMPLE_LEN = 80  # longer spans make the label description unwieldy

LABELS = ["TITLE", "OTHER_TITLE", "PERSON"]

BASE_DESCRIPTIONS = {
    "TITLE": "main title of the bibliographic work",  # verbatim from 50-setup.tex
    "OTHER_TITLE": "subtitle or alternative title elaborating the main title",  # docs/annotation-guide.md §3.1
    "PERSON": "person named in the statement of responsibility",  # matches ADR's role-neutral PERSON design
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of records, skipping blank lines."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def draw_examples(
    silver_spans_path: Path,
    seed: int,
    labels: list[str],
    n: int = MAX_EXAMPLES_DRAWN,
    min_len: int = MIN_EXAMPLE_LEN,
    max_len: int = MAX_EXAMPLE_LEN,
) -> dict[str, list[str]]:
    """Draw a fixed-seed sample of up to n example span texts per label from tier-1 silver.

    Filters to silver_tier == 1 records, collects span texts per label, drops degenerate
    spans (too short -- likely truncation artifacts -- or too long -- unwieldy in a label
    description), deduplicates, then samples n per label once with the given seed so that
    k=1/k=3/k=5 nest (B4-gliner-fewshot-script-plan.md §4.3).
    """
    candidates: dict[str, list[str]] = {label: [] for label in labels}
    seen: dict[str, set[str]] = {label: set() for label in labels}
    for rec in read_jsonl(silver_spans_path):
        if rec.get("silver_tier") != 1:
            continue
        text = rec["text"]
        for start, end, label in rec["label"]:
            if label not in candidates:
                continue
            span_text = text[start:end]
            if not (min_len <= len(span_text) <= max_len) or span_text in seen[label]:
                continue
            seen[label].add(span_text)
            candidates[label].append(span_text)

    rng = random.Random(seed)
    return {label: rng.sample(spans, min(n, len(spans))) for label, spans in candidates.items()}


def build_label_descriptions(base: dict[str, str], examples: dict[str, list[str]], k: int) -> dict[str, str]:
    """Enrich each bare label's base description with its first k drawn example spans."""
    descriptions = {}
    for label, base_text in base.items():
        joined = ", ".join(f"'{example}'" for example in examples[label][:k])
        descriptions[label] = f"{base_text}, e.g. {joined}"
    return descriptions


def predict_spans(
    model: Any,
    text: str,
    enriched_labels: list[str],
    reverse_map: dict[str, str],
    threshold: float,
) -> list[list[Any]]:
    """Run GLiNER few-shot prediction, remapping GLiNER's echoed description labels back to bare form.

    GLiNER's `entity_types` argument becomes its classification vocabulary directly, so a
    predicted span's `entity["label"]` is the exact enriched description string passed in, not
    "TITLE"/"OTHER_TITLE"/"PERSON" (plan §4.2). `reverse_map` undoes that before this project's
    span-triple schema is written.
    """
    entities = model.predict_entities(text, enriched_labels, threshold=threshold)
    return [[e["start"], e["end"], reverse_map[e["label"]]] for e in entities]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate GLiNER few-shot predictions on gold, for eval_ner_gold.py score."
    )
    parser.add_argument("--k-shot", type=int, required=True, choices=[1, 3, 5],
                        help="Number of few-shot examples per label appended to its description.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace Hub model id or local checkpoint path.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--silver-spans", type=Path, default=DEFAULT_SILVER_SPANS)
    parser.add_argument("--examples-seed", type=int, default=DEFAULT_EXAMPLES_SEED)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="GLiNER's own default, not tuned")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to data/eval/predictions_gliner_fewshot_k{K}.jsonl")
    parser.add_argument("--examples-output", type=Path, default=None,
                        help="Defaults to data/eval/fewshot_examples_k{K}.json")
    args = parser.parse_args()

    output = args.output or ROOT / "data" / "eval" / f"predictions_gliner_fewshot_k{args.k_shot}.jsonl"
    examples_output = args.examples_output or ROOT / "data" / "eval" / f"fewshot_examples_k{args.k_shot}.json"

    logging.info("Drawing up to %d-shot examples per label from %s (seed=%d)",
                 MAX_EXAMPLES_DRAWN, args.silver_spans, args.examples_seed)
    examples = draw_examples(args.silver_spans, args.examples_seed, LABELS)
    for label, spans in examples.items():
        if len(spans) < args.k_shot:
            raise ValueError(f"Only {len(spans)} tier-1 examples available for {label}, need {args.k_shot}")

    examples_output.parent.mkdir(parents=True, exist_ok=True)
    with open(examples_output, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    logging.info("Wrote drawn examples -> %s", examples_output)

    label_descriptions = build_label_descriptions(BASE_DESCRIPTIONS, examples, args.k_shot)
    enriched_labels = list(label_descriptions.values())
    reverse_map = {description: label for label, description in label_descriptions.items()}

    logging.info("Loading gold from %s", args.gold)
    gold_records = read_jsonl(args.gold)
    logging.info("%d gold records to predict on.", len(gold_records))

    logging.info("Loading GLiNER model %s", args.model)
    from gliner import GLiNER  # type: ignore[import-untyped]
    model = GLiNER.from_pretrained(args.model)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for rec in gold_records:
            predicted_spans = predict_spans(model, rec["text"], enriched_labels, reverse_map, args.threshold)
            f.write(json.dumps({"id": rec["id"], "spans": predicted_spans}, ensure_ascii=False) + "\n")

    logging.info("Wrote %d predictions -> %s", len(gold_records), output)


if __name__ == "__main__":
    main()
