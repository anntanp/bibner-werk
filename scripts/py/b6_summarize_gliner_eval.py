#!/usr/bin/env python3
# Purpose:      Summarize eval_ner_gold.py's eval_summary.csv into a support-weighted
#               F1 per label for one system, plus a flag list of near-zero cells —
#               the numbers needed to decide whether current silver is good enough to
#               proceed with the data-scaling sweep, or bad enough to prioritize
#               A2-silver-spans-revision-plan.md first. See B6-gliner-eval.md.
#               Originally B6-only (fine-tuned GLiNER); eval_summary.csv now also
#               carries B3's gliner_zero rows in the same file (2026-07-14), so
#               --system was added to filter explicitly -- without it, rows from
#               different systems would silently average together into a
#               meaningless number. Default stays "gliner" so existing B6 callers
#               are unaffected.
# Usage:        python3 scripts/py/b6_summarize_gliner_eval.py
#                   [--eval-summary PATH] [--system STR] [--near-zero-threshold FLOAT]
#                   [--output PATH]
# Inputs:       data/eval/eval_summary.csv (eval_ner_gold.py aggregate output)
# Outputs:      data/processed/b6_gliner_eval_weighted_summary.csv by default
#                   (label, weighted_f1, n_gold_spans, plus an OVERALL row); pass
#                   --output explicitly when summarizing a non-default --system to
#                   avoid overwriting another system's summary
# Dependencies: none (stdlib csv only)
# Assumptions:  Only TITLE/OTHER_TITLE/PERSON are in this project's Phase 1 label scope
#               (30-corpus.tex Table labelset); PARALLEL_TITLE/TRANSLATOR rows in
#               eval_summary.csv are artifacts of gold's richer original annotation
#               schema (the model was never trained to predict them, so their F1 is
#               always exactly 0 by construction) -- excluded from the weighted
#               average, reported separately as a known scope gap, not a failure.
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVAL_SUMMARY = ROOT / "data" / "eval" / "eval_summary.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "b6_gliner_eval_weighted_summary.csv"
DEFAULT_SYSTEM = "gliner"

CORE_LABELS = ("TITLE", "OTHER_TITLE", "PERSON")


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def weighted_f1_by_label(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Support-weighted (by n_gold_spans) mean F1 per core label, plus an OVERALL row."""
    out: list[dict[str, object]] = []
    grand_num, grand_den = 0.0, 0
    for label in CORE_LABELS:
        label_rows = [r for r in rows if r["label"] == label]
        num = sum(float(r["mean_f1"]) * int(r["n_gold_spans"]) for r in label_rows)
        den = sum(int(r["n_gold_spans"]) for r in label_rows)
        out.append({"label": label, "weighted_f1": round(num / den, 4), "n_gold_spans": den})
        grand_num += num
        grand_den += den
    out.append({"label": "OVERALL", "weighted_f1": round(grand_num / grand_den, 4), "n_gold_spans": grand_den})
    return out


def near_zero_cells(rows: list[dict[str, str]], threshold: float) -> list[dict[str, str]]:
    """Core-label (era, label) cells whose mean_f1 is below threshold -- not small-cell-flagged noise."""
    return [
        r for r in rows
        if r["label"] in CORE_LABELS
        and float(r["mean_f1"]) < threshold
        and r["flagged_small_cell"] == "False"
    ]


def out_of_scope_labels(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """PARALLEL_TITLE/TRANSLATOR rows -- always F1=0 by construction, not part of this scope."""
    return [r for r in rows if r["label"] not in CORE_LABELS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a system's GLiNER-vs-gold eval_summary.csv rows.")
    parser.add_argument("--eval-summary", type=Path, default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM,
                         help="eval_summary.csv 'system' column value to filter on "
                              "(default: %(default)s, e.g. also 'gliner_zero')")
    parser.add_argument("--near-zero-threshold", type=float, default=0.06)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_rows = load_rows(args.eval_summary)
    rows = [r for r in all_rows if r["system"] == args.system]
    if not rows:
        known_systems = sorted({r["system"] for r in all_rows})
        sys.exit(f"b6_summarize_gliner_eval.py: no rows for --system {args.system!r} in "
                  f"{args.eval_summary} (systems present: {known_systems})")

    weighted = weighted_f1_by_label(rows)
    near_zero = near_zero_cells(rows, args.near_zero_threshold)
    out_of_scope = out_of_scope_labels(rows)

    print(f"{'label':<12} {'weighted F1':>12} {'n_gold_spans':>14}")
    for r in weighted:
        print(f"{r['label']:<12} {r['weighted_f1']:>12} {r['n_gold_spans']:>14}")

    print(f"\nNear-zero cells (F1 < {args.near_zero_threshold}, core labels, not small-cell-flagged):")
    for nz in near_zero:
        print(f"  {nz['label']:<12} {nz['era']:<10} F1={nz['mean_f1']:<8} n_gold_spans={nz['n_gold_spans']}")

    print("\nOut-of-scope labels (never trained, F1=0 by construction, excluded from weighted average):")
    for oos in out_of_scope:
        print(f"  {oos['label']:<15} {oos['era']:<10} n_gold_spans={oos['n_gold_spans']} (flagged_small_cell={oos['flagged_small_cell']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "weighted_f1", "n_gold_spans"])
        writer.writeheader()
        writer.writerows(weighted)
    print(f"\nWrote weighted summary -> {args.output}")


if __name__ == "__main__":
    main()
