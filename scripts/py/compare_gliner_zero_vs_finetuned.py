#!/usr/bin/env python3
# Purpose:      Combine two systems' support-weighted per-label F1 (from
#               eval_ner_gold.py's eval_summary.csv) into one side-by-side comparison
#               CSV -- the source for B3-gliner-zeroshot-eval.md §3/§4's comparison
#               table (zero-shot B3 vs. fine-tuned B6). Written because that table was
#               first hand-assembled in prose from two separately-generated weighted-
#               summary files; this script produces one real, rerunnable artifact
#               instead, removing the transcription risk.
# Usage:        python3 scripts/py/compare_gliner_zero_vs_finetuned.py
#                   [--eval-summary PATH] [--system-a STR] [--system-b STR]
#                   [--output PATH]
# Inputs:       data/eval/eval_summary.csv (eval_ner_gold.py aggregate output;
#                   must contain rows for both --system-a and --system-b)
# Outputs:      data/processed/b3_vs_b6_gliner_weighted_comparison.csv
#                   (label, weighted_f1_<system-a>, weighted_f1_<system-b>,
#                   n_gold_spans, plus an OVERALL row)
# Dependencies: none (stdlib csv only); imports load_rows/weighted_f1_by_label from
#                   b6_summarize_gliner_eval.py (reuse the live per-system logic, not
#                   a frozen copy -- same cross-script-import exception as
#                   verify_resp_classifier.py/spotcheck_xlmr_predictions.py)
# Assumptions:  n_gold_spans is a property of the gold set, not the system under
#                   evaluation, so it must be identical for both systems at every
#                   label -- asserted explicitly per row rather than silently trusted,
#                   since a mismatch would mean the two systems were scored against
#                   different fold/gold slices.
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b6_summarize_gliner_eval import load_rows, weighted_f1_by_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVAL_SUMMARY = ROOT / "data" / "eval" / "eval_summary.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "b3_vs_b6_gliner_weighted_comparison.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-summary", type=Path, default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--system-a", type=str, default="gliner_zero")
    parser.add_argument("--system-b", type=str, default="gliner")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_rows = load_rows(args.eval_summary)
    rows_a = [r for r in all_rows if r["system"] == args.system_a]
    rows_b = [r for r in all_rows if r["system"] == args.system_b]
    if not rows_a:
        sys.exit(f"compare_gliner_zero_vs_finetuned.py: no rows for --system-a {args.system_a!r}")
    if not rows_b:
        sys.exit(f"compare_gliner_zero_vs_finetuned.py: no rows for --system-b {args.system_b!r}")

    weighted_a = {r["label"]: r for r in weighted_f1_by_label(rows_a)}
    weighted_b = {r["label"]: r for r in weighted_f1_by_label(rows_b)}

    out_rows = []
    for label in weighted_a:
        a, b = weighted_a[label], weighted_b[label]
        if a["n_gold_spans"] != b["n_gold_spans"]:
            sys.exit(
                f"compare_gliner_zero_vs_finetuned.py: n_gold_spans mismatch for {label!r} "
                f"({args.system_a}={a['n_gold_spans']} vs {args.system_b}={b['n_gold_spans']}) "
                "-- systems were scored against different gold/fold slices, refusing to compare."
            )
        out_rows.append({
            "label": label,
            f"weighted_f1_{args.system_a}": a["weighted_f1"],
            f"weighted_f1_{args.system_b}": b["weighted_f1"],
            "n_gold_spans": a["n_gold_spans"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"{'label':<12} {'weighted_f1_' + args.system_a:>22} {'weighted_f1_' + args.system_b:>18} {'n_gold_spans':>14}")
    for r in out_rows:
        print(f"{r['label']:<12} {r[f'weighted_f1_{args.system_a}']:>22} "
              f"{r[f'weighted_f1_{args.system_b}']:>18} {r['n_gold_spans']:>14}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
