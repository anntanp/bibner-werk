#!/usr/bin/env python3
# Purpose:      Layer-2 verification (plan §8.2) for build_silver_spans.py's f_resp_*
#               sub-classifier (ADR-06). classify_resp() has never been run against
#               real ground truth — this script checks it against the 100-record
#               SR-04 validation sample (sr04_translator_validation_sample.csv,
#               true_class: PERSON/CORPORATE/EDITOR/OTHER, human-annotated), before
#               it's trusted to label PERSON spans across the full silver set.
# Usage:        python3 scripts/py/verify_resp_classifier.py [--sample PATH]
# Inputs:       data/processed/sr04_translator_validation_sample.csv (sor_text,
#               true_class columns) — copied into bibner-werk from gemea on
#               2026-07-06 (a P2-migration straggler: the other sr04 outputs
#               already lived in bibner-werk, this one alone was left behind).
# Outputs:      stdout — confusion matrix + per-class precision/recall/F1
# Dependencies: pandas
# Assumptions:  true_class in {PERSON, CORPORATE, EDITOR, OTHER} only — the sample
#               predates f_resp_family (ADR-06 notes it as "not yet validated"), so
#               a FAMILY prediction here can only ever be a miss, never confirmed.
#               Imports classify_resp_batch directly from build_silver_spans.py
#               rather than reimplementing it — a deliberate exception to the
#               sr-series "no cross-script imports" convention
#               (adjudication-plan.md §2.1): the entire point of this script
#               is to test the live classifier (FLERT + keyword regexes,
#               docs/bibner-adr.md D-04), so a duplicated copy would silently
#               drift out of sync and this check would stop meaning anything.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_silver_spans import classify_resp_batch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DEFAULT = ROOT / "data" / "processed" / "sr04_translator_validation_sample.csv"

# classify_resp() output -> SR-04 sample's true_class taxonomy
RESP_TO_TRUE_CLASS = {
    "f_resp_person": "PERSON",
    "f_resp_org": "CORPORATE",
    "f_resp_editor": "EDITOR",
    "f_resp_other": "OTHER",
    "f_resp_family": "FAMILY",  # not in the sample's taxonomy; always a miss here
}


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(df: pd.DataFrame) -> None:
    classes = sorted(set(df["true_class"]) | set(df["predicted_class"]))

    print("=" * 64)
    print("Per-class precision / recall / F1")
    print("=" * 64)
    print(f"{'Class':<12}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print("-" * 64)
    for cls in classes:
        tp = int(((df["predicted_class"] == cls) & (df["true_class"] == cls)).sum())
        fp = int(((df["predicted_class"] == cls) & (df["true_class"] != cls)).sum())
        fn = int(((df["predicted_class"] != cls) & (df["true_class"] == cls)).sum())
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        flag = "  <-- gates PERSON silver spans" if cls == "PERSON" else ""
        print(f"{cls:<12}  {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}  {tp:>4}  {fp:>4}  {fn:>4}{flag}")
    print()

    print("=" * 64)
    print("Confusion matrix  (rows=true_class, cols=predicted_class)")
    print("=" * 64)
    matrix = pd.crosstab(df["true_class"], df["predicted_class"], rownames=["true \\ pred"], colnames=[""])
    matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)
    print(matrix.to_string())
    print()

    print_person_false_positives(df)


def print_person_false_positives(df: pd.DataFrame) -> None:
    """Detail rows where classify_resp() says PERSON but the true label isn't.

    This is the failure mode that matters most for build_silver_spans.py: a
    false-positive PERSON prediction becomes a mislabeled span in the silver
    training set, not just a wrong stat here.
    """
    fp_mask = (df["predicted_class"] == "PERSON") & (df["true_class"] != "PERSON")
    fp_rows = df[fp_mask]

    print("=" * 64)
    print(f"PERSON false positives (predicted PERSON, true != PERSON): {len(fp_rows)}")
    print("=" * 64)
    if fp_rows.empty:
        print("None.")
        return
    for _, row in fp_rows.iterrows():
        print(f"  [true={row['true_class']:<10}] sor: {row['sor_text'][:70]!r}")
        if pd.notna(row.get("notes")) and row.get("notes"):
            print(f"    note: {row['notes']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify build_silver_spans.py's f_resp_* classifier against SR-04 ground truth."
    )
    parser.add_argument("--sample", type=Path, default=SAMPLE_DEFAULT,
                        help="Path to sr04_translator_validation_sample.csv (default: %(default)s)")
    args = parser.parse_args()

    df = pd.read_csv(args.sample, dtype=str)

    blank_mask = df["true_class"].isna() | (df["true_class"].str.strip() == "")
    if blank_mask.any():
        print(f"WARNING: {blank_mask.sum()} row(s) with blank true_class dropped.", file=sys.stderr)
        df = df[~blank_mask].copy()
    if df.empty:
        print("ERROR: no annotated rows to evaluate.", file=sys.stderr)
        sys.exit(1)

    predicted = classify_resp_batch(df["sor_text"].fillna("").tolist())
    df["predicted_class"] = pd.Series(predicted, index=df.index).map(RESP_TO_TRUE_CLASS)
    evaluate(df)


if __name__ == "__main__":
    main()
