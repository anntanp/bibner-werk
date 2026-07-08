#!/usr/bin/env python3
# Purpose:      Compute Krippendorff's alpha (nominal) for BibNER-Werk gold annotation
# Usage:        python scripts/ner/sr08_compute_iaa.py \
#                   --export-dir data/annotation/doccano/export-20260623-1110
# Inputs:       *.jsonl files in --export-dir (skips dummy.doccano.jsonl, ann.jsonl, *.meta.jsonl, sr08_*)
# Outputs:      data/iaa/sr08_iaa_counts_<export>.csv        (raw label assignments)
#               data/iaa/sr08_iaa_<export>.csv                (Krippendorff's alpha per scope)
#               data/iaa/sr08_iaa_pairwise_<export>.csv       (pairwise alpha for all annotator pairs)
#               data/iaa/sr08_iaa_span_pairwise_<export>.csv  (pairwise span agreement: exact-match %, mean Jaccard)
#               stdout: annotator coverage, alpha table (overall + per label), pairwise tables, span tables, pass/fail
# Dependencies: none (stdlib only)
# Assumptions:  reviewed = label != [] OR Comments != []; empty-label+comment = O assignment
#               span agreement (ADR-02) is conditional on existence agreement: only (pair, doc,
#               label) triples where both annotators marked the label are compared for span
#               overlap; not chance-corrected (see notes/fiz/bibner-werk/adr-02-plan.md)

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

SKIP_NAMES    = {"dummy.doccano.jsonl", "ann.jsonl", "harald.jsonl"}
SKIP_SUFFIXES = (".meta.jsonl",)
SKIP_PREFIXES = ("sr08_",)

ALPHA_THRESHOLD = 0.800


# ── Data loading ──────────────────────────────────────────────────────────────

def get_labels(record: dict) -> list[str] | None:
    """Return deduplicated label types for a reviewed record, or None if unreviewed."""
    reviewed = bool(record["label"]) or bool(record.get("Comments"))
    if not reviewed:
        return None
    types = {span[2] for span in record["label"]}
    return list(types) if types else ["O"]


def load_annotator(path: Path) -> dict[int, list[str]]:
    """Return {doc_id: [label_types]} for reviewed records only."""
    result: dict[int, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels = get_labels(rec)
            if labels is not None:
                result[rec["id"]] = labels
    return result


def load_data(export_dir: Path) -> dict[str, dict[int, list[str]]]:
    data: dict[str, dict[int, list[str]]] = {}
    for path in sorted(export_dir.glob("*.jsonl")):
        name = path.name
        if (name in SKIP_NAMES
                or any(name.endswith(s) for s in SKIP_SUFFIXES)
                or any(name.startswith(p) for p in SKIP_PREFIXES)):
            continue
        records = load_annotator(path)
        if records:
            data[path.stem] = records
        else:
            print(f"  [skip] {path.stem}: 0 reviewed records", file=sys.stderr)
    return data


# ── Span loading (ADR-02) ───────────────────────────────────────────────────────

def get_label_spans(record: dict) -> dict[str, set[int]] | None:
    """Return {label_type: set(char positions covered)} for a reviewed record, or None if
    unreviewed. Multiple spans of the same label type in one doc are unioned into one
    position set (a doc with two separate TITLE spans yields one TITLE key covering both).
    """
    reviewed = bool(record["label"]) or bool(record.get("Comments"))
    if not reviewed:
        return None
    spans: defaultdict[str, set[int]] = defaultdict(set)
    for start, end, label in record["label"]:
        spans[label].update(range(start, end))
    return dict(spans)


def load_annotator_spans(path: Path) -> dict[int, dict[str, set[int]]]:
    """Return {doc_id: {label_type: char_position_set}} for reviewed records only."""
    result: dict[int, dict[str, set[int]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            spans = get_label_spans(rec)
            if spans is not None:
                result[rec["id"]] = spans
    return result


# ── n_ik matrix ───────────────────────────────────────────────────────────────

def build_n(
    data: dict[str, dict[int, list[str]]],
) -> tuple[defaultdict[int, Counter], Counter]:
    """Build n[doc_id][label] (count of annotators per label) and pooled n_k."""
    n: defaultdict[int, Counter] = defaultdict(Counter)
    n_k: Counter = Counter()
    for records in data.values():
        for doc_id, labels in records.items():
            for lbl in labels:
                n[doc_id][lbl] += 1
                n_k[lbl] += 1
    return n, n_k


# ── Krippendorff's alpha ──────────────────────────────────────────────────────

def compute_alpha(
    n: defaultdict[int, Counter],
    n_k: Counter,
    scope: str = "ALL",
) -> dict:
    """Compute Krippendorff's alpha (nominal distance) from n_ik and pooled n_k.

    Formula: alpha = 1 - D_o / D_e
    Nominal simplification: sum_{j!=l} n_ij * n_il = ci^2 - sum_k n_ik^2
    Source: Artstein & Poesio (2008) §2.6.1
    """
    N = sum(n_k.values())

    Do_num = sum(
        ci ** 2 - sum(v ** 2 for v in n[i].values())
        for i in n
        if (ci := sum(n[i].values())) > 1
    )
    Do_den = sum(
        ci * (ci - 1)
        for i in n
        if (ci := sum(n[i].values())) > 1
    )
    if Do_den == 0:
        raise ValueError(
            f"Do_den=0 for scope '{scope}': every reviewed document was seen "
            "by only one annotator — alpha undefined"
        )
    D_o = Do_num / Do_den

    D_e = (N ** 2 - sum(v ** 2 for v in n_k.values())) / (N * (N - 1))

    if D_e == 0:
        warnings.warn(
            f"D_e=0 for scope '{scope}': all annotations are the same label — alpha set to 1.0"
        )
        alpha = 1.0
    else:
        alpha = 1.0 - D_o / D_e

    return {
        "label":       scope,
        "n_items":     len(n),
        "N_judgments": N,
        "D_o":         D_o,
        "D_e":         D_e,
        "alpha":       alpha,
    }


def per_label_alpha(
    data: dict[str, dict[int, list[str]]],
    label: str,
) -> dict:
    """Compute alpha for a single label L vs O.

    For each reviewed document, each annotator who reviewed it contributes:
      - L  if they assigned label L (regardless of other labels on that doc)
      - O  if they reviewed the doc but did not assign L
    """
    reviewed_by: dict[str, set[int]] = {
        annotator: set(records.keys())
        for annotator, records in data.items()
    }
    all_doc_ids: set[int] = set().union(*reviewed_by.values())

    n: defaultdict[int, Counter] = defaultdict(Counter)
    n_k: Counter = Counter()

    for annotator, records in data.items():
        for doc_id in all_doc_ids:
            if doc_id not in reviewed_by[annotator]:
                continue
            assigned = label if label in records[doc_id] else "O"
            n[doc_id][assigned] += 1
            n_k[assigned] += 1

    return compute_alpha(n, n_k, scope=label)


# ── Pairwise alpha ────────────────────────────────────────────────────────────

def compute_pairwise(
    data: dict[str, dict[int, list[str]]],
) -> list[tuple[str, list[dict]]]:
    """Return pairwise alpha rows for every annotator pair."""
    result = []
    for a1, a2 in sorted(combinations(sorted(data), 2)):
        pair_name = f"{a1}/{a2}"
        pair_data = {a1: data[a1], a2: data[a2]}
        n, n_k = build_n(pair_data)
        rows: list[dict] = [compute_alpha(n, n_k, scope="ALL")]
        for label in sorted(k for k in n_k if k != "O"):
            try:
                rows.append(per_label_alpha(pair_data, label))
            except ValueError as e:
                print(f"  [warn] {pair_name} {e}", file=sys.stderr)
        result.append((pair_name, rows))
    return result


# ── Span agreement (ADR-02) ──────────────────────────────────────────────────────

def span_stats(a: set[int], b: set[int]) -> tuple[bool, float]:
    """Return (exact_match, jaccard) for two char-position sets. Both empty -> (True, 1.0)."""
    if not a and not b:
        return True, 1.0
    exact = a == b
    union = a | b
    jaccard = len(a & b) / len(union) if union else 1.0
    return exact, jaccard


def compute_span_agreement(
    span_data: dict[str, dict[int, dict[str, set[int]]]],
) -> list[tuple[str, list[dict]]]:
    """Return span-agreement rows (ALL + per label) for every annotator pair.

    Restricts to (doc, label) where BOTH annotators in the pair marked the label present
    (existence agreement) -- see ADR-02. A doc where one annotator marked the label and the
    other didn't is excluded here; that's an existence disagreement, already measured by the
    per-label alpha in compute_pairwise().

    ALL pools every existence-agreed (doc, label) comparison across all label types for the
    pair -- computed from the raw counts, not by averaging the per-label percentages, mirroring
    how the existing alpha ALL scope pools raw n_ik counts rather than averaging per-label alpha.
    """
    result = []
    for a1, a2 in sorted(combinations(sorted(span_data), 2)):
        pair_name = f"{a1}/{a2}"
        common_docs = set(span_data[a1]) & set(span_data[a2])
        labels = {
            lbl
            for doc_id in common_docs
            for lbl in set(span_data[a1][doc_id]) & set(span_data[a2][doc_id])
        }

        rows: list[dict] = []
        all_matches = 0
        all_jaccards: list[float] = []

        for label in sorted(labels):
            matches = 0
            jaccards: list[float] = []
            for doc_id in common_docs:
                s1 = span_data[a1][doc_id].get(label)
                s2 = span_data[a2][doc_id].get(label)
                if s1 is None or s2 is None:
                    continue  # existence disagreement -- excluded, not a span comparison
                exact, jac = span_stats(s1, s2)
                matches += exact
                jaccards.append(jac)
            n = len(jaccards)
            if n == 0:
                continue
            rows.append({
                "label": label,
                "n_existence_agreed": n,
                "n_exact_match": matches,
                "pct_exact_match": matches / n,
                "mean_jaccard": sum(jaccards) / n,
            })
            all_matches += matches
            all_jaccards.extend(jaccards)

        n_all = len(all_jaccards)
        all_row = {
            "label": "ALL",
            "n_existence_agreed": n_all,
            "n_exact_match": all_matches,
            "pct_exact_match": all_matches / n_all if n_all else float("nan"),
            "mean_jaccard": sum(all_jaccards) / n_all if n_all else float("nan"),
        }
        result.append((pair_name, [all_row, *rows]))
    return result


# ── Output ────────────────────────────────────────────────────────────────────

def write_counts_csv(
    data: dict[str, dict[int, list[str]]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "annotator", "label"])
        writer.writeheader()
        for annotator in sorted(data):
            for doc_id in sorted(data[annotator]):
                for lbl in sorted(data[annotator][doc_id]):
                    writer.writerow({"doc_id": doc_id, "annotator": annotator, "label": lbl})
    print(f"Counts  -> {out_path}")


def write_scores_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "n_items", "N_judgments", "D_o", "D_e", "alpha"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scores  -> {out_path}")


def write_pairwise_csv(
    pairwise: list[tuple[str, list[dict]]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair", "label", "n_items", "N_judgments", "D_o", "D_e", "alpha"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair_name, rows in pairwise:
            for r in rows:
                writer.writerow({"pair": pair_name, **r})
    print(f"Pairwise -> {out_path}")


def write_span_csv(
    span_pairwise: list[tuple[str, list[dict]]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair", "label", "n_existence_agreed", "n_exact_match", "pct_exact_match", "mean_jaccard"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair_name, rows in span_pairwise:
            for r in rows:
                writer.writerow({"pair": pair_name, **r})
    print(f"Span    -> {out_path}")


def print_pairwise_summary(pairwise: list[tuple[str, list[dict]]]) -> None:
    print("\n-- Pairwise Krippendorff's alpha (nominal) ----------------------")
    for pair_name, rows in pairwise:
        print(f"\n  Pair: {pair_name}")
        print(f"  {'Label':<16} {'alpha':>7}  {'n_items':>7}  {'N_judg':>7}  {'D_o':>8}  {'D_e':>8}")
        print(f"  {'-'*16} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")
        for r in rows:
            flag = "  *" if r["alpha"] < ALPHA_THRESHOLD else ""
            print(
                f"  {r['label']:<16} {r['alpha']:>7.3f}  {r['n_items']:>7}  "
                f"{r['N_judgments']:>7}  {r['D_o']:>8.4f}  {r['D_e']:>8.4f}{flag}"
            )
    print()


def print_span_summary(span_pairwise: list[tuple[str, list[dict]]]) -> None:
    print("\n-- Pairwise span agreement (exact-match, Jaccard) ----------------")
    for pair_name, rows in span_pairwise:
        print(f"\n  Pair: {pair_name}")
        print(f"  {'Label':<16} {'n_exist':>8}  {'n_exact':>8}  {'exact%':>7}  {'jaccard':>7}")
        print(f"  {'-'*16} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")
        for r in rows:
            print(
                f"  {r['label']:<16} {r['n_existence_agreed']:>8}  {r['n_exact_match']:>8}  "
                f"{r['pct_exact_match']*100:>6.1f}%  {r['mean_jaccard']:>7.3f}"
            )
    print()


def print_summary(
    data: dict[str, dict[int, list[str]]],
    score_rows: list[dict],
) -> None:
    print("\n-- Annotator coverage -------------------------------------------")
    print(f"  {'Annotator':<12} {'Reviewed':>8}")
    print(f"  {'-'*12} {'-'*8}")
    for annotator in sorted(data):
        print(f"  {annotator:<12} {len(data[annotator]):>8}")

    print("\n-- Krippendorff's alpha (nominal) -------------------------------")
    print(f"  {'Label':<16} {'alpha':>7}  {'n_items':>7}  {'N_judg':>7}  {'D_o':>8}  {'D_e':>8}")
    print(f"  {'-'*16} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")
    for r in score_rows:
        flag = "  *" if r["alpha"] < ALPHA_THRESHOLD else ""
        print(
            f"  {r['label']:<16} {r['alpha']:>7.3f}  {r['n_items']:>7}  "
            f"{r['N_judgments']:>7}  {r['D_o']:>8.4f}  {r['D_e']:>8.4f}{flag}"
        )

    overall = next(r for r in score_rows if r["label"] == "ALL")
    print()
    if overall["alpha"] >= ALPHA_THRESHOLD:
        print(f"  PASS  alpha={overall['alpha']:.3f} >= {ALPHA_THRESHOLD}")
    else:
        print(f"  FAIL  alpha={overall['alpha']:.3f} < {ALPHA_THRESHOLD}  (Krippendorff minimum for reliable data)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Krippendorff's alpha (nominal) for BibNER-Werk gold annotation"
    )
    parser.add_argument(
        "--export-dir", required=True, type=Path,
        help="Path to Doccano export directory (read-only)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BASE / "data/iaa",
        help="Output directory (default: data/iaa/)"
    )
    args = parser.parse_args()

    export_dir: Path = args.export_dir
    if not export_dir.is_dir():
        sys.exit(f"Not a directory: {export_dir}")

    export_stem = export_dir.name

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading from: {export_dir}")
    data = load_data(export_dir)
    if len(data) < 2:
        sys.exit(f"Need >= 2 annotators with reviewed records; found {len(data)}: {list(data)}")
    print(f"Annotators:   {sorted(data)}")

    # ── Counts CSV ───────────────────────────────────────────────────────────
    counts_path = args.output_dir / f"sr08_iaa_counts_{export_stem}.csv"
    write_counts_csv(data, counts_path)

    # ── Build global n_ik ────────────────────────────────────────────────────
    n, n_k = build_n(data)

    # ── Overall alpha ────────────────────────────────────────────────────────
    score_rows: list[dict] = [compute_alpha(n, n_k, scope="ALL")]

    # ── Per-label alpha ──────────────────────────────────────────────────────
    for label in sorted(k for k in n_k if k != "O"):
        try:
            score_rows.append(per_label_alpha(data, label))
        except ValueError as e:
            print(f"  [warn] {e}", file=sys.stderr)

    # ── Scores CSV ───────────────────────────────────────────────────────────
    scores_path = args.output_dir / f"sr08_iaa_{export_stem}.csv"
    write_scores_csv(score_rows, scores_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(data, score_rows)

    # ── Pairwise alpha ───────────────────────────────────────────────────────
    pairwise = compute_pairwise(data)
    pairwise_path = args.output_dir / f"sr08_iaa_pairwise_{export_stem}.csv"
    write_pairwise_csv(pairwise, pairwise_path)
    print_pairwise_summary(pairwise)

    # ── Span agreement (ADR-02) ────────────────────────────────────────────────
    span_data = {
        annotator: load_annotator_spans(export_dir / f"{annotator}.jsonl")
        for annotator in data
    }
    span_pairwise = compute_span_agreement(span_data)
    span_path = args.output_dir / f"sr08_iaa_span_pairwise_{export_stem}.csv"
    write_span_csv(span_pairwise, span_path)
    print_span_summary(span_pairwise)


if __name__ == "__main__":
    main()
