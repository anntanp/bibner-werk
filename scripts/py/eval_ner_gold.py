#!/usr/bin/env python3
# Purpose:      A3 — formal k-fold evaluation for BibNER-Werk NER systems.
#               Three subcommands, matching the split -> score -> aggregate
#               pipeline: split-folds (stratified k-fold assignment over the
#               reconciled gold set), score (per-fold per-system exact-span-
#               match P/R/F1), aggregate (cross-fold mean +/- std and 95%
#               bootstrap CI, merged into the paper-ready summary table).
#               See notes/fiz/bibner-werk/A3-eval-ner-gold-script-plan.md.
# Usage:        python3 scripts/py/eval_ner_gold.py split-folds
#                   --gold PATH --gold-sample PATH --export-dir PATH
#                   --k INT [--seed INT] [--output PATH]
#               python3 scripts/py/eval_ner_gold.py score
#                   --gold PATH --folds PATH --predictions PATH
#                   --system STR --fold INT [--output PATH]
#               python3 scripts/py/eval_ner_gold.py aggregate
#                   [--eval-dir PATH] [--iaa-summary PATH]
#                   [--bootstrap-resamples INT] [--ci FLOAT] [--output PATH]
# Inputs:       data/gold/gold_reconciled.jsonl (A7 output, not yet
#                   built as of 2026-07-07 -- see plan for a synthetic fixture
#                   to develop and test against in the meantime)
#               data/annotation/sr08_gold_sample.csv (era/silver_tier join)
#               data/annotation/doccano/export-*/*.jsonl (Doccano id -> obj_id
#                   join, via each record's "url" field)
#               data/eval/predictions_{system}_fold{f}.jsonl (B2-B6
#                   output, not yet built)
# Outputs:      data/eval/folds_k{k}.json
#               data/eval/eval_fold{f}_{system}.json
#               data/eval/eval_summary.csv
# Dependencies: scikit-learn, numpy
# Assumptions:  gold_reconciled.jsonl ids are Doccano ids shared with the
#               per-annotator export directory used to resolve obj_id (last
#               path segment of each record's "url"); span_key()/compute_f1()
#               match scripts/ner/sr09_eval_nuner_tier2.py's convention.
#               No pooling across labels (ADR D-10): every P/R/F1 figure is
#               reported per label, never averaged together. Two thresholds
#               govern stratification, not one: the plan's 5-records-per-fold
#               bar (data-quality flag, reported downstream) is separate from
#               StratifiedKFold's hard requirement of >= k members per class
#               (or the split simply cannot run) -- see build_stratification_
#               keys()'s docstring.

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_GOLD_SAMPLE = ROOT / "data" / "annotation" / "sr08_gold_sample.csv"
DEFAULT_EXPORT_DIR = ROOT / "data" / "annotation" / "doccano" / "export-20260623-1110"
DEFAULT_EVAL_DIR = ROOT / "data" / "eval"

LABELS = ["TITLE", "OTHER_TITLE", "PERSON"]  # ADR D-10: never pooled across labels


# ── Shared helpers ───────────────────────────────────────────────────────────

def compute_f1(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Same formula as scripts/ner/sr09_eval_nuner_tier2.py's compute_f1()."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)}


def load_gold_records(gold_path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def load_obj_id_by_doc_id(export_dir: Path) -> dict[int, str]:
    """Map Doccano `id` -> obj_id (last path segment of each record's `url`).

    Mirrors scripts/ner/sr08_lookup_record_provenance.py's join logic.
    """
    result: dict[int, str] = {}
    for path in sorted(export_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] not in result and rec.get("url"):
                    result[rec["id"]] = rec["url"].rstrip("/").rsplit("/", 1)[-1]
    return result


def load_era_tier_by_obj_id(gold_sample_path: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    with open(gold_sample_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["obj_id"]] = (row["era"], row["silver_tier"])
    return result


def join_era_tier(
    doc_ids: list[int], export_dir: Path, gold_sample_path: Path,
) -> tuple[list[str], list[str]]:
    """For each doc_id, resolve (era, silver_tier) via id -> obj_id -> era/tier.

    Fails loudly (not a silent drop or default) on any unresolvable id, per
    the plan's testing contract (A3-eval-ner-gold-script-plan.md §7).
    """
    obj_id_by_doc_id = load_obj_id_by_doc_id(export_dir)
    era_tier_by_obj_id = load_era_tier_by_obj_id(gold_sample_path)

    eras, tiers = [], []
    for doc_id in doc_ids:
        obj_id = obj_id_by_doc_id.get(doc_id)
        if obj_id is None:
            sys.exit(
                f"eval_ner_gold.py: doc id {doc_id} not found in any export "
                f"JSONL under {export_dir} (needed to resolve obj_id)"
            )
        era_tier = era_tier_by_obj_id.get(obj_id)
        if era_tier is None:
            sys.exit(
                f"eval_ner_gold.py: obj_id {obj_id} (doc id {doc_id}) not "
                f"found in {gold_sample_path} (needed for era/silver_tier)"
            )
        era, tier = era_tier
        eras.append(era)
        tiers.append(tier)
    return eras, tiers


def build_stratification_keys(
    eras: list[str], tiers: list[str], k: int,
) -> tuple[list[str], list[bool]]:
    """Build StratifiedKFold class labels from era x silver_tier, and flag cells
    too small to report reliably.

    Two separate thresholds are in play, and conflating them would either
    under-warn or crash the split:
      - Reporting bar (plan §3): a cell averaging < 5 records/fold at the
        chosen k is flagged `flagged_small_cell` for downstream reporting,
        but is NOT collapsed for that reason alone.
      - StratifiedKFold's hard requirement: every class must have >= k
        members or the split raises. Only cells failing *this* bar are
        collapsed -- first to their era (merging tiers), and if an era-level
        bucket is still under k, into one shared fallback class -- so the
        split can always run.
    """
    pair_counts = Counter(zip(eras, tiers))
    flagged_small_cell = [pair_counts[(e, t)] < 5 * k for e, t in zip(eras, tiers)]

    keys = [
        f"{e}__{t}" if pair_counts[(e, t)] >= k else e
        for e, t in zip(eras, tiers)
    ]
    key_counts = Counter(keys)
    keys = [key if key_counts[key] >= k else "__collapsed_other__" for key in keys]

    return keys, flagged_small_cell


# ── split-folds ──────────────────────────────────────────────────────────────

def cmd_split_folds(args: argparse.Namespace) -> None:
    gold_records = load_gold_records(args.gold)
    doc_ids = sorted(gold_records)

    eras, tiers = join_era_tier(doc_ids, args.export_dir, args.gold_sample)
    strat_keys, flagged_small_cell = build_stratification_keys(eras, tiers, args.k)

    n_flagged = sum(flagged_small_cell)
    if n_flagged:
        print(
            f"eval_ner_gold.py split-folds: {n_flagged} record(s) fall in an "
            f"era x silver_tier cell averaging < 5 records/fold at k={args.k}; "
            f"flagged_small_cell=true for these in {args.output}",
            file=sys.stderr,
        )

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    fold_of = np.empty(len(doc_ids), dtype=int)
    for fold_index, (_, test_idx) in enumerate(skf.split(doc_ids, strat_keys)):
        fold_of[test_idx] = fold_index

    folds = {
        str(doc_id): {
            "fold": int(fold_of[i]),
            "era": eras[i],
            "silver_tier": tiers[i],
            "flagged_small_cell": flagged_small_cell[i],
        }
        for i, doc_id in enumerate(doc_ids)
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(folds)} fold assignments (k={args.k}) to {args.output}")


# ── score ────────────────────────────────────────────────────────────────────

def cmd_score(args: argparse.Namespace) -> None:
    gold_records = load_gold_records(args.gold)

    with open(args.folds, encoding="utf-8") as f:
        folds = json.load(f)

    fold_doc_ids = [
        int(doc_id) for doc_id, info in folds.items() if info["fold"] == args.fold
    ]
    if not fold_doc_ids:
        sys.exit(f"eval_ner_gold.py score: no gold records assigned to fold {args.fold}")

    predictions: dict[int, list[list[Any]]] = {}
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            predictions[rec["id"]] = rec["spans"]

    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    per_record: list[dict[str, Any]] = []

    for doc_id in fold_doc_ids:
        gold_rec = gold_records.get(doc_id)
        if gold_rec is None:
            sys.exit(
                f"eval_ner_gold.py score: doc id {doc_id} in {args.folds} "
                f"not found in gold file {args.gold}"
            )
        pred_spans = predictions.get(doc_id)
        if pred_spans is None:
            sys.exit(
                f"eval_ner_gold.py score: doc id {doc_id} (fold {args.fold}) "
                f"missing from predictions file {args.predictions}"
            )

        era = folds[str(doc_id)]["era"]
        gold_set = {tuple(span) for span in gold_rec["label"]}
        pred_set = {tuple(span) for span in pred_spans}

        for start, end, label in gold_set | pred_set:
            key = (label, era)
            span = (start, end, label)
            if span in gold_set and span in pred_set:
                outcome = "tp"
            elif span in pred_set:
                outcome = "fp"
            else:
                outcome = "fn"
            counts[key][outcome] += 1
            per_record.append({
                "doc_id": doc_id, "label": label, "era": era, "outcome": outcome,
            })

    results = [
        {"label": label, "era": era, "system": args.system, "fold": args.fold,
         **compute_f1(**c)}
        for (label, era), c in counts.items()
    ]

    output = {
        "system": args.system,
        "fold": args.fold,
        "n_records": len(fold_doc_ids),
        "results": results,
        "per_record": per_record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Scored {len(fold_doc_ids)} records (system={args.system}, fold={args.fold}) -> {args.output}")


# ── aggregate ────────────────────────────────────────────────────────────────

def bootstrap_ci(
    outcomes: list[str], resamples: int, ci: float, rng: np.random.Generator,
) -> tuple[float, float]:
    """95% (or `ci`) bootstrap CI on F1, resampling records with replacement."""
    n = len(outcomes)
    f1s = np.empty(resamples)
    outcomes_arr = np.array(outcomes)
    for i in range(resamples):
        sample = outcomes_arr[rng.integers(0, n, size=n)]
        tp = int(np.sum(sample == "tp"))
        fp = int(np.sum(sample == "fp"))
        fn = int(np.sum(sample == "fn"))
        f1s[i] = compute_f1(tp, fp, fn)["f1"]
    alpha = (1 - ci) / 2
    return (
        float(np.quantile(f1s, alpha)),
        float(np.quantile(f1s, 1 - alpha)),
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    eval_files = sorted(args.eval_dir.glob("eval_fold*_*.json"))
    if not eval_files:
        sys.exit(f"eval_ner_gold.py aggregate: no eval_fold*_*.json files found in {args.eval_dir}")

    # per_stratum[(system, label, era)] = {"fold_f1s": [...], "outcomes": [...], "n_folds": set()}
    per_stratum: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"fold_f1s": [], "outcomes": [], "folds_seen": set()}
    )

    for path in eval_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        system = data["system"]
        fold = data["fold"]
        for row in data["results"]:
            key = (system, row["label"], row["era"])
            per_stratum[key]["fold_f1s"].append(row["f1"])
            per_stratum[key]["folds_seen"].add(fold)
        for rec in data["per_record"]:
            key = (system, rec["label"], rec["era"])
            per_stratum[key]["outcomes"].append(rec["outcome"])

    rng = np.random.default_rng(42)
    rows = []
    for (system, label, era), agg in sorted(per_stratum.items()):
        fold_f1s = agg["fold_f1s"]
        outcomes = agg["outcomes"]
        n_gold = sum(1 for o in outcomes if o in ("tp", "fn"))
        ci_low, ci_high = bootstrap_ci(outcomes, args.bootstrap_resamples, args.ci, rng)
        rows.append({
            "system": system,
            "label": label,
            "era": era,
            "n_folds": len(agg["folds_seen"]),
            "mean_f1": round(float(np.mean(fold_f1s)), 4),
            "std_f1": round(float(np.std(fold_f1s)), 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "n_gold_spans": n_gold,
            "n_spans_total": len(outcomes),
            "flagged_small_cell": n_gold < 5,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} (system, label, era) rows to {args.output}")

    if args.iaa_summary and args.iaa_summary.exists():
        print(f"Note: --iaa-summary given ({args.iaa_summary}) but merge-in is not yet "
              f"implemented -- see plan §9 open item.", file=sys.stderr)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="A3: k-fold NER evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split-folds")
    p_split.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p_split.add_argument("--gold-sample", type=Path, default=DEFAULT_GOLD_SAMPLE)
    p_split.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    p_split.add_argument("--k", type=int, required=True)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--output", type=Path, default=None)
    p_split.set_defaults(func=cmd_split_folds)

    p_score = sub.add_parser("score")
    p_score.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p_score.add_argument("--folds", type=Path, required=True)
    p_score.add_argument("--predictions", type=Path, required=True)
    p_score.add_argument("--system", type=str, required=True)
    p_score.add_argument("--fold", type=int, required=True)
    p_score.add_argument("--output", type=Path, default=None)
    p_score.set_defaults(func=cmd_score)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p_agg.add_argument("--iaa-summary", type=Path, default=None)
    p_agg.add_argument("--bootstrap-resamples", type=int, default=1000)
    p_agg.add_argument("--ci", type=float, default=0.95)
    p_agg.add_argument("--output", type=Path, default=None)
    p_agg.set_defaults(func=cmd_aggregate)

    args = parser.parse_args()

    if args.command == "split-folds" and args.output is None:
        args.output = DEFAULT_EVAL_DIR / f"folds_k{args.k}.json"
    elif args.command == "score" and args.output is None:
        args.output = DEFAULT_EVAL_DIR / f"eval_fold{args.fold}_{args.system}.json"
    elif args.command == "aggregate" and args.output is None:
        args.output = DEFAULT_EVAL_DIR / "eval_summary.csv"

    args.func(args)


if __name__ == "__main__":
    main()
