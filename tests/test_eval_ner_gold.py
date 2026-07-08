# Purpose:      Tests for scripts/py/eval_ner_gold.py (A3), run against the
#               synthetic fixtures in tests/fixtures/ (gold_reconciled_fixture
#               .jsonl, gold_sample_fixture.csv, export_dir/) since the real
#               gold_reconciled.jsonl doesn't exist yet (A7 adjudication still
#               in progress) -- see notes/fiz/bibner-werk/A3-eval-ner-gold-
#               script-plan.md §7.
# Usage:        pytest tests/test_eval_ner_gold.py
# Dependencies: pytest, numpy
import argparse
import json
from pathlib import Path

import pytest

from eval_ner_gold import (
    build_stratification_keys,
    cmd_aggregate,
    cmd_score,
    cmd_split_folds,
    compute_f1,
    join_era_tier,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLD_FIXTURE = FIXTURES / "gold_reconciled_fixture.jsonl"
GOLD_SAMPLE_FIXTURE = FIXTURES / "gold_sample_fixture.csv"
EXPORT_DIR_FIXTURE = FIXTURES / "export_dir"


# ── compute_f1 ───────────────────────────────────────────────────────────────

def test_compute_f1_known_values():
    result = compute_f1(tp=3, fp=1, fn=2)
    assert result["precision"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(0.6)
    assert result["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6), abs=1e-4)


def test_compute_f1_no_predictions_no_gold():
    result = compute_f1(tp=0, fp=0, fn=0)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


# ── build_stratification_keys ───────────────────────────────────────────────

def test_build_stratification_keys_flags_small_cells():
    # 2 records of ("eraA", "0"), 1 record of ("eraA", "2") -- both under 5*k
    # at k=1, so neither is flagged; raise k so the 5-per-fold bar bites.
    eras = ["eraA", "eraA", "eraA"]
    tiers = ["0", "0", "2"]
    keys, flagged = build_stratification_keys(eras, tiers, k=1)
    # k=1: 5*k=5, all three cells (count 2 and count 1) are below 5 -> flagged
    assert flagged == [True, True, True]


def test_build_stratification_keys_collapses_for_sklearn_minimum():
    # Single ("eraA", "2") record: count=1 < k=2, must collapse to survive
    # StratifiedKFold's >= k-per-class requirement.
    eras = ["eraA", "eraA", "eraA"]
    tiers = ["0", "0", "2"]
    keys, _ = build_stratification_keys(eras, tiers, k=2)
    # the lone tier "2" record collapses to era "eraA" alone (count still < k=2
    # after collapsing to era-only, since eraA__0 pair already has count=2>=k)
    assert keys[2] in ("eraA", "__collapsed_other__")


# ── join_era_tier ────────────────────────────────────────────────────────────

def test_join_era_tier_resolves_all_fixture_records():
    doc_ids = list(range(1, 21))
    eras, tiers = join_era_tier(doc_ids, EXPORT_DIR_FIXTURE, GOLD_SAMPLE_FIXTURE)
    assert len(eras) == len(tiers) == 20
    assert set(eras) == {"pre-1700", "1700-1800", "19th-c", "modern"}
    assert set(tiers) == {"0", "1", "2"}


def test_join_era_tier_fails_loudly_on_unknown_doc_id():
    with pytest.raises(SystemExit):
        join_era_tier([9999], EXPORT_DIR_FIXTURE, GOLD_SAMPLE_FIXTURE)


# ── split-folds (integration, via cmd_split_folds) ──────────────────────────

def _split_folds_args(tmp_path: Path, k: int = 5, seed: int = 42) -> argparse.Namespace:
    return argparse.Namespace(
        gold=GOLD_FIXTURE,
        gold_sample=GOLD_SAMPLE_FIXTURE,
        export_dir=EXPORT_DIR_FIXTURE,
        k=k,
        seed=seed,
        output=tmp_path / f"folds_k{k}.json",
    )


def test_split_folds_covers_all_records_with_balanced_sizes(tmp_path):
    args = _split_folds_args(tmp_path, k=5)
    cmd_split_folds(args)

    folds = json.loads(args.output.read_text())
    assert len(folds) == 20
    fold_sizes = {}
    for info in folds.values():
        fold_sizes[info["fold"]] = fold_sizes.get(info["fold"], 0) + 1
    assert set(fold_sizes) == {0, 1, 2, 3, 4}
    assert all(size == 4 for size in fold_sizes.values())


def test_split_folds_reproducible_with_same_seed(tmp_path):
    args1 = _split_folds_args(tmp_path, k=5, seed=7)
    args1.output = tmp_path / "folds_a.json"
    cmd_split_folds(args1)

    args2 = _split_folds_args(tmp_path, k=5, seed=7)
    args2.output = tmp_path / "folds_b.json"
    cmd_split_folds(args2)

    assert json.loads(args1.output.read_text()) == json.loads(args2.output.read_text())


def test_split_folds_preserves_era_stratification(tmp_path):
    args = _split_folds_args(tmp_path, k=5)
    cmd_split_folds(args)
    folds = json.loads(args.output.read_text())

    # each era has 5 records (see fixture generation); with k=5, stratified
    # folds should place ~1 record of each era per fold.
    from collections import defaultdict
    era_per_fold = defaultdict(lambda: defaultdict(int))
    for info in folds.values():
        era_per_fold[info["era"]][info["fold"]] += 1
    for era, per_fold in era_per_fold.items():
        assert set(per_fold.values()) == {1}, f"{era} not evenly split: {per_fold}"


# ── score ────────────────────────────────────────────────────────────────────

def test_score_exact_span_match_counts(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps({"id": 1, "text": "Foo Bar", "label": [[0, 3, "TITLE"], [4, 7, "PERSON"]]}) + "\n"
        + json.dumps({"id": 2, "text": "Baz", "label": [[0, 3, "TITLE"]]}) + "\n"
    )
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(json.dumps({
        "1": {"fold": 0, "era": "modern", "silver_tier": "0", "flagged_small_cell": False},
        "2": {"fold": 0, "era": "modern", "silver_tier": "0", "flagged_small_cell": False},
    }))
    predictions_path = tmp_path / "preds.jsonl"
    predictions_path.write_text(
        # doc 1: TITLE correct (tp), PERSON missed (fn)
        json.dumps({"id": 1, "spans": [[0, 3, "TITLE"]]}) + "\n"
        # doc 2: TITLE correct (tp) plus a spurious TITLE (fp)
        + json.dumps({"id": 2, "spans": [[0, 3, "TITLE"], [10, 15, "TITLE"]]}) + "\n"
    )
    output_path = tmp_path / "eval_fold0_testsys.json"

    args = argparse.Namespace(
        gold=gold_path, folds=folds_path, predictions=predictions_path,
        system="testsys", fold=0, output=output_path,
    )
    cmd_score(args)

    data = json.loads(output_path.read_text())
    by_label = {row["label"]: row for row in data["results"]}
    assert by_label["TITLE"] == {
        "label": "TITLE", "era": "modern", "system": "testsys", "fold": 0,
        "tp": 2, "fp": 1, "fn": 0,
        "precision": round(2 / 3, 4), "recall": 1.0, "f1": round(2 * (2 / 3) / (2 / 3 + 1), 4),
    }
    assert by_label["PERSON"]["tp"] == 0
    assert by_label["PERSON"]["fn"] == 1


def test_score_offset_mismatch_is_fp_plus_fn_not_partial_match(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps({"id": 1, "text": "Foo Bar Baz", "label": [[0, 7, "TITLE"]]}) + "\n")
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(json.dumps({
        "1": {"fold": 0, "era": "modern", "silver_tier": "0", "flagged_small_cell": False},
    }))
    predictions_path = tmp_path / "preds.jsonl"
    # off-by-one on the end boundary: predicted [0, 8] vs gold [0, 7]
    predictions_path.write_text(json.dumps({"id": 1, "spans": [[0, 8, "TITLE"]]}) + "\n")
    output_path = tmp_path / "eval_fold0_testsys.json"

    args = argparse.Namespace(
        gold=gold_path, folds=folds_path, predictions=predictions_path,
        system="testsys", fold=0, output=output_path,
    )
    cmd_score(args)

    data = json.loads(output_path.read_text())
    row = data["results"][0]
    assert row["tp"] == 0
    assert row["fp"] == 1
    assert row["fn"] == 1


# ── aggregate ────────────────────────────────────────────────────────────────

def test_aggregate_flags_small_cell_and_computes_ci(tmp_path):
    eval_dir = tmp_path
    # Two folds, one system, one (label, era) stratum with 3 gold spans total
    # (< 5) so flagged_small_cell should trigger.
    for fold, (tp, fp, fn) in enumerate([(2, 0, 0), (1, 0, 0)]):
        per_record = [{"doc_id": fold * 10 + i, "label": "TITLE", "era": "modern", "outcome": "tp"}
                      for i in range(tp)]
        f1 = compute_f1(tp, fp, fn)["f1"]
        (eval_dir / f"eval_fold{fold}_sysA.json").write_text(json.dumps({
            "system": "sysA", "fold": fold,
            "results": [{"label": "TITLE", "era": "modern", "system": "sysA", "fold": fold,
                         "tp": tp, "fp": fp, "fn": fn, "precision": 1.0, "recall": 1.0, "f1": f1}],
            "per_record": per_record,
        }))

    output_path = tmp_path / "eval_summary.csv"
    args = argparse.Namespace(
        eval_dir=eval_dir, iaa_summary=None, bootstrap_resamples=200, ci=0.95, output=output_path,
    )
    cmd_aggregate(args)

    import csv
    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["system"] == "sysA"
    assert row["label"] == "TITLE"
    assert row["flagged_small_cell"] == "True"
    assert 0.0 <= float(row["ci_low"]) <= float(row["mean_f1"]) <= float(row["ci_high"]) <= 1.0
