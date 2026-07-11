# Purpose:      Tests for scripts/py/spotcheck_xlmr_predictions.py's pure-logic pieces —
#               word-alignment, comparison formatting, and dev-split sampling. No model/GPU needed;
#               predict_word_tags() (the one function that calls a real model) is untested here,
#               same "full runs aren't unit tested" posture as train_xlmr_silver.py's trainer.train().
# Usage:        pytest tests/test_spotcheck_xlmr_predictions.py
# Dependencies: pytest

from __future__ import annotations

import json
from pathlib import Path

from spotcheck_xlmr_predictions import (
    align_word_predictions,
    format_comparison,
    sample_dev_records,
)
from train_xlmr_silver import LABEL2ID


def test_align_word_predictions_labels_only_first_subword() -> None:
    # Word 0 spans two subwords (both predict B-TITLE), word 1 is one subword (I-TITLE).
    word_ids: list[int | None] = [None, 0, 0, 1, None]
    pred_ids = [
        LABEL2ID["O"],
        LABEL2ID["B-TITLE"],
        LABEL2ID["B-TITLE"],  # second subword of word 0 — must be ignored, not just repeated
        LABEL2ID["I-TITLE"],
        LABEL2ID["O"],
    ]

    tags = align_word_predictions(word_ids, pred_ids)

    assert tags == ["B-TITLE", "I-TITLE"]


def test_align_word_predictions_empty_word_ids_returns_empty() -> None:
    assert align_word_predictions([None, None], [LABEL2ID["O"], LABEL2ID["O"]]) == []


def test_format_comparison_counts_mismatches_and_flags_them() -> None:
    report, mismatches = format_comparison(
        "rec1",
        ["Der", "Titel", "."],
        ["O", "B-TITLE", "O"],
        ["O", "B-TITLE", "B-PERSON"],  # last word mismatches
    )

    assert mismatches == 1
    assert "id: rec1" in report
    assert "MISMATCH" in report
    assert report.count("MISMATCH") == 1


def test_format_comparison_perfect_match_has_zero_mismatches() -> None:
    report, mismatches = format_comparison("rec2", ["A"], ["O"], ["O"])

    assert mismatches == 0
    assert "MISMATCH" not in report


def test_sample_dev_records_reproduces_dev_split_and_respects_n_samples(tmp_path: Path) -> None:
    path = tmp_path / "bio.jsonl"
    records = [
        {"id": i, "tokens": ["x"], "tags": ["O"], "era": "19th-c" if i % 2 == 0 else "modern", "silver_tier": 1}
        for i in range(20)
    ]
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    sample = sample_dev_records(path, dev_fraction=0.5, dev_split_seed=42, n_samples=3, sample_seed=0)

    assert len(sample) == 3
    all_ids = {r["id"] for r in records}
    assert {r["id"] for r in sample} <= all_ids


def test_sample_dev_records_caps_at_dev_set_size(tmp_path: Path) -> None:
    path = tmp_path / "bio.jsonl"
    records = [{"id": i, "tokens": ["x"], "tags": ["O"], "era": "19th-c", "silver_tier": 1} for i in range(10)]
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # dev_fraction=0.1 of 10 records -> 1 dev record; requesting 8 samples should not raise
    sample = sample_dev_records(path, dev_fraction=0.1, dev_split_seed=42, n_samples=8, sample_seed=0)

    assert len(sample) <= 1
