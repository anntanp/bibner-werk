# Purpose:      Tests for scripts/py/train_xlmr_silver.py — the GPU-free parts
#               of A5's XLM-R fine-tuning script: leakage guard, BIO-tag
#               validation, dev split, per-tag class weights, and word->subword
#               label alignment. No torch/transformers/GPU needed — the
#               alignment test uses a minimal fake fast-tokenizer stand-in
#               rather than downloading a real XLM-R checkpoint, per
#               A5-finetuning-script-plan.md §7 ("Unit tests, no GPU needed").
# Usage:        pytest tests/test_train_xlmr_silver.py
# Dependencies: pytest, pyarrow

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from train_xlmr_silver import (
    BIO_TAGS,
    LABEL2ID,
    compute_class_weights,
    leakage_guard,
    read_jsonl,
    stratified_dev_split,
    tokenize_and_align_labels,
    validate_bio_tags,
)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_provenance(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in ("doc_id", "is_gold", "obj_id")}), path)


def test_validate_bio_tags_accepts_known_vocabulary() -> None:
    validate_bio_tags([{"tags": ["O", "B-TITLE", "I-TITLE"]}, {"tags": ["B-PERSON"]}])  # no raise


def test_validate_bio_tags_rejects_unknown_tag() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_bio_tags([{"tags": ["O", "B-BOGUS"]}])


def test_leakage_guard_passes_when_disjoint(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, [{"id": 1}, {"id": 2}])
    provenance_path = tmp_path / "provenance.parquet"
    _write_provenance(
        provenance_path,
        [
            {"doc_id": 1, "is_gold": True, "obj_id": "OBJ001"},
            {"doc_id": 2, "is_gold": True, "obj_id": "OBJ002"},
            {"doc_id": 0, "is_gold": False, "obj_id": "OBJ099"},
        ],
    )
    bio_records = [{"id": "OBJ099"}]  # silver-only obj_id, disjoint from gold's OBJ001/OBJ002

    leakage_guard(bio_records, gold_path, provenance_path)  # no raise


def test_leakage_guard_raises_on_overlap(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, [{"id": 1}])
    provenance_path = tmp_path / "provenance.parquet"
    _write_provenance(
        provenance_path,
        [
            {"doc_id": 1, "is_gold": True, "obj_id": "OBJ001"},
            {"doc_id": 0, "is_gold": False, "obj_id": "OBJ001"},
        ],
    )
    bio_records = [{"id": "OBJ001"}]  # same obj_id as the gold row -> leakage

    with pytest.raises(AssertionError, match="Leakage guard failed"):
        leakage_guard(bio_records, gold_path, provenance_path)


def test_stratified_dev_split_covers_all_records_and_respects_fraction() -> None:
    records = [
        {"id": i, "era": era, "silver_tier": tier}
        for i in range(40)
        for era, tier in [("19th-c", 1)]
    ][:40]
    # give enough per-stratum members for sklearn's stratified split to succeed
    for i, rec in enumerate(records):
        rec["era"] = "19th-c" if i % 2 == 0 else "modern"

    train, dev = stratified_dev_split(records, dev_fraction=0.25, seed=42)

    assert len(train) + len(dev) == 40
    assert len(dev) == 10
    assert {r["id"] for r in train} | {r["id"] for r in dev} == {r["id"] for r in records}
    assert {r["id"] for r in train} & {r["id"] for r in dev} == set()


def test_stratified_dev_split_falls_back_when_strata_too_small() -> None:
    # Every record has a unique era x silver_tier stratum -> sklearn's stratified
    # split can't run (needs >= 2 members per class); must fall back, not raise.
    records = [{"id": i, "era": f"era{i}", "silver_tier": i} for i in range(10)]

    train, dev = stratified_dev_split(records, dev_fraction=0.2, seed=42)

    assert len(train) + len(dev) == 10


def test_compute_class_weights_upweights_rare_tag() -> None:
    train_records = [
        {"tags": ["O"] * 100},
        {"tags": ["B-PERSON"]},  # PERSON appears once, O appears 100x
    ]

    weights = compute_class_weights(train_records)

    o_weight = weights[BIO_TAGS.index("O")]
    person_weight = weights[BIO_TAGS.index("B-PERSON")]
    assert person_weight > o_weight
    # normalized to mean 1.0 across all 7 tags
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-9


def test_compute_class_weights_handles_absent_tag_without_div_by_zero() -> None:
    train_records = [{"tags": ["O", "B-TITLE"]}]  # PERSON never appears in this split

    weights = compute_class_weights(train_records)  # must not raise ZeroDivisionError

    assert len(weights) == len(BIO_TAGS)


class _FakeEncoding:
    """Minimal stand-in for a HF BatchEncoding: dict-style access + word_ids()."""

    def __init__(self, input_ids: list[int], attention_mask: list[int], word_ids: list[int | None]) -> None:
        self._data = {"input_ids": input_ids, "attention_mask": attention_mask}
        self._word_ids = word_ids

    def __getitem__(self, key: str) -> list[int]:
        return self._data[key]

    def word_ids(self) -> list[int | None]:
        return self._word_ids


class _FakeTokenizer:
    """Minimal stand-in for a HF fast tokenizer, exercising tokenize_and_align_labels()'s
    alignment logic without downloading a real XLM-R checkpoint (no network needed).
    Splits each word into 2 fake subwords if longer than 6 chars, else 1 — enough to
    verify the "label only the first subword" rule against a real multi-subword word."""

    def __call__(
        self, words: list[str], is_split_into_words: bool = True, truncation: bool = True, max_length: int = 256
    ) -> _FakeEncoding:
        input_ids = [0]  # CLS
        word_ids: list[int | None] = [None]
        for i, word in enumerate(words[:max_length]):
            n_subwords = 2 if len(word) > 6 else 1
            input_ids.extend([100 + i] * n_subwords)
            word_ids.extend([i] * n_subwords)
        input_ids.append(2)  # SEP
        word_ids.append(None)
        return _FakeEncoding(input_ids, [1] * len(input_ids), word_ids)


def test_tokenize_and_align_labels_labels_only_first_subword() -> None:
    record = {"tokens": ["Kurzertitel", "ab"], "tags": ["B-TITLE", "I-TITLE"]}

    result = tokenize_and_align_labels(record, _FakeTokenizer(), max_length=256)

    # "Kurzertitel" (11 chars) -> 2 fake subwords, "ab" -> 1; CLS/SEP get -100.
    assert result["labels"] == [-100, LABEL2ID["B-TITLE"], -100, LABEL2ID["I-TITLE"], -100]
    assert len(result["input_ids"]) == len(result["labels"]) == len(result["attention_mask"])


def test_read_jsonl_streams_records_in_order(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    _write_jsonl(path, [{"id": "a"}, {"id": "b"}])

    assert [r["id"] for r in read_jsonl(path)] == ["a", "b"]
