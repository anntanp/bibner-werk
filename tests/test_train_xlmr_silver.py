# Purpose:      Tests for scripts/py/train_xlmr_silver.py — the GPU-free parts
#               of A5's XLM-R fine-tuning script: leakage guard, BIO-tag
#               validation, dev split, per-tag class weights, word->subword
#               label alignment, the data-scaling sweep's --train-size
#               subsampling, and the unattended-run hardening (preflight
#               checks, resume-from-checkpoint). No torch/transformers/GPU
#               needed for most of this — the alignment test uses a minimal
#               fake fast-tokenizer stand-in rather than downloading a real
#               XLM-R checkpoint, per A5-finetuning-script-plan.md §7 ("Unit
#               tests, no GPU needed"); the GPU-visibility preflight check is
#               exercised via monkeypatched torch, not real hardware, per
#               A5-finetuning-scaling-sweep-plan.md checklist §1.
# Usage:        pytest tests/test_train_xlmr_silver.py
# Dependencies: pytest, pyarrow

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from train_xlmr_silver import (
    BIO_TAGS,
    LABEL2ID,
    check_resumable,
    compute_class_weights,
    find_last_checkpoint,
    leakage_guard,
    read_jsonl,
    run_preflight_checks,
    stratified_dev_split,
    subsample_train_records,
    tokenize_and_align_labels,
    validate_bio_tags,
    write_run_config,
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


# --- subsample_train_records (A5-finetuning-scaling-sweep-plan.md) --------------------------------


def _make_records(n: int) -> list[dict[str, Any]]:
    eras = ["pre-1700", "1700-1800", "19th-c", "modern"]
    return [{"id": i, "era": eras[i % len(eras)], "silver_tier": 1 + (i % 2)} for i in range(n)]


def test_subsample_train_records_returns_exact_size() -> None:
    records = _make_records(100)

    subsample = subsample_train_records(records, train_size=20, seed=42)

    assert len(subsample) == 20
    # every sampled record must actually come from the input pool (no fabrication)
    assert {r["id"] for r in subsample} <= {r["id"] for r in records}


def test_subsample_train_records_same_seed_is_deterministic() -> None:
    records = _make_records(100)

    first = subsample_train_records(records, train_size=20, seed=7)
    second = subsample_train_records(records, train_size=20, seed=7)

    assert {r["id"] for r in first} == {r["id"] for r in second}


def test_subsample_train_records_keeps_all_when_size_exceeds_pool() -> None:
    records = _make_records(10)

    subsample = subsample_train_records(records, train_size=50, seed=42)

    assert len(subsample) == 10


def test_subsample_train_records_falls_back_when_strata_too_small() -> None:
    # Every record has a unique era x silver_tier stratum -> sklearn's stratified
    # split can't run (needs >= 2 members per class); must fall back, not raise.
    records = [{"id": i, "era": f"era{i}", "silver_tier": i} for i in range(20)]

    subsample = subsample_train_records(records, train_size=5, seed=42)

    assert len(subsample) == 5


# --- run_preflight_checks (unattended-run hardening) -----------------------------------------------


def test_run_preflight_checks_raises_on_missing_input(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError, match="not found"):
        run_preflight_checks([missing], output_dir=tmp_path / "out", require_gpu=False)


def test_run_preflight_checks_passes_with_valid_inputs_and_no_gpu_required(tmp_path: Path) -> None:
    present = tmp_path / "present.jsonl"
    present.write_text("{}")

    run_preflight_checks([present], output_dir=tmp_path / "out", require_gpu=False)  # no raise

    assert (tmp_path / "out").is_dir()  # output dir created as a side effect


def test_run_preflight_checks_raises_on_low_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    present = tmp_path / "present.jsonl"
    present.write_text("{}")
    fake_usage = types.SimpleNamespace(total=0, used=0, free=1 * 1024**3)  # 1GB free
    monkeypatch.setattr("shutil.disk_usage", lambda _path: fake_usage)

    with pytest.raises(OSError, match="free"):
        run_preflight_checks(
            [present], output_dir=tmp_path / "out", require_gpu=False, min_free_disk_gb=5.0
        )


def test_run_preflight_checks_raises_when_no_cuda_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = tmp_path / "present.jsonl"
    present.write_text("{}")
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="no CUDA device is visible"):
        run_preflight_checks([present], output_dir=tmp_path / "out", require_gpu=True)


def test_run_preflight_checks_raises_when_more_than_one_gpu_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = tmp_path / "present.jsonl"
    present.write_text("{}")
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="2 CUDA devices visible"):
        run_preflight_checks([present], output_dir=tmp_path / "out", require_gpu=True)


def test_run_preflight_checks_passes_with_exactly_one_gpu_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = tmp_path / "present.jsonl"
    present.write_text("{}")
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    run_preflight_checks([present], output_dir=tmp_path / "out", require_gpu=True)  # no raise


# --- resume-from-checkpoint (unattended-run hardening) ---------------------------------------------


def test_find_last_checkpoint_returns_none_when_absent(tmp_path: Path) -> None:
    assert find_last_checkpoint(tmp_path / "does_not_exist") is None
    (tmp_path / "empty").mkdir()
    assert find_last_checkpoint(tmp_path / "empty") is None


def test_find_last_checkpoint_returns_highest_numbered_dir(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-12").mkdir()
    (tmp_path / "checkpoint-7").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()

    result = find_last_checkpoint(tmp_path)

    assert result == tmp_path / "checkpoint-12"


def test_check_resumable_returns_none_when_no_checkpoint(tmp_path: Path) -> None:
    assert check_resumable(tmp_path, config={"a": 1}) is None


def test_check_resumable_returns_checkpoint_when_config_matches(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-5").mkdir()
    config = {"fold_index": 0, "train_size": 1000}
    write_run_config(tmp_path, config)

    result = check_resumable(tmp_path, config)

    assert result == tmp_path / "checkpoint-5"


def test_check_resumable_raises_when_config_mismatched(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-5").mkdir()
    write_run_config(tmp_path, {"fold_index": 0, "train_size": 1000})

    with pytest.raises(RuntimeError, match="different arguments"):
        check_resumable(tmp_path, {"fold_index": 0, "train_size": 5000})


def test_check_resumable_raises_when_sidecar_missing(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-5").mkdir()  # no run_config.json written

    with pytest.raises(RuntimeError, match="no run_config.json"):
        check_resumable(tmp_path, {"fold_index": 0})


def test_write_run_config_roundtrips_via_check_resumable(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1").mkdir()
    config = {"fold_index": 3, "learning_rate": 2e-5, "train_size": None}

    write_run_config(tmp_path, config)
    result = check_resumable(tmp_path, config)

    assert result == tmp_path / "checkpoint-1"
