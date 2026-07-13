# Purpose:      Tests for the new, GPU-free additions to
#               scripts/py/train_gliner_silver.py: the data-scaling sweep's
#               --train-size subsampling, and the unattended-run hardening
#               (preflight checks, resume-from-checkpoint). Does not backfill
#               tests for this script's pre-existing functions (oversample_tier2,
#               stratified_dev_split, leakage_guard, validate_labels) — out of
#               scope for this change; see A5-finetuning-scaling-sweep-plan.md
#               checklist §1. GPU-visibility preflight check is exercised via
#               monkeypatched torch, not real hardware.
# Usage:        pytest tests/test_train_gliner_silver.py
# Dependencies: pytest

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from train_gliner_silver import (
    check_resumable,
    find_last_checkpoint,
    run_preflight_checks,
    subsample_train_records,
    write_run_config,
)


# --- subsample_train_records ------------------------------------------------------------------------


def _make_records(n: int) -> list[dict[str, Any]]:
    eras = ["pre-1700", "1700-1800", "19th-c", "modern"]
    return [{"id": i, "era": eras[i % len(eras)], "silver_tier": 1 + (i % 2)} for i in range(n)]


def test_subsample_train_records_returns_exact_size() -> None:
    records = _make_records(100)

    subsample = subsample_train_records(records, train_size=20, seed=42)

    assert len(subsample) == 20
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
    records = [{"id": i, "era": f"era{i}", "silver_tier": i} for i in range(20)]

    subsample = subsample_train_records(records, train_size=5, seed=42)

    assert len(subsample) == 5


# --- run_preflight_checks ----------------------------------------------------------------------------


def test_run_preflight_checks_raises_on_missing_input(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError, match="not found"):
        run_preflight_checks([missing], output_dir=tmp_path / "out", require_gpu=False)


def test_run_preflight_checks_passes_with_valid_inputs_and_no_gpu_required(tmp_path: Path) -> None:
    present = tmp_path / "present.jsonl"
    present.write_text("{}")

    run_preflight_checks([present], output_dir=tmp_path / "out", require_gpu=False)  # no raise

    assert (tmp_path / "out").is_dir()


def test_run_preflight_checks_raises_on_low_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    present = tmp_path / "present.jsonl"
    present.write_text("{}")
    fake_usage = types.SimpleNamespace(total=0, used=0, free=1 * 1024**3)
    monkeypatch.setattr("shutil.disk_usage", lambda _path: fake_usage)

    with pytest.raises(OSError, match="free"):
        run_preflight_checks(
            [present], output_dir=tmp_path / "out", require_gpu=False, min_free_disk_gb=5.0
        )


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


# --- resume-from-checkpoint --------------------------------------------------------------------------


def test_find_last_checkpoint_returns_none_when_absent(tmp_path: Path) -> None:
    assert find_last_checkpoint(tmp_path / "does_not_exist") is None
    (tmp_path / "empty").mkdir()
    assert find_last_checkpoint(tmp_path / "empty") is None


def test_find_last_checkpoint_returns_highest_numbered_dir(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-12").mkdir()
    (tmp_path / "checkpoint-7").mkdir()

    assert find_last_checkpoint(tmp_path) == tmp_path / "checkpoint-12"


def test_check_resumable_returns_none_when_no_checkpoint(tmp_path: Path) -> None:
    assert check_resumable(tmp_path, config={"a": 1}) is None


def test_check_resumable_returns_checkpoint_when_config_matches(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-5").mkdir()
    config = {"fold_index": 0, "train_size": 1000}
    write_run_config(tmp_path, config)

    assert check_resumable(tmp_path, config) == tmp_path / "checkpoint-5"


def test_check_resumable_raises_when_config_mismatched(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-5").mkdir()
    write_run_config(tmp_path, {"fold_index": 0, "train_size": 1000})

    with pytest.raises(RuntimeError, match="different arguments"):
        check_resumable(tmp_path, {"fold_index": 0, "train_size": 5000})


def test_check_resumable_raises_when_sidecar_missing(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-5").mkdir()

    with pytest.raises(RuntimeError, match="no run_config.json"):
        check_resumable(tmp_path, {"fold_index": 0})
