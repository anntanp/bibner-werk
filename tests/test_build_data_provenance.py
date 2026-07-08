# Purpose:      Tests for scripts/py/build_data_provenance.py — the gold +
#               silver cross-pipeline provenance table builder.
# Usage:        pytest tests/test_build_data_provenance.py
# Dependencies: pytest, pyarrow

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from build_data_provenance import build_row_id_index, load_gold_obj_ids, load_silver_obj_ids, main

FIXTURES = Path(__file__).parent / "fixtures"


def _write_parquet(path: Path, obj_ids: list[str]) -> None:
    pq.write_table(pa.table({"obj_id": obj_ids}), path)


def _write_silver_spans(path: Path, obj_ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj_id in obj_ids:
            f.write(json.dumps({"id": obj_id, "text": "x", "label": []}) + "\n")


def test_load_gold_obj_ids_reads_export_dir_fixture() -> None:
    result = load_gold_obj_ids(FIXTURES / "export_dir")
    assert len(result) == 20
    assert result[1] == "OBJ001"
    assert result[20] == "OBJ020"


def test_load_silver_obj_ids_uses_line_index_as_doc_id(tmp_path: Path) -> None:
    silver_path = tmp_path / "silver_spans.jsonl"
    _write_silver_spans(silver_path, ["OBJ010", "OBJ011"])
    assert load_silver_obj_ids(silver_path) == {0: "OBJ010", 1: "OBJ011"}


def test_build_row_id_index_maps_obj_id_to_row_position(tmp_path: Path) -> None:
    parquet_path = tmp_path / "s2_meta.parquet"
    _write_parquet(parquet_path, ["OBJ001", "OBJ002", "OBJ003"])
    assert build_row_id_index(parquet_path) == {"OBJ001": 0, "OBJ002": 1, "OBJ003": 2}


def test_main_writes_expected_columns_and_flags(tmp_path: Path) -> None:
    parquet_path = tmp_path / "s2_meta.parquet"
    _write_parquet(parquet_path, ["OBJ001", "OBJ002", "OBJ003", "OBJ099"])
    silver_path = tmp_path / "silver_spans.jsonl"
    _write_silver_spans(silver_path, ["OBJ099"])  # silver-only obj_id, distinct from gold fixture's OBJ001-020
    output_path = tmp_path / "out.parquet"

    import sys

    argv = [
        "build_data_provenance.py",
        "--export-dir",
        str(FIXTURES / "export_dir"),
        "--silver-spans",
        str(silver_path),
        "--parquet-file",
        str(parquet_path),
        "--output",
        str(output_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv

    table = pq.read_table(output_path).to_pylist()
    by_doc = {(row["doc_id"], row["is_gold"]): row for row in table}

    assert by_doc[(1, True)]["obj_id"] == "OBJ001"
    assert by_doc[(1, True)]["parquet_row_id"] == 0
    assert by_doc[(0, False)]["obj_id"] == "OBJ099"
    assert by_doc[(0, False)]["parquet_row_id"] == 3
    assert len(table) == 21  # 20 gold (fixture) + 1 silver


def test_main_raises_on_gold_silver_overlap(tmp_path: Path) -> None:
    parquet_path = tmp_path / "s2_meta.parquet"
    _write_parquet(parquet_path, ["OBJ001", "OBJ002", "OBJ003"])
    silver_path = tmp_path / "silver_spans.jsonl"
    _write_silver_spans(silver_path, ["OBJ001"])  # OBJ001 also in gold fixture -> leakage
    output_path = tmp_path / "out.parquet"

    import sys

    argv = [
        "build_data_provenance.py",
        "--export-dir",
        str(FIXTURES / "export_dir"),
        "--silver-spans",
        str(silver_path),
        "--parquet-file",
        str(parquet_path),
        "--output",
        str(output_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(ValueError, match="leakage"):
            main()
    finally:
        sys.argv = old_argv
