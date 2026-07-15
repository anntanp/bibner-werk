#!/usr/bin/env python3
"""Build a cross-pipeline provenance table for every title in the gold and
silver NER datasets, joining each record to its row in the s2 DDB parquet
metadata file.

Usage:
    python3 build_data_provenance.py \
        --export-dir data/annotation/doccano/export-20260623-1110 \
        --silver-spans data/processed/silver_spans.jsonl \
        --parquet-file /path/to/gemea/output/parquet/s2_meta.parquet \
        --output data/provenance/bibner-data-provenance.parquet

Inputs:
    --export-dir: directory of per-annotator Doccano export JSONL files,
        each record with an "id" (gold doc_id, small int) and "url" (DDB
        item URL, whose last path segment is obj_id). All annotator files
        share the same doc_id -> obj_id mapping; only one is needed per
        doc_id, so files are read in sorted order and the first match wins.
    --silver-spans: data/processed/silver_spans.jsonl, whose "id" field is
        the obj_id itself; silver doc_id = 0-based line index in this file.
    --parquet-file: the s2 DDB parquet metadata file (all gold and silver
        records live in sector s2, so no sector column or per-sector glob
        is needed).
Outputs: a single Parquet file with one row per gold or silver record and
    columns: doc_id (int; gold's Doccano id, or silver's line index -- two
    independent counters disambiguated only by is_gold), is_gold (bool),
    obj_id (str), parquet_row_id (int, or null if obj_id not found in
    --parquet-file).
Dependencies: pyarrow.
Assumptions: gold and silver are disjoint by construction (leakage
    avoidance) -- the script asserts no obj_id appears in both and raises
    if the assumption is violated. obj_id is unique within the parquet
    file and within silver_spans.jsonl.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_gold_obj_ids(export_dir: Path) -> dict[int, str]:
    """Map each gold Doccano doc_id to its obj_id, taking the first match
    across annotator files (they share the same doc_id -> obj_id mapping)."""
    result: dict[int, str] = {}
    for path in sorted(export_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                doc_id = rec["id"]
                if doc_id not in result and rec.get("url"):
                    result[doc_id] = rec["url"].rstrip("/").rsplit("/", 1)[-1]
    return result


def load_silver_obj_ids(silver_spans_path: Path) -> dict[int, str]:
    """Map each silver line index to its obj_id (silver's own "id" field)."""
    result: dict[int, str] = {}
    with open(silver_spans_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            result[i] = json.loads(line)["id"]
    return result


def build_row_id_index(parquet_file: Path) -> dict[str, int]:
    """Single-pass hash-join index from obj_id to row index in parquet_file."""
    table = pq.read_table(parquet_file, columns=["obj_id"])
    obj_ids = table.column("obj_id").to_pylist()
    return {obj_id: row_id for row_id, obj_id in enumerate(obj_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--silver-spans", required=True, type=Path)
    parser.add_argument("--parquet-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gold_obj_id_by_doc = load_gold_obj_ids(args.export_dir)
    silver_obj_id_by_doc = load_silver_obj_ids(args.silver_spans)
    logger.info("Loaded %d gold and %d silver records", len(gold_obj_id_by_doc), len(silver_obj_id_by_doc))

    overlap = set(gold_obj_id_by_doc.values()) & set(silver_obj_id_by_doc.values())
    if overlap:
        raise ValueError(
            f"{len(overlap)} obj_id(s) appear in both gold and silver, violating leakage "
            f"avoidance: {sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}"
        )

    row_id_by_obj_id = build_row_id_index(args.parquet_file)
    logger.info("Indexed %d rows from %s", len(row_id_by_obj_id), args.parquet_file)

    doc_ids: list[int] = []
    is_gold_flags: list[bool] = []
    obj_ids: list[str] = []
    parquet_row_ids: list[int | None] = []

    missing = 0
    for source_is_gold, obj_id_by_doc in ((True, gold_obj_id_by_doc), (False, silver_obj_id_by_doc)):
        for doc_id, obj_id in obj_id_by_doc.items():
            row_id = row_id_by_obj_id.get(obj_id)
            if row_id is None:
                missing += 1
            doc_ids.append(doc_id)
            is_gold_flags.append(source_is_gold)
            obj_ids.append(obj_id)
            parquet_row_ids.append(row_id)

    if missing:
        logger.warning("%d record(s) had no matching obj_id in %s", missing, args.parquet_file)

    table = pa.table(
        {
            "doc_id": doc_ids,
            "is_gold": is_gold_flags,
            "obj_id": obj_ids,
            "parquet_row_id": parquet_row_ids,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output)
    logger.info("Wrote %d rows -> %s", table.num_rows, args.output)


if __name__ == "__main__":
    main()
