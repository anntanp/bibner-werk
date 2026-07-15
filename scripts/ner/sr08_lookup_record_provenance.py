#!/usr/bin/env python3
"""Look up cross-pipeline identifiers for one or more gold adjudication
doc_ids, for the record-provenance table in
notes/fiz/bibner-werk/adj-precedents.md §3.

Usage:
    python3 sr08_lookup_record_provenance.py --doc-ids 6 227 \
        --export-dir data/annotation/doccano/export-20260623-1110 \
        --parquet-glob "/path/to/gemea/output/parquet/*_meta.parquet" \
        --silver-spans data/processed/silver_spans.jsonl \
        --output data/gold/sr08_record_provenance.csv

Inputs:
    --export-dir: any per-annotator Doccano export JSONL with an "id" (gold
        doc_id) and "url" (DDB item URL, whose last path segment is obj_id).
    --parquet-glob: gemea per-sector *_meta.parquet files, each with an
        "obj_id" column; sector name is taken from each file's own stem
        (`s3_meta.parquet` -> `s3`), not its parent directory, so the glob
        works whether sector files live in per-sector subdirectories or
        flat in one directory; PARQUET row id = row index within the
        sector file that first matches.
    --silver-spans: data/processed/silver_spans.jsonl, whose "id" field is
        the obj_id; silver doc id = line/row index within that file.
Outputs: data/gold/sr08_record_provenance.csv — one row per
    doc_id (gold_doc_id, obj_id, parquet_sector, parquet_row_id,
    silver_doc_id). Upserts by gold_doc_id: re-running with previously
    seen doc_ids updates those rows in place instead of duplicating;
    new doc_ids are appended.
Dependencies: pyarrow.
Assumptions: obj_id is unique within each parquet sector file and within
    silver_spans.jsonl. A doc_id not found in --export-dir, or an obj_id
    not found in any parquet sector / in silver_spans.jsonl, is recorded
    with an empty value for that field rather than failing the whole run.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import pyarrow.parquet as pq


def load_obj_id_by_doc_id(export_dir: Path, doc_ids: set) -> dict:
    result = {}
    for path in sorted(export_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] in doc_ids and rec["id"] not in result and rec.get("url"):
                    result[rec["id"]] = rec["url"].rstrip("/").rsplit("/", 1)[-1]
        if len(result) == len(doc_ids):
            break
    return result


def find_parquet_row(obj_id: str, parquet_glob: str) -> tuple:
    for path in sorted(glob.glob(parquet_glob)):
        table = pq.read_table(path, columns=["obj_id"])
        ids = table.column("obj_id").to_pylist()
        if obj_id in ids:
            return Path(path).stem.removesuffix("_meta"), ids.index(obj_id)
    return "", ""


def find_silver_row(obj_id: str, silver_spans_path: Path) -> str:
    with open(silver_spans_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if json.loads(line)["id"] == obj_id:
                return i
    return ""


def load_existing_rows(output_path: Path) -> dict:
    if not output_path.exists():
        return {}
    with open(output_path, encoding="utf-8") as f:
        return {int(row["gold_doc_id"]): row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-ids", type=int, nargs="+", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--silver-spans", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    silver_spans_path = Path(args.silver_spans)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc_ids = set(args.doc_ids)
    obj_id_by_doc = load_obj_id_by_doc_id(export_dir, doc_ids)

    rows = load_existing_rows(output_path)
    for doc_id in args.doc_ids:
        obj_id = obj_id_by_doc.get(doc_id, "")
        parquet_sector, parquet_row_id = find_parquet_row(obj_id, args.parquet_glob) if obj_id else ("", "")
        silver_doc_id = find_silver_row(obj_id, silver_spans_path) if obj_id else ""
        rows[doc_id] = {
            "gold_doc_id": doc_id,
            "obj_id": obj_id,
            "parquet_sector": parquet_sector,
            "parquet_row_id": parquet_row_id,
            "silver_doc_id": silver_doc_id,
        }

    fields = ["gold_doc_id", "obj_id", "parquet_sector", "parquet_row_id", "silver_doc_id"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for doc_id in sorted(rows):
            writer.writerow(rows[doc_id])

    print(f"{len(args.doc_ids)} doc_id(s) looked up, {len(rows)} total row(s) -> {output_path}")


if __name__ == "__main__":
    main()
