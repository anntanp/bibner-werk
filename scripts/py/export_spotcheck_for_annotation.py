#!/usr/bin/env python3
# Purpose:      One-off export of the 16 records reviewed in A5-finetuning-xlmr-spotcheck.md into a
#               Doccano-importable JSONL, unlabeled, so the PI can annotate them independently (fresh
#               eyes, not anchored by the silver labels or model predictions already shown in that
#               note) — an informal second opinion on whether the silver tags used for the spot-check
#               are actually right, not a formal IAA round. Enriched with htype + DDB URL meta, same
#               nested "meta" schema as scripts/ner/sr08_add_meta_to_export.py's enrich mode.
# Usage:        python3 scripts/py/export_spotcheck_for_annotation.py
#                   [--bio-data PATH] [--silver-spans PATH] [--sample-seeds INT [INT ...]]
#                   [--n-samples INT] [--dev-fraction FLOAT] [--dev-split-seed INT]
#                   [--ddb-parquet PATH] [--htype-csv PATH] [--output PATH]
# Inputs:       data/silver/bio_dataset_silver.jsonl (to reproduce the exact dev split
#                   train_xlmr_silver.py used)
#               data/silver/silver_spans.jsonl (char-offset text/spans, for the original
#                   un-tokenized sentence)
#               /path/to/gemea/data/out/s2/s2_meta.parquet (DDB metadata, for dc_type ->
#                   htype derivation; lives in the separate gemea project, not bibner-werk)
#               /path/to/gemea/data/schema/ddbedm-htype.csv (dc_type -> htype code lookup)
# Outputs:      data/annotation/doccano/spotcheck_xlmr_fold0_import.jsonl (Doccano import schema:
#                   {"id", "text", "meta": {"htype", "url"}, "label": [], "Comments": []} — label
#                   deliberately empty, not pre-filled with silver or predicted spans)
# Dependencies: pyarrow (DDB parquet lookup)
# Assumptions:  --sample-seeds defaults to (0, 1), reproducing the exact two samples already
#                   reviewed in A5-finetuning-xlmr-spotcheck.md — pass different seeds to export a
#                   different/additional set instead. Doccano's own "id" field is a sequential int
#                   (1..N), not the DDB obj_id, matching the existing export schema's convention.
#                   htype derivation logic (split on "|", take last segment, "Monograph" ->
#                   MONOGRAPH_FALLBACK, else label_de lookup) mirrors
#                   scripts/ner/sr08_add_meta_to_export.py's derive_htype() exactly — that script's
#                   own HTYPE path (BASE/data/schema/ddbedm-htype.csv, expected inside bibner-werk)
#                   doesn't actually exist in this repo; the file only lives in gemea, so this
#                   script's --htype-csv default points there directly rather than assuming a local
#                   copy that isn't present. Missing dc_type (obj_id not found, or empty string) ->
#                   htype is None in the output rather than failing the whole export; DDB parquet
#                   lookup is best-effort — if --ddb-parquet doesn't resolve (e.g. running somewhere
#                   without access to the sibling gemea checkout), the export still succeeds with
#                   htype left as None for every record rather than crashing on missing metadata.
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BIO_DATA = ROOT / "data" / "silver" / "bio_dataset_silver.jsonl"
DEFAULT_SILVER_SPANS = ROOT / "data" / "silver" / "silver_spans.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "annotation" / "doccano" / "spotcheck_xlmr_fold0_import.jsonl"
DEFAULT_DDB_PARQUET = ROOT.parent / "gemea" / "data" / "out" / "s2" / "s2_meta.parquet"
DEFAULT_HTYPE_CSV = ROOT.parent / "gemea" / "data" / "schema" / "ddbedm-htype.csv"
DDB_URL_TEMPLATE = "https://www.deutsche-digitale-bibliothek.de/item/{obj_id}"
MONOGRAPH_FALLBACK = "htype_021 (Monografie)"

sys.path.insert(0, str(ROOT / "scripts" / "py"))
from train_xlmr_silver import read_jsonl, stratified_dev_split  # noqa: E402


def collect_sample_ids(
    bio_data: Path, dev_fraction: float, dev_split_seed: int, n_samples: int, sample_seeds: list[int]
) -> list[str]:
    """Reproduce train_xlmr_silver.py's dev split, then the same --sample-seed draws used for review."""
    records = read_jsonl(bio_data)
    _, dev_records = stratified_dev_split(records, dev_fraction, dev_split_seed)

    ids: list[str] = []
    for seed in sample_seeds:
        rng = random.Random(seed)
        sample = rng.sample(dev_records, min(n_samples, len(dev_records)))
        ids.extend(str(rec["id"]) for rec in sample)
    return ids


def lookup_dc_types(obj_ids: list[str], ddb_parquet: Path) -> dict[str, str | None]:
    """Look up each obj_id's dc_type directly from the DDB metadata parquet.

    Best-effort: returns an all-None mapping (logged, not raised) if ddb_parquet doesn't exist,
    so the rest of the export can still proceed without gemea's data available.
    """
    if not ddb_parquet.exists():
        logging.warning("DDB parquet not found at %s — htype will be left blank for all records.", ddb_parquet)
        return dict.fromkeys(obj_ids)

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.compute as pc  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    table = pq.read_table(ddb_parquet, columns=["obj_id", "dc_type"])
    mask = pc.is_in(table["obj_id"], value_set=pa.array(obj_ids))
    found = {row["obj_id"]: row["dc_type"] for row in table.filter(mask).to_pylist()}
    return {obj_id: found.get(obj_id) for obj_id in obj_ids}


def build_htype_lookup(htype_csv: Path) -> dict[str, str]:
    """Load the dc_type (German label) -> 'htype_code (label_de)' lookup table.

    Same source table and output format as sr08_add_meta_to_export.py's build_htype_lookup().
    """
    with open(htype_csv, encoding="utf-8") as f:
        return {row["label_de"]: f"{row['htype_code']} ({row['label_de']})" for row in csv.DictReader(f)}


def derive_htype(dc_type: str | None, lookup: dict[str, str]) -> str | None:
    """Derive an htype label from a raw dc_type string.

    Mirrors sr08_add_meta_to_export.py's derive_htype() exactly: split on "|" and take the last
    segment (this project's dc_type values are single German labels with no "|", so this is a
    no-op here but kept for consistency with the upstream logic), special-case the English
    "Monograph" to the same fallback constant, else exact-match lookup by German label. Returns
    None (not an error) for unrecognized or missing values — matches upstream behavior.
    """
    if not isinstance(dc_type, str) or not dc_type:
        return None
    last = dc_type.split("|")[-1].strip()
    if last == "Monograph":
        return MONOGRAPH_FALLBACK
    return lookup.get(last)


def build_import_records(
    sample_ids: list[str],
    silver_spans_path: Path,
    dc_types: dict[str, str | None],
    htype_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    """Look up original text + derived htype for each sampled obj_id; format as unlabeled Doccano import records."""
    silver_by_id = {rec["id"]: rec for rec in read_jsonl(silver_spans_path)}

    records = []
    for doccano_id, obj_id in enumerate(sample_ids, start=1):
        source = silver_by_id.get(obj_id)
        if source is None:
            raise KeyError(f"obj_id {obj_id} not found in {silver_spans_path} — data may be out of sync")
        records.append({
            "id": doccano_id,
            "text": source["text"],
            "meta": {
                "htype": derive_htype(dc_types.get(obj_id), htype_lookup),
                "url": DDB_URL_TEMPLATE.format(obj_id=obj_id),
            },
            "label": [],
            "Comments": [],
        })
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Export A5-finetuning-xlmr-spotcheck.md's reviewed records as an unlabeled "
        "Doccano import (with htype + DDB URL meta) for independent PI annotation."
    )
    parser.add_argument("--bio-data", type=Path, default=DEFAULT_BIO_DATA)
    parser.add_argument("--silver-spans", type=Path, default=DEFAULT_SILVER_SPANS)
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=[0, 1],
                        help="Default: the two seeds already used in A5-finetuning-xlmr-spotcheck.md")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--dev-split-seed", type=int, default=42)
    parser.add_argument("--ddb-parquet", type=Path, default=DEFAULT_DDB_PARQUET,
                        help="DDB metadata parquet, lives in the sibling gemea repo (default: %(default)s)")
    parser.add_argument("--htype-csv", type=Path, default=DEFAULT_HTYPE_CSV,
                        help="dc_type -> htype lookup CSV, lives in the sibling gemea repo (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sample_ids = collect_sample_ids(
        args.bio_data, args.dev_fraction, args.dev_split_seed, args.n_samples, args.sample_seeds
    )
    dc_types = lookup_dc_types(sample_ids, args.ddb_parquet)
    htype_lookup = build_htype_lookup(args.htype_csv) if args.htype_csv.exists() else {}

    records = build_import_records(sample_ids, args.silver_spans, dc_types, htype_lookup)
    write_jsonl(records, args.output)
    logging.info("Wrote %d unlabeled records -> %s", len(records), args.output)


if __name__ == "__main__":
    main()
