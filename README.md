# BibNER-Werk: Bibliographic NER for Historical German Catalogs

**BibNER-Werk** = Bibliographic NER for FRBR Work-level entities. *Werk* (German for "work") denotes the FRBR/LRM Work level.

Named entity recognition for FRBR Work-level entities (TITLE, OTHER_TITLE, PERSON) in historical German bibliographic catalog strings from the Deutsche Digitale Bibliothek (DDB).

## Overview

Standard NER models fail on ISBD-formatted bibliographic catalog strings across three dimensions: wrong label set, wrong domain, and wrong register. BibNER-Werk addresses this via:

1. **Silver labeling**: ISBD-guided signal extraction covering 7.6% of 9.2M DDB records (tier-1/2)
2. **Gold evaluation**: 395-record stratified gold set (by era and ISBD tier)
3. **Fine-tuned model**: `urchade/gliner_multi-v2.1` trained on silver labels

## Setup

```bash
cd bibner-werk
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

`torch` is deliberately not pinned in `requirements-train.txt` — install it separately, matched to the server's CUDA driver (see the note at the top of that file for the A100 vs. V100S path).

Dedicated venv — does not share or depend on the companion GeMeA project's venv, following an earlier project split.

## Datasets

- `data/silver/silver_spans.jsonl`: 49,999 tier-1 + 2,384 tier-2 silver-labeled records (52,383 total)
- `data/annotation/doccano/export-20260623-1110/`: 395-record gold set (per-annotator exports; adjudicated version at `data/annotation/doccano/adjucated-20260707-1557.jsonl`)
- `data/gold/`, `data/iaa/`, `data/adjudication/`, `data/heuristics/`: git-tracked ground truth, IAA scores, and adjudication decisions (not reproducible from scripts alone)
- `data/provenance/bibner-data-provenance.parquet`: individually un-ignored provenance record
- `data/inputs.md`: input file inventory (paths, formats, provenance, schema)

## Documentation

- `docs/`: dataset summary, annotation guides (EN/DE), adjudication method and rulings, ISBD field-rating and other ADRs, inter-annotator agreement, GeMeA context
- `scripts/README.md`: index of all scripts in `scripts/py/` and `scripts/sh/`

## Related

- GeMeA — German Memory Atlas KG (companion repository, link omitted for anonymous review); BibNER-Werk is GeMeA Phase 0 NER component
- JCDL 2026 paper: *BibNER-Werk: Bibliographic NER for Historical German Catalogs*
