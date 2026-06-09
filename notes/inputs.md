# WEM Model — Input Inventory

## Gold annotation set

**Path**: `data/annotation/doccano/export-20260608-1221/`
**Format**: Doccano JSONL — `{"id": N, "text": "...", "label": [[start, end, "LABEL"], ...]}`
**Provenance**: Manual annotation of 395 DDB catalog title strings, stratified by era and ISBD tier
**Size**: 395 records per annotator file
**Key fields**: `id` (DDB record ID), `text` (raw title string), `label` (char-offset span triples)
**Annotators**:
- `frankp.jsonl`: 393/395 annotated (primary annotator)
- `maria.jsonl`: 131/395 (IAA subset)
- `gundi.jsonl`: 82/395 (IAA subset)
- `ann.jsonl`: 183/395 (Claude pre-annotation; seeding only — not used for IAA)
**Schema quirks**: `label: []` means unannotated (not "no entities"); always check before treating empty label as confirmed negative

---

## Doccano label schema

**Path**: `data/annotation/doccano/label_import.json`
**Format**: Doccano label configuration JSON
**Provenance**: Exported from Doccano project for WEM Phase 1
**Labels**: TITLE, OTHER_TITLE, PERSON (Phase 1); TRANSLATOR, PARALLEL_TITLE, SERIES reserved for Phase 2

---

## Gold set prefilled (pre-annotation)

**Path**: `data/annotation/sr08_gold_prefilled.jsonl` (also `.json`)
**Format**: JSONL with prefilled Claude annotations
**Provenance**: Claude pre-annotation of the 395-record gold sample; used to seed Doccano annotation
**Size**: 395 records

---

## Gold sample metadata

**Path**: `data/annotation/sr08_gold_sample.csv`
**Format**: CSV with metadata columns (record ID, era, tier, dc_type, etc.)
**Provenance**: Stratified sample drawn from DDB parquet per sr08_gold-set-composition.md
**Key fields**: `id`, `era`, `tier`, `dc_type`, `title_length_tokens`

---

## Prompt validation data

**Path**: `data/annotation/sr11_prompt_validation_manual.jsonl`
**Format**: JSONL with LLM-labeled spans
**Provenance**: Manual validation of sr11 labeling strategy prompts

---

## DDB parquet (upstream — stays in gemea)

**Path**: `~/Documents/claude/gemea/data/` (do not copy; pass as CLI argument)
**Format**: Parquet; schema in `gemea/data/schema/parquet_schema.json`
**Size**: 9,213,339 records (9.21M)
**Key fields**: `id`, `title` (raw title string), `tier` (0/1/2), `era`, `dc_type`, `sparte`, `mt`
**Encoding**: UTF-8; historical orthography variation in pre-1800 records
**Processing assumptions**: BPE tokenization (xlm-roberta-large SentencePiece); median 15 tokens per title
