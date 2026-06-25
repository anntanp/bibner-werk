---
up: "[[sr08_gold-set-composition]]"
related: "[[sr08_doccano-meta]]"
created: 2026-06-23
updates: 2026-06-25, in: "[[gemea]]"
in: bibner-werk
tags: [ner, iaa, annotation, script-plan]
---

# SR-08: IAA Computation Script Plan

## 1. Context

`sr08_gold-set-composition.md §7.1` specifies the IAA methodology — pairwise span F1, not Cohen's κ, over exact character-offset + label triples — but no script exists yet to compute it. The doccano gold annotation lives in per-annotator JSONL files under `data/annotation/doccano/`. Targets: ≥ 0.80 overall, ≥ 0.75 for pre-1700 PERSON. This script makes IAA reportable for the paper.

## 2. Output file

`scripts/ner/sr08_compute_iaa.py`

## 3. Data inputs

- `--export-dir` (required): path to a doccano export directory, e.g. `data/annotation/doccano/export-20260623-1110` (latest as of 2026-06-23)
- Active annotator files in that dir (auto-discovered): `frankp.jsonl`, `gundi.jsonl`, `maria.jsonl`
- Gold file: `ann.jsonl` — included only when `--include-gold` is passed

**Skip unconditionally:** `dummy.doccano.jsonl`, any `*.meta.jsonl`, any `sr08_*` prefixed file

**JSONL record format** (one JSON object per line):
```json
{"id": 4, "text": "Zur Frage...", "label": [[0,45,"TITLE"],[48,93,"OTHER_TITLE"]], "Comments": []}
```
Spans are `[start, end, label]` character offsets (Python slice notation).

## 4. Algorithm

For each pair of annotators (A, B):
1. Restrict to documents where **both reviewed** the record. A record counts as reviewed if `label != []` OR `Comments != []` — annotators leave a comment when they find no entities (e.g. gundi's "Keine Annotation" for structural htypes). Never use `label != []` alone: gundi has 130 empty-label records with comments; maria has 98; these are valid "no entity" annotations and must be included.
2. For each scope in `[ALL] + per-label` (TITLE, OTHER_TITLE, PERSON, etc.):
   - `spans_a` = frozenset of `(start, end, label)` from A, filtered to scope
   - `spans_b` = frozenset of `(start, end, label)` from B, filtered to scope
   - `tp = |spans_a ∩ spans_b|`, `fp = |spans_b \ spans_a|`, `fn = |spans_a \ spans_b|`
   - Symmetric F1: `2*tp / (2*tp + fp + fn)` (micro-aggregate across all co-annotated docs)

This is the symmetric span F1 specified in §7.1 — treating one as reference and other as system yields the same F1 in both directions.

## 5. CLI

```bash
python scripts/ner/sr08_compute_iaa.py \
    --export-dir data/annotation/doccano/export-20260623-1110 \
    [--include-gold]      # also compare each annotator vs ann.jsonl
    [--output-dir DIR]    # default: data/processed/ner/
```

## 6. Outputs

Two CSVs (named with the export dir stem):

**`sr08_iaa_summary_<export>.csv`** — one row per (annotator_a, annotator_b, label):
`annotator_a, annotator_b, label, n_docs, tp, fp, fn, precision, recall, f1`

**`sr08_iaa_detail_<export>.csv`** — one row per (annotator_a, annotator_b, doc_id, label):
`annotator_a, annotator_b, doc_id, label, tp, fp, fn`

**Stdout:** N co-annotated docs per pair; pairwise F1 matrix (overall); per-label F1 table.

## 7. Code conventions (follow `sr08_count_comments.py`)

- `BASE = Path(__file__).resolve().parents[2]`
- `SKIP_NAMES = {"dummy.doccano.jsonl"}`, `SKIP_SUFFIXES = (".meta.jsonl",)`, `SKIP_PREFIXES = ("sr08_",)`
- `ann.jsonl` added to `SKIP_NAMES` unless `--include-gold` is set
- stdlib only — no pandas, no seqeval
- `--export-dir` required (no hardcoded default); `--output-dir` defaults to `BASE / "data/processed/ner/"`
- `out_path.parent.mkdir(parents=True, exist_ok=True)` before writing

### Header block

```python
#!/usr/bin/env python3
# Purpose:      Compute pairwise span-F1 IAA between annotators in a Doccano export
# Usage:        python scripts/ner/sr08_compute_iaa.py \
#                   --export-dir data/annotation/doccano/export-20260623-1110
# Inputs:       *.jsonl files in --export-dir (skips dummy, ann unless --include-gold)
# Outputs:      data/processed/ner/sr08_iaa_summary_<export>.csv
#               data/processed/ner/sr08_iaa_detail_<export>.csv
#               stdout: pairwise F1 matrix + per-label table
# Dependencies: none (stdlib only)
# Assumptions:  label field is [[start,end,label_str],...]; empty list = no entities annotated
```

## 8. Verification

```bash
python scripts/ner/sr08_compute_iaa.py \
    --export-dir data/annotation/doccano/export-20260623-1110
```

Annotation counts as of 2026-06-23 export (`label != []` OR `Comments != []`):

| Annotator | Reviewed / Total | Notes |
|-----------|-----------------|-------|
| frankp    | 394 / 395       | 1 unreviewed |
| gundi     | 395 / 395       | 130 empty-label + comment ("Keine Annotation") |
| maria     | 392 / 395       | 98 empty-label + comment; 3 empty+no-comment treated as reviewed |

Note: `export-20260623-1110` already subsumes `export-20260608-1221` — no cross-export merge needed.

Expected co-reviewed doc counts:

| Pair            | Co-reviewed |
|-----------------|------------|
| frankp / gundi  | 394        |
| frankp / maria  | 391        |
| gundi / maria   | 392        |

Expected outputs:
- Summary CSV at `data/processed/ner/sr08_iaa_summary_export-20260623-1110.csv`
- Per-label table printed to stdout; overall F1 above sanity floor (≥ 0.70)
