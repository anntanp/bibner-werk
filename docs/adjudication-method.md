---
up: []
related:
  - "[[inter-annotator-agreement]]"
created: 2026-07-06T00:00:00
updates:
  - 2026-07-07
in: bibner-werk
tags:
  - ner
  - iaa
  - adjudication
---

# Adjudication Method — SR-08 Gold Reconciliation

## 1. Objective

Resolve three-way annotation disagreements in the round-1 export (Annotator A, Annotator C, Annotator B; 395 docs — see [inter-annotator-agreement.md](inter-annotator-agreement.md) §1 for the underlying IAA scores) into a single `gold_reconciled.jsonl`. Automatic resolution is used for unambiguous agreement; every genuine conflict goes through PI-led, LLM-assisted adjudication.

**PI** (Principal Investigator) — the project lead and sole adjudicator for this study; not one of the three annotators (Annotator A, Annotator C, Annotator B), and distinct from the LLM, which only drafts a suggestion (§3).

- **Adjudicator**: PI, assisted by an LLM-drafted suggestion per conflict document — not auto-applied.
- **Scope disagreement** (one annotator labels a chapter/section/article-level unit; the other two mark it `O`): resolved case-by-case, not by a blanket rule.
- **Boundary disagreement** (2/3 agree a label is present but spans differ): always routed to adjudication, never auto-merged.

## 2. Resolution tiers

Computed against the round-1 export, unit = one document across its full label set (not per label type, since scope and label-set conflicts interact within a document).

| Bucket | Meaning | n docs |
|---|---|---|
| 1 — auto-accept | All reviewing annotators (2 or 3) assigned the exact same labels **and** the exact same character spans — no disagreement, used as-is | 42 |
| 2 — majority-auto | Exactly 2 of 3 annotators agree on labels and spans; the third's dissent is discarded automatically | 50 |
| 3 — adjudicate: scope | Systematic disagreement over whether a chapter/section/article-level unit counts as a Work. No blanket rule — every one of these goes to PI review individually | 142 |
| 3 — adjudicate: boundary | Annotators agree an entity is present but disagree on the exact start/end offsets | 138 |
| 3 — adjudicate: three-way | No majority at all — three genuinely different readings of the string, not explained by the scope question | 23 |
| 3 — adjudicate: insufficient coverage | Fewer than 2 annotators reviewed the doc | 0 |
| **Total** | | **395** |

Tiers 1 and 2 (92 docs, 23%) resolve automatically. Tier 3 (303 docs, 77%) requires PI review — consistent with the negative overall Krippendorff's α reported in [inter-annotator-agreement.md](inter-annotator-agreement.md) §1.5.1: most of the corpus needs a human decision, not a formula.

**Why document-level, not label-level, tiering**: IAA is computed per label type because that is what Krippendorff's α needs. Adjudication instead resolves one document at a time — a document with a scope conflict on `TITLE` and a boundary conflict on `PERSON` is one adjudication decision, not two, since the underlying judgment call (is this a Work at all?) usually determines both.

## 3. Pipeline

```bash
python scripts/ner/sr08_classify_disagreements.py --export-dir data/annotation/doccano/export-20260623-1110
python scripts/ner/sr08_draft_adjudication.py --export-dir data/annotation/doccano/export-20260623-1110
# --- manual: LLM drafts a suggested resolution per prompt batch, saved as batch_NN.json ---
python scripts/ner/sr08_ingest_llm_drafts.py --export-dir data/annotation/doccano/export-20260623-1110 \
    --responses-dir data/annotation/adjudication_responses/
python scripts/ner/sr08_adjudication_to_doccano.py \
    --export-dir data/annotation/doccano/export-20260623-1110 \
    --classification-csv data/adjudication/sr08_disagreement_classification_export-20260623-1110.csv \
    --sheet data/adjudication/sr08_adjudication_sheet_export-20260623-1110.jsonl \
    --output-en data/adjudication/sr08_adjudication_import_en_export-20260623-1110.jsonl \
    --output-de data/adjudication/sr08_adjudication_import_de_export-20260623-1110.jsonl
# --- manual: PI imports the EN or DE file into Doccano, reviews/edits all 395 docs (not just tier-3), ---
# --- exports the completed project as data/annotation/doccano/adjucated-<date>.jsonl ---
# --- manual: `id` renumbered 1-395 to match doc_id, then copied directly to gold_reconciled.jsonl -- ---
# --- no merge script: the export already has every doc's final label, so there was nothing to merge ---
# --- manual: verification checked directly (395 records, ids 1-395, no duplicates) -- no dedicated script ---
```

LLM drafting is manual, not API-driven: a script batches tier-3 documents into paste-able prompt files rather than calling the Anthropic SDK, and a second script reads the pasted responses back in.

**Deviation from the original plan** (`adjudication-plan.md` §0.1, §2.6/§2.7): a separate `sr08_ingest_adjudication.py` → `sr08_merge_gold.py` → `sr08_verify_gold_reconciled.py` chain was planned to fold the PI's tier-3 decisions back into the canonical sheet and merge with tiers 1/2. It was never built: the PI's Doccano export already covers all 395 docs (tiers 1+2+3) with final resolved labels, not just tier-3's `pi_decision`, so there was nothing left to merge. `gold_reconciled.jsonl` is therefore a direct copy of the Doccano export (`id`/`text`/`Comments`/`label` only) — it does **not** carry the `_adjudication` metadata key or per-tier `resolution_source` that the original plan called for. `sr08_adjudication_sheet_<export>.jsonl`'s `pi_decision` field is consequently null for all 303 tier-3 docs; the PI's real final decision for each doc is recoverable only by joining against the Doccano export by `doc_id`/`id` — which is exactly what §5's `sr08_compute_llm_acceptance.py` does.

## 4. Outputs

| File | Producer | Description |
|---|---|---|
| `sr08_disagreement_classification_<export>.csv` | `sr08_classify_disagreements.py` | Tier + conflict_type per document |
| `sr08_adjudication_prompt_batch_NN_<export>.txt` | `sr08_draft_adjudication.py` | Paste-able LLM prompts, ~10–15 tier-3 docs each |
| `sr08_adjudication_sheet_<export>.jsonl` (skeleton) | `sr08_draft_adjudication.py` | One record per tier-3 doc; LLM/PI fields null |
| `data/annotation/adjudication_responses/batch_NN.json` | manual (LLM) | LLM responses, one file per batch |
| `sr08_adjudication_sheet_<export>.jsonl` (filled) | `sr08_ingest_llm_drafts.py` | Skeleton sheet with `llm_suggested_label`/`llm_rationale`/`llm_confidence` filled; `pi_decision` stays null (see deviation note above) |
| `sr08_adjudication_import_en_<export>.jsonl`, `..._de_<export>.jsonl` | `sr08_adjudication_to_doccano.py` | All 395 docs reshaped for Doccano import (tier 1 "100% Agreed", tier 2 majority reason, tier 3 LLM suggestion pre-filled) |
| `data/annotation/doccano/adjucated-<date>.jsonl` | manual (PI, Doccano) | PI's completed Doccano export for all 395 docs — the actual final-decision file, despite the different name from the original plan's `sr08_adjudication_export_<export>.jsonl` |
| `data/gold/gold_reconciled.jsonl` | manual (direct copy) | Final gold set — `id`/`text`/`Comments`/`label` only, copied from the Doccano export above; no separate merge script, no inline `_adjudication` metadata |

`sr08_adjudication_sheet_resolved_<export>.jsonl` and `sr08_adjudication_provenance_<export>.csv` (originally planned as §2.6/§2.7 outputs) were never produced — see the deviation note above. `sr08_llm_acceptance_<export>.csv` (§5) now serves the tier-3 provenance role for the paper's methods section instead.

## 5. LLM acceptance/override analysis

Measures how often the PI kept vs. overrode the LLM's tier-3 suggestion, for the paper's reproducibility/methods claims about LLM-assisted adjudication. Full methodological justification (why this is a valid practice, not just a convenience) is in [adjudication-llm-justification.md](../../notes/fiz/bibner-werk/adjudication-llm-justification.md); this section documents the script and results.

**Script**: `scripts/ner/sr08_compute_llm_acceptance.py`

```bash
python scripts/ner/sr08_compute_llm_acceptance.py \
    --sheet data/adjudication/sr08_adjudication_sheet_export-20260623-1110.jsonl \
    --adjudicated data/annotation/doccano/adjucated-20260707-1557.jsonl
```

**Method**: joins each tier-3 doc's `llm_suggested_label` (from the sheet) against the PI's final `label` (from the Doccano export, keyed by `doc_id` == `id`) and classifies the outcome by direct span comparison — never by presence/absence of a PI `Comments` entry, since only 13/395 docs carry any comment at all and most PI edits are silent:

- `accepted_verbatim` — final span set identical to the LLM's suggestion
- `accepted_boundary_edit` — same label composition, at least one offset moved
- `overridden` — label composition changed (a label added, removed, or its count changed)

**Outputs**:

| File | Description |
|---|---|
| `sr08_llm_acceptance_<export>.csv` | One row per tier-3 doc: `doc_id`, `conflict_type`, `llm_confidence`, `outcome`, `contested_ref`, `precedent_ref`, `pi_comment` |
| `sr08_llm_acceptance_summary_<export>.csv` | One row per stratum (overall / each `conflict_type` / each `llm_confidence`): outcome counts, `total`, `override_pct` — backs the tables below |

**Results** (`export-20260623-1110`, 2026-07-07):

| conflict_type | accepted_verbatim | accepted_boundary_edit | overridden | total | override % |
|---|---|---|---|---|---|
| scope | 87 | 5 | 50 | 142 | 35.2% |
| boundary | 90 | 12 | 36 | 138 | 26.1% |
| three_way | 13 | 4 | 6 | 23 | 26.1% |
| **overall** | **190** | **21** | **92** | **303** | **30.4%** |

| llm_confidence | accepted_verbatim | accepted_boundary_edit | overridden | total | override % |
|---|---|---|---|---|---|
| high | 111 | 12 | 29 | 152 | 19.1% |
| medium | 66 | 9 | 55 | 130 | 42.3% |
| low | 13 | 0 | 8 | 21 | 38.1% |

**Key findings**: overall acceptance (verbatim + boundary edit) is 69.6% — the PI kept the LLM's substantive call on 7 in 10 tier-3 docs. Override is highest on scope conflicts (35.2%), matching §1's "no blanket rule" design — these are exactly the docs the LLM was told to judge case-by-case, and the PI still disagrees with it more than a third of the time. Confidence calibration is only partial: `high` correlates with a much lower override rate (19.1%), but `medium` is overridden *more* than `low` (42.3% vs. 38.1%). This traces largely to ISBD-style punctuation (`" : "`, `" / "`, `" = "`) being present but structurally unreliable — within medium-confidence docs, those containing such a delimiter are overridden nearly as often as not (54%, vs. 35% without one), both for pre-modern documents where the punctuation predates ISBD convention entirely and for 20th-century serial reports where a leading year mimics a title-introducing colon. (The same ISBD-delimiter breakdown was not separately checked within the low-confidence bucket, which has only 21 docs total — this comparison is suggestive, not a fully controlled one.) See `adjudication-llm-justification.md` §3.2 for the doc-level examples.

## 6. Gold-set `obj_id` reconciliation against `s2_meta.parquet` (2026-07-07)

**Why this exists**: the gold sample was drawn (`sr08_gold-set-composition.md` §4.1) from `data/DF_DE_TITLES_20240125b.pkl`, a corpus snapshot dated 2024-01-25. This was the correct, best-available corpus at annotation time — the snapshot was not outdated at the point of sampling, and the annotated text itself is untouched by anything downstream. The mismatch surfaced only when building `data/provenance/bibner-data-provenance.parquet` (joining gold + silver `doc_id`s to their row in the current GeMeA `s2_meta.parquet`, built 2026-05-22; moved 2026-07-08 from `data/gold/` since the table covers silver records too, not gold alone): DDB re-ingested/restructured part of its periodical holdings between the two snapshots, reassigning `obj_id`s and in some cases splitting single records into finer-grained sub-issues. 18 of 395 gold `obj_id`s (4.6%) had no match in `s2_meta.parquet` as a result.

**Resolution**: for each unmatched `obj_id`, searched `s2_meta.parquet`'s `title` column for the closest surviving title, normalizing cosmetic reformatting (duplicate volume/year tokens collapsed, `:`/`=` spacing changes), then picked the candidate whose remaining distinguishing detail (volume/issue number, or full-metadata comparison) still agreed with the original. 17 of 18 resolved:

- **14 docs** (23, 49, 52, 58, 77, 85, 133, 144, 178, 225, 242, 332, 387, 395): exact-content match after normalization — reformatting only, no ambiguity.
- **2 docs** (10, 257): the surviving record exists as 2–3 byte-for-byte identical duplicates (same title, `dc_type`, `agents`, `dates`, `provider_id`, `dataset_id` — everything) under different `obj_id`s. No field distinguishes them; one was picked arbitrarily.
- **1 doc** (293): the closest surviving title differs substantively (an inserted clause, a different Beilage/Nr. number), not just cosmetically. `gold_reconciled.jsonl`'s `text` and `label` offsets were edited to match — the `OTHER_TITLE` span was recomputed to point at the corresponding substring in the new text, not reused verbatim. This is the only gold record whose annotated text changed; all other renamed docs kept their original text and label offsets, only `obj_id`/`parquet_row_id` changed.

**1 doc left unresolved** (doc_id 393, `WA23UASKQOQTRYB27EXUD4BXIF2SKMPI`, "Praxis Fidei Salvificae..." — a pre-1700 Predigtsammlung, dated 1627): exhaustive search found nothing. Zero titles in `s2_meta.parquet` contain any distinctive fragment of the original title (`Praxis Fidei Salvificae`, `Ernewrung des innern Menschen`, `Sontages Episteln`, `Erbawung und Besserung` all returned zero hits); broadening to metadata alone (`lang_obj=ger`, `dc_type` containing `Predigtsammlung`, publication year within ±10 of 1627) also returned zero candidates carrying both original `dc_type` tags (`Predigtsammlung`+`Monografie`). This record was accepted as genuinely absent from the current corpus rather than replaced with a low-confidence substitute — **doc_id 393 keeps its original (now-stale) `obj_id`** in both `gold_reconciled.jsonl` (text/label untouched) and `bibner-data-provenance.parquet` (`parquet_row_id` is null). This is a known, documented gap, not an oversight: 1/395 gold records (0.25%) has no corresponding row in `s2_meta.parquet`.

**Caveat**: matches beyond an exact reformatting-only string (the 3 cases above) were chosen as "closest available" from title text and metadata, not independently verified against the live DDB item page. Current `obj_id` per `doc_id` is in `data/provenance/bibner-data-provenance.parquet`.
