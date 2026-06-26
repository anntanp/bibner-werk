---
up: "[[sr08-gold-iaa-script-plan]]"
related:
  - "[[iaa-score-walkthrough]]"
created: 2026-06-26T00:00:00
updates: []
in: bibner-werk
tags:
  - iaa
  - annotation
  - krippendorff
---

# Inter-Annotator Agreement

## 0. Annotators

| Handle  | Background |
|---------|-----------|
| frankp  | Multilingual native German speaker; history major; computational linguist |
| maria   | Library science major; DDB-EDM XSLT mapping expert |
| gundi   | Works with Maria |

## 1. Round 1 — export-20260623-1110

**Script**: `scripts/ner/sr08_compute_iaa.py`
**Export**: `data/annotation/doccano/export-20260623-1110`
**Metric**: Krippendorff's α (nominal distance), threshold α ≥ 0.800 (Artstein & Poesio 2008)

### 1.1 Annotator coverage

| Annotator | Reviewed / Total | Notes |
|-----------|-----------------|-------|
| frankp    | 394 / 395       | 1 unreviewed |
| gundi     | 395 / 395       | 130 empty-label records with "Keine Annotation" comment → O |
| maria     | 392 / 395       | 98 empty-label records with comment → O; 3 unreviewed |

### 1.2 Results

| Scope          |     α | n_items | N_judg |   D_o |   D_e |
|----------------|------:|--------:|-------:|------:|------:|
| ALL            | −0.036 |     395 |   1757 | 0.6532 | 0.6306 |
| OTHER_TITLE    | 0.387  |     395 |   1181 | 0.2744 | 0.4478 |
| PARALLEL_TITLE | 0.174  |     395 |   1181 | 0.0221 | 0.0268 |
| PERSON         | 0.364  |     395 |   1181 | 0.1427 | 0.2245 |
| TITLE          | 0.202  |     395 |   1181 | 0.2498 | 0.3129 |
| TRANSLATOR     | 0.439  |     395 |   1181 | 0.0085 | 0.0151 |

**Outcome**: FAIL — overall α = −0.036, below chance agreement. No per-label score reaches 0.800.

### 1.3 Pairwise α

| Pair           | Scope          |      α | n_items | N_judg |    D_o |    D_e |
|----------------|----------------|-------:|--------:|-------:|-------:|-------:|
| frankp/gundi   | ALL            | −0.182 |     395 |   1205 | 0.7426 | 0.6283 |
| frankp/gundi   | OTHER_TITLE    |  0.358 |     395 |    789 | 0.2919 | 0.4547 |
| frankp/gundi   | PARALLEL_TITLE |  0.362 |     395 |    789 | 0.0254 | 0.0398 |
| frankp/gundi   | PERSON         |  0.312 |     395 |    789 | 0.1777 | 0.2582 |
| frankp/gundi   | TITLE          | −0.181 |     395 |    789 | 0.3274 | 0.2773 |
| frankp/gundi   | TRANSLATOR     |  0.396 |     395 |    789 | 0.0076 | 0.0126 |
| frankp/maria   | ALL            | −0.201 |     395 |   1161 | 0.7114 | 0.5923 |
| frankp/maria   | OTHER_TITLE    |  0.332 |     395 |    786 | 0.2762 | 0.4135 |
| frankp/maria   | PARALLEL_TITLE | −0.019 |     395 |    786 | 0.0307 | 0.0301 |
| frankp/maria   | PERSON         |  0.393 |     395 |    786 | 0.1637 | 0.2696 |
| frankp/maria   | TITLE          | −0.114 |     395 |    786 | 0.2455 | 0.2205 |
| frankp/maria   | TRANSLATOR     |  0.746 |     395 |    786 | 0.0051 | 0.0202 |
| gundi/maria    | ALL            | −0.018 |     395 |   1148 | 0.6677 | 0.6559 |
| gundi/maria    | OTHER_TITLE    |  0.456 |     395 |    787 | 0.2551 | 0.4686 |
| gundi/maria    | PARALLEL_TITLE | −0.008 |     395 |    787 | 0.0102 | 0.0101 |
| gundi/maria    | PERSON         |  0.366 |     395 |    787 | 0.0867 | 0.1367 |
| gundi/maria    | TITLE          |  0.573 |     395 |    787 | 0.1760 | 0.4121 |
| gundi/maria    | TRANSLATOR     | −0.009 |     395 |    787 | 0.0128 | 0.0126 |

gundi/maria has the highest ALL α (−0.018) and TITLE α (0.573), indicating the strongest pairwise agreement. frankp/maria has the weakest ALL α (−0.201) and the highest TRANSLATOR α (0.746 — the only per-label score approaching the 0.800 threshold across all pairs). TITLE is the most variable label across pairs (−0.181 to 0.573), reflecting boundary disagreements between annotators.

### 1.4 Verification

Values verified against script output (`sr08_compute_iaa.py`, export-20260623-1110).

**§1.2 (3-way)**

| Scope | N_judg | D_o | D_e | Status |
|---|---|---|---|---|
| ALL | 1757 | 0.6532 | 0.6306 | ✓ |
| OTHER_TITLE | 1181 | 0.2744 | 0.4478 | ✓ |
| PARALLEL_TITLE | 1181 | 0.0221 | 0.0268 | ✓ |
| PERSON | 1181 | 0.1427 | 0.2245 | ✓ |
| TITLE | 1181 | 0.2498 | 0.3129 | ✓ |
| TRANSLATOR | 1181 | 0.0085 | 0.0151 | ✓ |

**§1.3 (pairwise)** — all 18 rows match.

N_judg: per-label rows for a pair always show exactly |docs reviewed by a1| + |docs reviewed by a2| (789, 786, 787 respectively) because each annotator contributes exactly one judgment per reviewed doc in per-label scope (either L or O). The ALL-scope N_judg is higher (1205, 1161, 1148) because multi-label docs (e.g. frankp assigns TITLE + PERSON) each contribute multiple judgments — one per label type.

### 1.5 Interpretation

Overall α below zero means observed disagreement exceeds what chance alone would produce. This is expected at the pre-adjudication stage: the three annotators applied different span boundaries and entity type conventions, and PARALLEL_TITLE and TRANSLATOR were not in the original label guide used by all annotators. TRANSLATOR shows the highest per-label score (0.439), likely because it is a rare and unambiguous role. TITLE and PARALLEL_TITLE are the weakest, reflecting boundary and nesting disagreements.

These scores motivate adjudication before releasing the gold set.

#### 1.5.1 On negative α

Negative α is mathematically valid. Since α = 1 − D_o/D_e, α < 0 whenever D_o > D_e — observed disagreement exceeds expected disagreement by chance. Here D_o = 0.6532 and D_e = 0.6306, so α = 1 − 1.036 = −0.036. The excess is small; this is essentially chance-level agreement.

The ALL-scope α is negative because the computation is multi-label: a document where frankp assigns TITLE + PERSON contributes two label judgments, while gundi assigns O to the same document. That structural mismatch inflates D_o relative to D_e, which is computed from the pooled label distribution and assumes single-label items. Annotators with very different labeling density (frankp labels almost everything; gundi assigns O to 130 documents) push D_o above D_e in the ALL scope.

The per-label α values are the more meaningful metric for this task. Each reduces to a clean binary (assigned L vs. did not assign L per reviewed document), which is the setting the formula was designed for. Those are all positive (0.174–0.439) and are the figures that should be reported in the paper. The ALL α is worth noting as a summary figure, but its sign is a known artefact of mixing multi-label counts into a single-label formula.

### 1.6 Output files

- `data/processed/ner/sr08_iaa_counts_export-20260623-1110.csv` — one row per label type per annotator per document
- `data/processed/ner/sr08_iaa_export-20260623-1110.csv` — α per scope (ALL + per label)
- `data/processed/ner/sr08_iaa_pairwise_export-20260623-1110.csv` — pairwise α for all annotator pairs
