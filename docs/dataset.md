---
up: 
related: 
created: 2026-07-03
updates: 2026-07-03
in: bibner-werk
tags: []
---
# BibNER-Werk — Dataset Summary

Summarizes the source corpus, stratification, evaluation targets, and gold set sizing behind the BibNER-Werk NER dataset. Synthesized from [ner-bibliographic.md](../notes/ner/ner-bibliographic.md), [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md), [sr08_evaluation-design.md](../notes/ner/sr08_evaluation-design.md), `gemea/notes/project/reprocess-before-after.md`, and `notes/fiz/rms/rm057/rm057.tex`.

---

## 1. Source data: parquet vs. pkl

### 1.1 Provenance

Two corpus generations exist:

| | Old (pkl) | New (parquet) |
|--|--|--|
| File | `DF_DE_TITLES_20240125b.pkl` | `data/processed/de_titles_tokenized.parquet` |
| Producing notebook / script | `2024.01 MT-QA.ipynb` (with `DF_DE_TITLES` originating in `2023.11 NER.ipynb`) | `export_ddb.py` + filter + tokenize (scripted) |

The pkl shares a methodological ancestor with the earlier CC2LLM paper (*How to Turn Card Catalogs into LLM Fodder*, DLnLD @ LREC-COLING 2024 — a self-citation, anonymized in the JCDL paper text per its double-blind policy, see `10-intro.tex`), but the two are not the same filtering pipeline. That paper filtered the DDB collection to German-tagged (`ger`/`zxx` + `langid`-confirmed) text objects, restricted to `hierarchyType ∈ {Monograph, Chapter, Essay, Volume, Manuscript, Letter, Multi-Volume Work}` (7 types), pruned to titles of at least 30 tokens (chosen above the corpus's 20.54-token average), then further restricted to `dc:type = "Monografie"` — yielding a much smaller, purpose-built sample of **108,827 objects**. DF_DE_TITLES (pkl) also confirms German via `langid`, but its `hierarchy_type=content` filter resolves to a different 9-type list (`monografie, kapitel, abschnitt, handschrift, mehrbändiges werk, aufsatz, band, musik, rezension` — includes Musik and Rezension, drops Letter), and it applies neither the token-length cutoff nor the `dc:type` restriction used for the CC2LLM benchmark. It is a methodological cousin, not a shared dataset or a shared filter definition.

### 1.2 Concrete differences between pkl and parquet

| | pkl (4.48M) | parquet (9.21M) | Change |
|--|--|--|--|
| Rows | 4,477,780 | 9,213,339 | +105.8% |
| Language filter | `dc:language=ger` AND `langid=ger` | `dc:language ∈ {ger, gmh, nds, lat}` | langid filter dropped |
| htype filter | `hierarchy_type=content` (notebook-defined) | ADR-01 BLANKET_EXCLUDE (8 types) | — |
| Tokenizer | spaCy (model unspecified) | xlm-roberta-large SentencePiece BPE | — |
| Median `all_tokens` | 8 | 15 | tokenizer granularity, not longer titles |
| Reproducible | No (notebook) | Yes (scripted) | — |

The ~1.1M records recovered in the parquet by dropping the `langid` filter are entirely `ger` (fasttext had misclassified them as non-German in the pkl generation). Latin (`lat`, 493,712 records, 5.4%), Low German (`nds`), and Middle High German (`gmh`) are present in the parquet but were absent from the pkl entirely. ISBD signal rates and silver-tier percentages are stable across both corpora (0.1% / 7.5% / 92.4% tier-2/1/0) — the heuristics are corpus-agnostic; only absolute counts scale. Full derivation: `gemea/notes/project/reprocess-before-after.md`.

---

## 2. Stratification

### 2.1 Why stratification was conducted

NER difficulty varies discontinuously by era — a single pooled F1 hides failures in the hardest and historically most significant stratum (pre-1700). The primary difficulty driver is title length: longer titles mean harder TITLE boundary detection. Token-count-based measures are tokenizer-dependent (spaCy in the pkl/cc2llm era, xlm-roberta BPE in the parquet era — see §1.1), so the defensible metric here is whitespace word count, which is comparable across corpus generations: on the current parquet corpus (`>12`-word threshold), long-title share is highest in pre-1700 (62.8%, 406,123 records) and 1700–1800 (45.4%, 670,580), lowest in 19th-c (21.4%, 3,757,659), then rises again in modern (25.8%, 3,857,886) — not a steady decline; median word count follows the same non-monotonic shape (19 / 11 / 6 / 8). Undated records (521,091, 5.7% of the corpus) skew longer than the surrounding dated eras (30.7% long, median 9 words) and are reported as their own "unknown" stratum, not folded into modern. This supersedes the rm057.tex slide figures (46.9% / 32.0% / 24.3% / 17.6%, a steady drop, `>14` spaCy BPE tokens), which were computed on the retired pkl corpus; see `sr10_era_length_summary_v2_words.csv` (`scripts/ner/sr10_era_length_summary_v2.py --word-tokens`, run against `de_titles_tokenized.parquet`). Either way, per-era reporting is the minimum for a meaningful viability claim — pooled F1 would hide the pre-1700/1700–1800 difficulty spike regardless of which corpus generation or length metric is cited (rm057.tex, "Stratification" and "Evaluation Protocol" frames, for the argument's structure only, not its numbers). Evaluation further excludes micro and macro averaging: micro F1 would be dominated by TITLE (present in ~100% of records, drowning out OTHER_TITLE and PERSON); macro F1 would weight rare PERSON (0.2–8.7% prevalence by era) equally with TITLE, misrepresenting the pipeline's actual utility for GND linking (sr08_evaluation-design.md §4.2).

### 2.2 How stratification was conducted

Gold records are stratified across four dimensions simultaneously, drawn by `sr08_sample_gold.py`:

- **Era**: pre-1700, 1700–1800, 19th-c, modern (≥1900), plus an "unknown" bucket for undated records
- **Silver tier**: tier-2 (structural, `. -` present), tier-1 (heuristic, partial ISBD signal), tier-0 (no ISBD signal — the main NER fallback target, 92.4% of the corpus)
- **`dc_type`**: Leichenpredigt and Einblattdruck are oversampled (drawn era-unconstrained, up to 50 each, then deduplicated against the era strata) because their title-page conventions are the most structurally distinctive and highest-risk failure modes — author-before-title placement, non-standard genre openings
- **Title length**: short/medium/long, since pre-1750 long strings stress boundary detection differently than the short modern majority

No dedicated Latin stratum is drawn: SR-06 found true Latin prevalence in the corpus is ~0.5%, too rare to stratify; Latin records are marked `lang=la` manually if encountered during annotation.

---

## 3. F1 targets

**How they were determined**: targets are grounded in benchmark ceilings, not an assumed intervention rate (sr08_evaluation-design.md §3).

- **HIPE-2022** (closest comparable — historical NER in multilingual documents): German NERC-Coarse strict F1 — best system 0.794, neural baseline 0.703; the *sonar* subset (no train set, most comparable to a fully novel corpus) scores only 0.529 best / 0.307 baseline (Ehrmann et al., 2022)
- **OntoNotes `WORK_OF_ART`**: BERT/RoBERTa typically 0.55–0.68, SOTA ~0.72 — consistently the lowest-scoring entity type in OntoNotes

The NER task here is structurally more favorable than both benchmarks: the entire input *is* the title string, so there is no document-context search problem, only segmentation. This justifies targeting 0.85 for modern records despite OntoNotes `WORK_OF_ART` SOTA sitting at ~0.72. Pre-1700 is where the target concedes ground (0.70) — author-before-title structure and non-standard Early Modern orthography introduce difficulty beyond anything in these benchmarks.

| Stratum | TITLE F1 target | PERSON F1 target |
|---|---|---|
| Modern (≥1900) | ≥ 0.85 | — (person names in title: 0.2%) |
| 19th-c | ≥ 0.80 | — |
| 1700–1800 | ≥ 0.75 | ≥ 0.70 |
| Pre-1700 | ≥ 0.70 | ≥ 0.70 |

PERSON targets apply only to pre-1700 and 1700–1800, where person names structurally appear in the title (8.7% and 5.0% respectively) and `dc:creator`/`dc:contributor` is absent in the majority of records (67.4% and 71.3%) — both conditions needed for NER PERSON extraction to add value over existing metadata. For modern/19th-c, person names almost never appear in the title even when `dc:creator` is absent, so PERSON extraction there is deprioritized regardless of F1.

---

## 4. Gold set size (395 records)

**How the number was reached**: derived from a Wilson-interval minimum-sample-size calculation per era stratum, then adjusted for actual corpus availability (sr08_evaluation-design.md §5–8).

1. **CI target selection**: at ±5 pp CI, the four era strata require a combined minimum of 1,054 records (not feasible under time constraints). At ±10 pp CI, the minimum drops to 265 records (81 pre-1700 + 73 1700–1800 + 62 19th-c + 49 modern) — adopted as the feasible target.
2. **Structural constraints**: the round-number allocation originally planned (~500 records, era × tier × dc_type) hit corpus limits — pre-1700 has **zero** tier-2 records in the corpus (ISBD's `. -` separator postdates pre-1700 printing conventions), so several cells in the original allocation table were structurally impossible.
3. **Actual draw**: `sr08_sample_gold.py` produced **350 records** across the four dated era strata (130 pre-1700, 80 1700–1800, 60 19th-c, 80 modern) against the 81/73/62/49 minimums. Three strata clear their minimum; 19th-c falls 2 records short (60 vs. 62) — rm057.tex marks this cell "≈" rather than a clean pass.
4. **dc_type oversampling add-on**: the era-unconstrained Leichenpredigt/Einblattdruck draw (up to 50 each) contributes **45 net-new records** after deduplication against the era strata — all 45 turned out to be undated, forming an "unknown" era bucket excluded from per-era F1 but retained as an unstratified spot-check.
5. **Total**: 350 (era-stratified) + 45 (dc_type draw, undated) = **395 records**.

PERSON reliability was evaluated separately and found not to be the binding constraint: reaching ±5 pp CI on pre-1700 PERSON would require 3,713 records (not achievable), so PERSON F1 is reported with an explicit wide CI (±8–10 pp) rather than driving gold-set size. TITLE reliability at ±10 pp is the actual binding constraint; the 395-record set clears it in three of four dated strata and comes within 2 records in the fourth (19th-c). `sr08_evaluation-design.md`'s own decision record (§8) rounds this up to "meets the target… across all strata" — worth treating as materially, not literally, true.

A later revision considered reallocating tier composition (boosting tier-0 from 45.1% → 78.5% of the set, since tier-0 is both the hardest and most prevalent inference path) but was deferred — it would have added an estimated 8–15 hours of re-annotation for a reporting-only benefit, since the existing 395-record set already meets the ±10 pp CI target. Decided 2026-03-30; documented as a limitation to report alongside F1, not a blocker.
