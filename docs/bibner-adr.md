---
up: 
related: 
created: 2026-03-26
updates:
  - 2026-06-25
  - 2026-07-06
  - 2026-07-16
in: bibner-werk
tags: []
---
# Architecture Decision Record — NER for Bibliographic Title Extraction

**Status:** In progress — SR-08 annotation pending; SR-09, SR-11 blocked
**Scope:** NER pipeline for extracting structured bibliographic entities from `DF_DE_TITLES` — a filtered snapshot of the DDB corpus (4.47M records; `ddb:hierarchyType` = `content`, `dc:language` + `langid` = German)
**Full notes:** [ner-bibliographic.md](../ner-bibliographic.md)

---

## 1. Context

DDB title strings are the primary text surface for bibliographic entity extraction in GeMeA. The goal is to extract Work-level entities (`TITLE`, `OTHER_TITLE`, `PERSON`) to support GND work linking. `DF_DE_TITLES` is a filtered snapshot (4.47M records) derived from the DDB corpus by selecting `ddb:hierarchyType` = `content`, then applying `dc:language` + `langid` = German — not the full DDB corpus. It spans pre-1700 Early Modern German to contemporary titles, with no consistent structural markup.

ISBD punctuation (`. -`, ` :`, ` /`) provides structural signals in a minority of records. NER is the fallback for the majority. The two approaches are complementary: ISBD parsing handles structured records reliably; NER handles the unstructured majority.

Each record is assigned a **silver tier** by `sr01_rate_isbd_fields.py` based on how many and which ISBD heuristic fields fire:

| Tier | Criteria | Corpus share | Role |
|---|---|---|---|
| **2** | `has_dot_dash` AND `f_person` AND at least one of `f_edition`, `f_place`, `f_publisher`, `f_year`, `f_series` | 0.1% (4,613 records) | Primary silver training set — structural area separator present; multi-field span labeling with high confidence |
| **1** | `n_fields ≥ 3` OR (`f_person` AND `f_year`) | 7.5% (335,524 records) | Augmentation set — partial ISBD evidence; Work + Expression level annotation |
| **0** | All others — no reliable ISBD signal | 92.4% (4,137,643 records) | Not selected as silver candidates; primary NER inference target |

`has_dot_dash` (`. -` area separator) is the gating condition for tier-2: it is present in only 1.2% of records, which directly caps tier-2 at 0.1%. Tier-0 is where NER does all the work.

**Note (2026-07-08): the tier-2 count above is what `sr01_rate_isbd_fields.py` itself computes and writes to `sr01_isbd_field_ratings.csv` — it is not the count actually used to select titles for `silver_spans.jsonl`.** `build_silver_spans.py` recomputes tier-2 locally, substituting `f_resp_person` for `f_person` (D-04's tightening), since the ratings CSV predates that amendment. The operative tier-2 count for span selection is **2,384**, not 4,613 — see D-03 below and `A2-silver-spans-script-plan.md` §2.4.

---

## 2. Decisions

### D-01 — ISBD as primary extractor, NER as fallback

**Decision:** Rule-based ISBD parser runs first. NER only runs when ISBD fails to produce a confident extraction.

**Why:** ISBD signals (`. -`, ` :`, ` /`) are high-precision when present. Silver label analysis (SR-01) shows tier-2 precision is near-perfect for structural records. NER is slower, less deterministic, and harder to explain — it should not replace a reliable rule when one exists.

**Consequence:** NER applies to ~92.4% of the corpus (tier-0, no ISBD signals) — the majority, not a small edge case.

**Notes:** [sr01_isbd-field-rating.md](sr01_isbd-field-rating.md) · [sr01_isbd-applicability.md](sr01_isbd-applicability.md) · [silver-dataset-pipeline.md](silver-dataset-pipeline.md)

---

### D-02 — Parser must prioritise ` :` over ` /`

**Decision:** `OTHER_TITLE` / `TITLE` boundary uses ` :` as the primary split signal. ` /` (SoR) is secondary.

**Why (SR-02):** ` :` appears in 20.2% of titles; ` /` appears in only 0.8%. Prioritising ` /` would miss 96% of subtitle splits.

**Notes:** [sr01_isbd-field-rating.md](../notes/fiz/bibner-werk/ner/sr01_isbd-field-rating.md) · [isbd-field-rating-adr.md](isbd-field-rating-adr.md)

---

### D-03 — Silver tier thresholds and field exclusions

**Decision:** Tier-2 = `has_dot_dash` + ≥2 additional heuristic fields. Excluded from silver labels: `f_parallel` (~80% FP), `f_edition` (~83% FP), trailing `.` (93% FP). Post-filter required on `f_person` (~36% FP) and `f_person_compound` (~29% FP).

**Why (SR-03, SR-05):** FP rates above 15% degrade silver label quality below usefulness. `f_parallel` and `f_edition` were found to fire predominantly on false positives. Trailing `.` adds no detection power beyond `has_dot_dash`.

**Implementation update (2026-07-08, A2/`build_silver_spans.py`):** this decision's tier-2 criterion is stated in terms of `f_person`, per the original `sr01_rate_isbd_fields.py` implementation (§1 Context table: 4,613 records). D-04's later tightening (`f_resp_person` replacing `f_person`, via FLERT) applies to tier-2 membership too, not just to which spans get a PERSON label — `build_silver_spans.py` recomputes tier-2 from raw flags using `f_resp_person`, since some `f_person=1` records are corporate bodies or editors, not individual-author SoRs, and shouldn't count as tier-2's "richest multi-field training data" under ADR-04's original rationale either. Recomputed tier-2 = **2,384 records** (down from 4,613) — see `A2-silver-spans-script-plan.md` §2.4 for the exact substitution and the full derivation.

**In short**: 4,613 is "how many records have a slash-marked responsibility statement" (stale formula, still what's on disk). 2,384 is "how many of those slash-marked statements are confirmed to actually name a person" (the corrected formula, applying a decision that was made but never propagated upstream).

**Notes:** [sr03_silver-label-fp-review.md](sr03_silver-label-fp-review.md) · [sr05_trailing-period-noise.md](sr05_trailing-period-noise.md) · [A2-silver-spans-script-plan.md](../../notes/fiz/bibner-werk/A2-silver-spans-script-plan.md)

---

### D-04 — TRANSLATOR and EDITOR dropped as silver label targets

**Decision:** `TRANSLATOR` and `EDITOR` are not viable silver label types. `f_person` sub-classified as: `f_resp_person` (true author SoR), `f_resp_org` (corporate body), `f_resp_family` (family), `f_resp_editor` (editor role), `f_resp_other` (non-SoR).

**Why (SR-04):** 0 true translators found in 100-record sample. EDITOR detection: 0 F1. Only 35% of `f_person` records are true author SoRs; 41% are non-SoR false positives, 19% corporate bodies, 5% editors. The ISBD/RDA agent model requires sub-classification, not a flat TRANSLATOR label.

**Implementation update (2026-07-06, A2/`build_silver_spans.py`):** the original keyword-list implementation of this sub-classification (`RESP_ORG_RE` matching institutional keywords, falling back to `f_resp_person` by default) measured only 41% PERSON precision against the same SR-04 100-record sample cited above — open-vocabulary person/institution name detection has no fixed keyword list to match against, and the "safe default" masked most non-SoR and corporate false positives as PERSON. Replaced with `flair/ner-german-large` (FLERT — Schweter & Akbik, 2020, document-level contextual embeddings over XLM-R) deciding PERSON vs. ORG directly from its `PER`/`ORG` tags; `f_resp_editor`/`f_resp_family` keyword regexes are unchanged (role qualifiers, not NER categories) and are checked *before* the FLERT tags, since an editor's own name still gets tagged `PER`. Re-verified against the SR-04 sample via `scripts/py/verify_resp_classifier.py`.

**Notes:** [sr04_translator-person-disambiguation.md](sr04_translator-person-disambiguation.md) · [A2-silver-spans-script-plan.md](../../notes/fiz/bibner-werk/A2-silver-spans-script-plan.md)

---

### D-05 — No Latin stratum; Early Modern German is the primary historical challenge

**Decision:** Latin records are not stratified or evaluated separately. True Latin prevalence is ~0.5% — too rare to justify a stratum. No Latin NER capability required for Phase 1. EARLY_MODERN_DE (1500–1750) is the primary historical challenge.

**Why (SR-06):** Latin heuristic has 83% FP rate — standard German Protestant/academic vocabulary (`Anno`, `Christi`, `Doctor`) triggers false positives. EARLY_MODERN_DE heuristic achieves F1 = 0.95 after two rule fixes.

**Notes:** [sr06_historical-scope.md](sr06_historical-scope.md)

---

### D-06 — FRBR scope: Work labels in Phase 1, Expression labels in Phase 2

**Decision:** Phase 1 evaluation covers `TITLE`, `OTHER_TITLE`, `PERSON` only. Phase 2 adds `TRANSLATOR`, `PARALLEL_TITLE`, `MEDIUM`. Manifestation labels (`PUBLISHER`, `PLACE`, `YEAR`, `EDITION`, `SERIES`, `VOLUME`) deferred.

**Why (SR-07):** DDB objects are Manifestation-level. Work-level labels are the minimum viable set for GND linking quality metrics. Expression labels require additional heuristics and are not needed to establish the paper's primary contribution. Annotating Phase 2 labels in the same pass avoids re-annotation.

**Notes:** [sr08_label-design-rationale.md](sr08_label-design-rationale.md) · [sr08_gold-set-composition.md](sr08_gold-set-composition.md)

---

### D-07 — NER model path: NuNER Zero → LLM labeling → fine-tune xlm-roberta-base

**Decision:** Evaluate NuNER Zero zero-shot first (SR-09). If TITLE F1 meets threshold: deploy zero-shot. If not: generate 4k–5k LLM-labeled records (SR-11) and fine-tune `xlm-roberta-base` on silver + LLM-labeled set.

**Why:** NuNER Zero is the strongest available zero-shot NER model for this task size and language profile (see [ref_gliner-nunerzero-comparison.md](ref_gliner-nunerzero-comparison.md)). Fine-tuning on silver labels alone risks propagating ISBD-induced label noise; LLM annotation of tier-0 records provides coverage where silver labels are absent.

**Notes:** [ref_gliner-nunerzero-comparison.md](ref_gliner-nunerzero-comparison.md) · [sr11_labeling-strategy.md](sr11_labeling-strategy.md)

---

### D-08 — Gold set: 395 records, ±10 pp CI, TITLE-driven allocation

**Decision:** Gold set = 395 records, stratified by era × tier × dc_type. CI target: ±10 pp on TITLE F1 per era (requires ~49–81 records per stratum; current ~100 per era is sufficient). PERSON evaluated as secondary with wide CI (±15–20 pp); reported as indicative only.

**Why (SR-08):** ±5 pp CI on TITLE would require ~1,054 records; ±5 pp CI on PERSON is structurally impractical (requires 3,700–6,500 records due to low prevalence). Time constraints make ±10 pp the right tradeoff. PERSON extraction is secondary because dc:creator/contributor is absent in 67% of records overall but person names appear in titles only for pre-1700 (8.7%) and 1700–1800 (5.0%) — not for modern (0.2%) or 19th-c (0.6%) records.

**F1 targets:**

| Era | TITLE | PERSON |
|---|---|---|
| Modern | ≥ 0.85 | — |
| 19th-c | ≥ 0.80 | — |
| 1700–1800 | ≥ 0.75 | ≥ 0.70 |
| Pre-1700 | ≥ 0.70 | ≥ 0.70 |

Targets are grounded in benchmark ceilings (OntoNotes WORK_OF_ART: 0.55–0.72; HIPE-2022 historical German: 0.60–0.78), not assumed intervention rates.

**Notes:** [sr08_evaluation-design.md](sr08_evaluation-design.md) · [sr08_gold-set-composition.md](sr08_gold-set-composition.md)

---

### D-09 — Tier allocation: oversample tier-0, minimise tier-2

**Decision:** Gold set allocation weights tier-0 at 79% (target), up from 45% (current). Tier-2 reduced to spot-check only (15 records). Pre-1700 and 1700–1800 oversampled within tier-0.

**Why (SR-08):** Tier-0 is the hardest inference path (no ISBD signals), the most prevalent in corpus (92.4%), and the stratum where NER matters most. Tier-2 records inflate F1 without testing difficult cases. Pre-1700 tier-0 is the primary failure risk: author-before-title structure (annotation-observed pattern, not independently sourced — Gaskell 1972, *A New Introduction to Bibliography*, was checked against this specific claim and does not document an author-before-title ordering convention; see `paper/jcdl-2026/30-corpus.tex` CVVD history, `notes/fiz/bibner-werk/B12-corpus-cvvd-findings.md` F3) + historical orthography + dc:creator absent in 67%.

**Notes:** [sr08_evaluation-design.md](sr08_evaluation-design.md) §8 · [sr08_gold-set-composition.md](sr08_gold-set-composition.md)

---

### D-10 — Evaluation metric: per-label point-estimate F1, exact span match; bootstrap deferred

**Decision:** Report micro P/R/F1 per label (TITLE, OTHER_TITLE, PERSON) per era as point estimates, following the HIPE-2022 reporting convention. Exact span match throughout (both character boundaries and label must match). No macro or micro averages across labels. Sample sizes reported alongside every F1 figure. Bootstrap CI deferred to future work.

**Why — exact span match:** Gives a clean, unambiguous signal consistent with HIPE-2022 strict regime and CoNLL benchmarks. Fuzzy (overlapping boundary) match would inflate scores without indicating whether the extraction is usable for GND linking. Risk: boundary disagreement between annotators on long pre-1700 titles (TITLE vs. trailing author credential) — should be monitored during annotation and flagged in the paper.

**Why — no macro/micro averages:**
- *Micro average* pools all TP/FP/FN before computing F1 — dominated by TITLE (~100% prevalence), making OTHER_TITLE and PERSON invisible.
- *Macro average* weights all labels equally regardless of prevalence — PERSON (0.2%–8.7% by era) would drag down the headline figure, misrepresenting pipeline utility for the primary goal of GND linking.
- Per-label reporting is transparent: it shows where the pipeline succeeds (modern TITLE) and where it struggles (pre-1700 PERSON).

**Why — point estimates over bootstrap:** The contribution is the pipeline, not a model comparison. Bootstrap CI is necessary when asking whether two systems differ significantly — that is not the question here. Per-stratum n (~100) would also produce wide CIs (±10–15 pp) that are more misleading than informative at this sample size.

**Notes:** [sr08_evaluation-design.md](sr08_evaluation-design.md) §4

---

### D-11 — IAA metric: Krippendorff's α with nominal distance function

**Decision:** Inter-coder agreement is measured using Krippendorff's α (Krippendorff 1980, 2004a) with the nominal distance function ($d_{ab} = 0$ if $a = b$, else $d_{ab} = 1$), computed on the pre-adjudication annotations (raw labels before manual resolution).

**Why:** Four reasons, following Artstein and Poesio (2008):
1. α is defined for any number of coders — it accommodates all three annotators (Annotator A, Annotator C, Annotator B) directly, unlike Cohen's κ which is restricted to two. "The coefficient α... is an extremely versatile agreement coefficient... It applies to multiple coders, and it allows for different magnitudes of disagreement" (§2.6.1, p. 564).
2. The nominal distance function is the correct and unambiguous choice for mutually exclusive entity labels (TITLE, OTHER_TITLE, PERSON): "the function for nominal categories... considers all distinct labels equally distant from one another: $d_{ab}=0$ if $a=b$; $1$ if $a \neq b$" (§2.6.1, pp. 566–567).
3. As a single-distribution coefficient, α measures the reliability of the annotation *procedure* rather than the behaviour of individual annotators — the appropriate target for a corpus released as a reusable resource: "The most appropriate measures of reliability for this purpose are therefore single-distribution coefficients like π and α, which generalize over the individual coders and exclude marginal disagreements from the expected agreement" (§3.1, p. 571); "We agree with the view that K and α are more appropriate, as they abstract away from the bias of specific coders" (§5.2, p. 590).
4. Near-complete coverage — missing data is negligible: α's formula uses $n_{ik}$ (number of coders assigning item $i$ to category $k$), which naturally handles items annotated by fewer than $c$ coders. With 1–3 missing judgments out of 395 items, the effect on the coefficient is negligible.

**Alternatives considered:**

| Metric | Reason rejected |
|---|---|
| Cohen's κ | Defined for 2 coders only |
| Fleiss's multi-π | Valid, but α is nearly identical for nominal data and preferred by A&P for reliability studies |
| Percentage agreement | Does not correct for chance; biased by category distribution (§2.3, p. 558) |

**Coefficient comparison (Artstein and Poesio 2008, §2.4–§2.6, §3.1):**

| | S (Bennett 1954) | π (Scott 1955) | κ (Cohen 1960) | α (Krippendorff 1980) |
|---|---|---|---|---|
| Chance model | Uniform: all labels equally likely | Pooled: single distribution over all coders | Per-coder marginals: each coder's own distribution | Pooled with unbiased small-sample correction |
| $c > 2$ coders | Via generalization | Fleiss multi-π (1971) | Davies & Fleiss (1982) | **Native** |
| Annotator bias | Ignores | Ignores | **Sensitive to** | Ignores |
| Weighted distances | No | No | κ_w (Cohen 1968) | **Yes** |
| Missing values | No | No | No | **Yes** |
| CL convention | Rare | De facto (called "kappa") | Common | Less common |

**Why not κ?** κ incorporates per-coder marginals into $A_e$, discounting disagreements that arise from different coder distributions. A&P note that the common claim that κ "rewards" annotators for distributional disagreement is "unwarranted" (§3.1, p. 571) — what κ actually does is absorb those disagreements into expected agreement rather than counting them against the procedure. For reliability assessment (the typical CL goal: would a different annotator trio following the same guidelines produce the same result?), all disagreements should count — making single-distribution coefficients like π or α the appropriate choice. κ remains informative for validity (trustworthiness of a specific annotator pair), but that is a different question. α is preferred over π because it uses an unbiased estimator of $D_e$ and is numerically equivalent to multi-π as $\mathbf{i}$ or $\mathbf{c}$ grows.

**Caveat:** α with a non-nominal (weighted) distance metric can be difficult to interpret because results vary substantially with the choice of weights (§2.6.1, p. 567). Does not apply here — the nominal distance function is unambiguous for NER labels. Bootstrap confidence intervals are deferred (same point-estimate-over-CI reasoning as D-10); if needed later, follow Krippendorff (2004a, §11.4.2) and Hayes and Krippendorff (2007).

**Data (export-20260623-1110):**

| Annotator | Reviewed / 395 |
|-----------|---------------|
| Annotator A    | 394 |
| Annotator C     | 395 |
| Annotator B     | 392 |

Co-reviewed pair counts: Annotator A/Annotator C 394, Annotator A/Annotator B 391, Annotator C/Annotator B 392.
Negative judgments (Annotator C: 130, Annotator B: 98 "no entity" records) are included as valid label-$O$ annotations.
α is computed on pre-adjudication annotations; reconciliation produces `gold_reconciled.jsonl`.

α values (ALL + per-label): `data/iaa/sr08_iaa_export-20260623-1110.csv`
Pairwise α values (ALL + per-label, all pairs): `data/iaa/sr08_iaa_pairwise_export-20260623-1110.csv`
Raw label assignments backing both: `data/iaa/sr08_iaa_counts_export-20260623-1110.csv`

**Notes:** [inter-annotator-agreement.md](inter-annotator-agreement.md) · [iaa-score-walkthrough.md](../../notes/fiz/bibner-werk/iaa-score-walkthrough.md) · Krippendorff (1980, 2004a); Artstein and Poesio (2008) §2.3, §2.6.1, §3.1, §5.2

---

### D-12 — Span agreement metric: exact-match rate + mean Jaccard, not chance-corrected

**Decision:** Given two annotators agree a label exists on a document (D-11's α measures that), report whether they agree on its *character span* as two plain statistics, computed pairwise, restricted to (pair, doc, label) triples where both annotators marked the label present: exact-match rate (identical character-position sets) and mean Jaccard overlap ($|A \cap B| / |A \cup B|$). Not routed through Krippendorff's α. Two annotators can both mark "TITLE present" while marking different spans (e.g., one includes a trailing person's name in the TITLE span, the other doesn't) — D-11's α cannot see this, since it only measures whether the label was assigned at all.

**Why:** α's expected-disagreement term $D_e$ pools *all* judgments in the corpus and averages distance over every possible pair, which assumes judgments are drawn from one shared, comparable universe. Character offsets in BibNER-Werk are document-local — a span at characters 10–20 in one document has no relationship to characters 10–20 in another — so pooling raw span positions the way $D_e$ requires would measure incidental differences in document length and span position, not real disagreement. Krippendorff's α_U (unitizing alpha, §4.3.2) sidesteps this for a single continuous text stream, which doesn't match BibNER-Werk's 395 independent short catalog strings; Artstein and Poesio (2008, p. 582) note α_U itself has "never been tested in CL," with known unintuitive behavior when one coder's span overlaps two of another's. Exact-match rate and mean Jaccard are the standard, citable way to report span agreement in NER literature instead — simpler than F1 (no precision/recall asymmetry to justify, since both annotators are peers, not gold vs. system) and bounded in [0,1] like α. A&P §10.2–10.3 already recommended reporting overlap alongside α as a sanity check ("reviewers familiar with NER will expect span-level agreement numbers"). A&P §4.4.3 independently demonstrates why a chance-corrected span coefficient is fragile in practice: computing α on the same discourse-antecedent annotations with different (but equally defensible) distance metrics yielded values ranging from 0.122 to 0.998 — the choice of metric, not the underlying agreement, drove the result.

**Alternatives considered:**

| Approach | Reason rejected |
|---|---|
| Extend Krippendorff's α with a span distance metric (α_U-style) | $D_e$ pooling is not meaningful across document-local character offsets; no CL precedent (A&P, p. 582) |
| Gamma γ (Mathet et al. 2015) | Designed for exactly this problem, but rarely used, no standard implementation, hard to cite |
| Token-level κ (BIO tagging) | Inflates agreement via the prevalence problem — most tokens are O (§3.2, A&P) |

**Caveat:** Not chance-corrected — doesn't distinguish genuine boundary agreement from what overlap would occur by chance, which was the whole point of using α for existence agreement in D-11. Accepted tradeoff: no chance-corrected alternative is both meaningful for document-local spans and citable.

**Results (export-20260623-1110, pairwise, ALL scope pooled across labels):**

| Pair | n_existence_agreed | exact% | mean Jaccard |
|---|---:|---:|---:|
| Annotator A/Annotator C | 373 | 38.9% | 0.726 |
| Annotator A/Annotator B | 388 | 43.8% | 0.737 |
| Annotator C/Annotator B | 354 | 50.3% | 0.792 |

Annotator C/Annotator B has both the best existence-α (−0.018, D-11) and the best span agreement (50.3% exact, 0.792 Jaccard) — consistent at the top end, despite the two metrics being computed from disjoint information (whether a label was assigned vs. where its boundaries fall). The weakest pair differs by metric, though: Annotator A/Annotator B has the lowest existence-α (−0.201) while Annotator A/Annotator C has the lowest span agreement (38.9% exact, 0.726 Jaccard). This points to two distinct failure modes rather than one annotator being uniformly weaker: Annotator A and Annotator B disagree most on *whether* a label applies at all, while Annotator A and Annotator C disagree most on *where* it applies once they've agreed it's present.

**Data:** `data/iaa/sr08_iaa_span_pairwise_export-20260623-1110.csv` (ALL + per-label, all pairs)

**Notes:** [inter-annotator-agreement.md](inter-annotator-agreement.md) §1.7 · [iaa-score-walkthrough.md](../../notes/fiz/bibner-werk/iaa-score-walkthrough.md) §10 · [adr-02-plan.md](../../notes/fiz/bibner-werk/adr-02-plan.md) (design-rationale record) · Artstein and Poesio (2008) §3.2, §4.3.2, §4.4.3

---

### D-13 — Series statements excluded from Round 1 adjudication (Manifestation-level, per D-06)

**Decision:** Bracketed/parenthetical series statements (e.g. `(Sitzungsberichte / Österreichische Akademie der Wissenschaften...; 477)`) encountered during Round 1 gold adjudication are left unlabeled — not `TITLE`, not `OTHER_TITLE` — regardless of how many annotators happened to tag one.

**Why:** A series statement is ISBD Area 6, deliberately separate from Area 1 (title/statement of responsibility); RDA models series membership as a Manifestation-level relationship (RDA 2.12) between this publication and the series, not an attribute of the Work or Expression. D-06 already defers `SERIES` to a later phase alongside `PUBLISHER`, `PLACE`, `YEAR`, `EDITION`, `VOLUME` — this decision applies that existing scope boundary consistently during adjudication, rather than letting per-document annotator agreement (which varied across cases) override it.

**Examples (Round 1, export-20260623-1110):**
- doc 11: `(Sitzungsberichte / Österreichische Akademie der Wissenschaften, Philosophisch-Historische Klasse ; 477)` and `(Österreichische Akademie der Wissenschaften <Wien> / Veröffentlichungen der Iranischen Kommission ; 20)` — 0 of 3 annotators agreed on a labeling
- doc 29: `(Quellensammlung für fränkische Geschichte, herausgegeben von dem historischen Vereine zu Bamberg; 3. Band)` — 1 annotator (Annotator A) labeled it OTHER_TITLE
- doc 100: `(Beiträge zur Heimatkunde des Landkreises Augsburg ; 11)` — both non-O annotators (Annotator A, Annotator B) independently agreed on the same OTHER_TITLE span

**Considered and rejected (for now):** tagging these now as a new `SERIES` label, front-loading Manifestation-level data collection the way Phase 2 labels (`TRANSLATOR`, `PARALLEL_TITLE`, `MEDIUM`) were front-loaded per D-06. Rejected because D-06's "annotate in the same pass" reasoning was scoped to Phase 2 (Expression-level) labels only — Manifestation-level labels were deferred outright, not fast-tracked alongside them. Introducing `SERIES` mid-adjudication would also produce an inconsistent gold set: tier-1/tier-2 auto-accepted documents were never reviewed for series statements, since no annotator was ever asked to look for them, so only the tier-3 subset would end up with (partial) `SERIES` coverage. If Manifestation-level annotation is scheduled later, it should be a dedicated pass over all 395 records with its own guide additions, not a retrofit onto the tier-3 subset alone.

**Caveat — a title-of-a-multi-volume-work is not a series statement:** doc 135 (`Lebens-Beschreibungen Der Biblischen Scribenten. [11], Georgii Serpilii, S. Salomo, ...`; `dc_type: Band`) was initially misclassified under this decision and excluded, then corrected. The distinguishing test: a series statement names a generic publishing vehicle that could host any number of otherwise-unrelated works (docs 11, 29, 100's series names would fit equally well on a completely different book on a different topic). A multi-volume work's own collective title does not pass that test — `Lebens-Beschreibungen Der Biblischen Scribenten` refers to nothing except this one encyclopedic project, and "volume 11" is thematically part of *that specific work*, not a work placed inside an unrelated series. The former is `SERIES` (D-13 applies, exclude); the latter is the Work's own `TITLE` (D-13 does not apply), with the volume number excluded the same way other volume/part designators are excluded elsewhere in this corpus.

**Notes:** [adjudication-method.md](adjudication-method.md) (resolution tiers, §2; pipeline, §3) · relates to D-06

---

## 3. Open decisions

| Decision | Blocked on | Notes |
|---|---|---|
| NuNER Zero viability (SR-09) | SR-08 annotation complete | Pass/fail gate for zero-shot vs. fine-tune path |
| LLM annotation prompt design (SR-11) | SR-08 50-record seed | Early Modern German author-before-title prompt spec |
| Silver tier field weighting (SR-12) | SR-03 extension, SR-08 | Replace binary threshold with precision-weighted score |
| Revised allocation re-sample | SR-08 annotation complete | Re-run `sr08_sample_gold.py` with tier-0 boost |
