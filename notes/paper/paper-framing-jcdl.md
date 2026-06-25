---
title: JCDL 2026 Paper Framing
up: []
related: []
created: 2026-05-20T00:00:00
updates:
  - 2026-06-24
in: bibner-werk
tags:
  - paper
  - jcdl
  - framing
context: Primary submission: Framing 2 (NER). TPDL cancelled — dual-submission constraints removed.
---

# JCDL 2026 — Paper Framing

## 0. Deadlines and contribution types

| Milestone | Date |
|---|---|
| Full and short paper submission | **30 June 2026** |
| Notification | 20 August 2026 |
| Camera-ready | 10 September 2026 |

**Contribution types**:
- Full paper: 10 pp + refs → ACM sigconf, double-blind
- Short paper: 4 pp + refs → ACM sigconf, double-blind

**Venue fit**: JCDL is ACM/IEEE, with a broader CS audience than TPDL. Reviewers expect empirical evaluation — run statistics alone do not constitute an evaluation at this venue. Strong fits from this work: "Metadata standards, linked data, semantic annotation, and interoperability in digital collections"; "NLP, information extraction, and semantic search"; "Knowledge graphs, semantic web, ontologies"; "Multilingual and multicultural digital libraries."

**Dual-submission check (mandatory before submitting)**:
- babel-ddb/SemDH 2026: submitted; no overlap in contribution if JCDL framings are evaluated, not design-focused
- GeMeA/ISWC 2026 Resource Track: if submitted, JCDL must not describe the same resource contribution; both framings below are distinct (methodology evaluation and NLP, respectively)
- TPDL 2026: submission cancelled — all TPDL-conditioned dual-submission constraints no longer apply

---

## 1. Framing 1 — Sector-aware corpus profiling for cultural heritage KG construction

**Working title**: *Profiles Before Predicates: A Corpus-Driven Pipeline for Aligning Heterogeneous Cultural Heritage Metadata to Domain Ontologies*

**One-sentence hook**: Because seven institutional sectors use the same EDM property (`dc:type`) with incompatible semantics, schema-level alignment systematically fails; a sector-stratified corpus profiling approach recovers per-sector conventions and maps them to WEMI-typed domain ontology classes, producing 44.4M triples from 115K records with quantified coverage — and scales to 65M objects in the full DDB pipeline.

### 1.1 Core argument

The DDB-EDM profile is a structural wrapper, not a semantic contract. `dc:type` carries controlled-vocabulary terms in libraries, free-text in museums, and structural hierarchy identifiers in archives. No ontology-to-ontology alignment tool can recover these conventions from the schema alone. The methodology contribution is: **profile corpus before modelling, not after**.

The three-layer dispatch architecture (Layer 1: property alignment; Layer 2: htype → RiC-O/DoCO class; Layer 3: dc:type × sector × mediatype → WEMI-slot class) operationalises this and produces 44.4M triples from 115,432 goethe-faust records with quantified coverage at each dispatch level. GeMeA full-corpus ingest statistics (65M objects) provide scale-up validation.

**Differentiation from submitted/planned papers**:
- babel-ddb/SemDH: argues *what* MOCHO is and why the hub ontology design is correct; this paper argues *how* to apply it at scale and whether it works (evaluation, not design)
- GeMeA/ISWC: resource description with FAIR criteria; this paper describes the pipeline methodology and evaluates alignment quality — different contribution type

### 1.2 Source material to repackage

| Source | What to take |
|---|---|
| `goethe-faust/notes/transform-writeup.md` §1–4 | Paper §§2–4 (layers diagram, htype table, dc:type dispatch, run stats) — nearly verbatim |
| `output/alignment_ddbedm_mocho.csv` | Coverage table (field × match\_method × WEMI level) |
| `output/transform_stats.json` | Per-layer hit rates, fallback rates, triple counts |
| babel-ddb §2–3 | Background motivation, hub ontology pattern, WEMI rationale |
| `mocho/notes/alignment-ddbedm-mocho-spec.md` | Known gaps table, success criteria |
| GeMeA full-corpus ingest stats (if available) | §5 scale-up: object counts, triple counts, provider distribution |

### 1.3 JCDL topic fit

- **Metadata standards, linked data, semantic annotation, and interoperability in digital collections** (primary)
- **Knowledge graphs, semantic web, ontologies, and graph-based knowledge representation** (secondary)
- **Data mining, data analytics, and visualization of library collections** (data profiling angle)
- **Domain-specific applications of digital libraries** (DDB/German CH domain)

### 1.4 Contribution type recommendation

**Full paper (10 pp)** if GeMeA full-corpus stats are available and a retrieval evaluation is included (even a lightweight 20–30 known-answer query comparison). JCDL reviewers expect evaluation; pipeline statistics without a retrieval or quality experiment will not pass review.

**Short paper (4 pp)** as a fallback if GeMeA stats are unavailable and the retrieval experiment cannot be run — present as a methodology paper with goethe-faust pilot results only; accept a lower acceptance ceiling.

### 1.5 Evaluation needed for JCDL (absent from TPDL version)

- **WEMI-type precision by stratum**: extend the heuristic eval figures from T12 (babel-ddb camera-ready: `dc:type="Buch"` 81.6%, `"Monografie"` 99.4%) to all `dc:type` values across all sector × mediatype strata; report weighted precision over the full goethe-faust corpus
- **Retrieval evaluation (recommended)**: 20–30 known-answer queries comparing entity-centric SPARQL retrieval (agent URIs, place URIs, work URIs from MOCHO-typed triples) to DDB native full-text search; metric = precision@k; use the Goethe-Faust corpus where ground truth is known

### 1.6 Pros

- Content ~85% exists in notes; `transform-writeup.md` §1–4 maps 1:1 to paper §§2–4
- Run stats are saved and reproducible
- babel-ddb provides ready-made related work; no new literature review needed
- Scale story (115K pilot → 65M full corpus) is compelling for JCDL
- No new annotation required — evaluation is automated (SPARQL queries + precision computation)
- Strong JCDL "Content, Collections" track fit; RDA, RiC-O, EDM are increasingly in scope at DL venues

### 1.7 Cons

- JCDL reviewers expect an IR evaluation, not just pipeline statistics; retrieval experiment must be designed and run before submission
- GeMeA full-corpus ingest statistics depend on the GeMeA pipeline completing in time
- The three-layer dispatch is complex; reviewers unfamiliar with EDM need background that competes with the 10-page budget

### 1.8 Task list

- [ ] Write abstract (≤250 words) — basis: `transform-writeup.md` §Problem + §Run results
- [ ] Draft §1 Introduction — profile-first vs. schema-first framing; 1 p
- [ ] Draft §2 Background — DDB-EDM sectors, WEMI, MOCHO hub ontology; 1.5 p; reuse babel-ddb §§1–2
- [ ] Draft §3 Methodology — three-layer dispatch with Mermaid diagram; 2 p; reuse `transform-writeup.md`
- [ ] Draft §4 Results — run stats table + per-layer coverage table; 1.5 p; reuse saved CSVs
- [ ] Draft §5 Scale-up — GeMeA full-corpus stats (if available); 1 p
- [ ] Draft §6 Retrieval Evaluation — SPARQL vs. DDB keyword search; 1.5 p (new work)
- [ ] Draft §7 Discussion — known gaps, scope limits, reproducibility; 1 p
- [ ] Draft §8 Related Work — EDM alignment papers, BIBFRAME–EDM, DDB2FaBiO; 0.5 p
- [ ] Draft §9 Conclusion; 0.5 p
- [ ] Convert to ACM sigconf LaTeX
- [ ] Submit via EasyChair

**Estimated writing effort**: 9–12 days (6–7 writing + reuse; 2–3 days retrieval eval; 1 day formatting).

---

## 2. Framing 2 — NER for historical German bibliographic title strings

**Title**: *BibNER-Werk: Bibliographic NER for Historical German Catalogs*

**Name rationale**: BibNER-Werk = Bibliographic NER for FRBR Work-level entities. *Werk* (German for "work") denotes the FRBR/LRM Work level; Phase 1 evaluates Work-level entities (TITLE, OTHER_TITLE, PERSON); E and M levels are future phases. (Formerly WEM — Work–Expression–Manifestation — renamed to avoid overclaiming scope.)

**One-sentence hook**: Standard NER models fail on ISBD-formatted bibliographic catalog strings across three independent dimensions — wrong label set, wrong domain, wrong register — and 92.4% of 9.2M German-language DDB records lack ISBD structural signals; BibNER-Werk delivers an ISBD-guided silver labeling pipeline, a 395-record stratified gold evaluation, and a four-system comparison establishing the zero-shot and fine-tuned performance ceiling for Work-level entity extraction from historical German library catalogs.

### 2.1 Core argument

Bibliographic catalog strings are a distinct NLP register: structured by ISBD punctuation, not prose syntax, and historically variable in orthography (Early Modern German). General-purpose NER fails on three dimensions: wrong label set (CoNLL PER/ORG/LOC vs. TITLE/OTHER\_TITLE/PERSON), wrong domain (newswire vs. GLAM), wrong register (prose vs. ISBD formulaic). The contribution is: (a) an ISBD-guided silver labeling strategy covering the 28.4% of records with structural signals, tiered by confidence; (b) a zero-shot NuNER Zero evaluation on a 395-record stratified gold set; (c) a characterization of where zero-shot fails (historical strata) as motivation for fine-tuning.

This is a **methodology + dataset + evaluation** paper. The claim is: *here is how to approach this task, why it is hard, and what the zero-shot ceiling is* — with F1 per stratum as the evidence.

**Differentiation from submitted/planned papers**:
- babel-ddb/SemDH: ontology design, no NLP; fully distinct
- GeMeA/ISWC: resource description, NER is one pipeline component not evaluated; fully distinct

### 2.2 Source material to repackage

| Source | What to take |
|---|---|
| `notes/ner/spiel_ner.md` §0–§8 | Paper §§1–6; chain-of-thought maps directly to section structure |
| `sr10_de-titles-distribution.md` + `fig_title_lengths.png` | §2 corpus statistics |
| `sr01_isbd-field-rating.md`, `sr01_isbd-applicability.md`, `sr01_isbd-title-analysis.md` | §3 ISBD background + silver labeling applicability |
| `silver-dataset-pipeline.md` | §3 silver labeling methodology |
| `sr08_gold-set-composition.md`, `sr08_evaluation-design.md`, `sr08_annotation-guide.md` | §4 gold set design and evaluation protocol |
| `sr09_nuner-tier2-sanity.md` | §5 baseline (needs formal eval run, not just sanity check) |
| `sr05_trailing-period-noise.md`, `sr06_historical-scope.md` | §3 register challenges and noise sources |
| `ref_gliner-nunerzero-comparison.md`, `ref_hipe2022-overview.md` | §6 related work |

### 2.3 JCDL topic fit

- **Natural language processing, information extraction, and semantic search** (primary)
- **Multilingual and multicultural digital libraries** (German, historical orthography)
- **Digitization, transcription, and analysis of cultural heritage and multimedia collections** (primary)
- **Digital humanities methods, computational social science** (DH angle)
- **Knowledge discovery, entity-centric retrieval** (downstream use)

### 2.4 Contribution type recommendation

**Full paper (10 pp)** if gold annotation is completed and formal F1 evaluation (NuNER Zero + at least one fine-tuned baseline) is run before submission. The silver dataset release and reproducibility statement strengthen the contribution further.

**Short paper (4 pp)** as a fallback if gold annotation is incomplete — present silver pipeline + NuNER Zero sanity baseline as a methodology paper; accept a significantly lower acceptance ceiling and risk reviewers requesting a full evaluation.

### 2.5 Critical path

1. **Gold annotation** — human annotators: frankp 393/395 (primary), maria 131/395, gundi 82/395; user annotation pass due 17 June; `ann.jsonl` = Claude pre-annotation (183/395, seeding only); reconcile all → `gold_reconciled.jsonl`
2. **IAA computation** — human-human pairs only: frankp∩maria (131 records), frankp∩gundi (81 records); report range in §4; computable now
3. **Early eval on frankp's 393 records** — NuNER Zero + gliner_multi zero-shot + few-shot; results confirmed on reconciled set after 17 June
4. **Build silver spans** (`build_silver_spans.py`, Stage 5) — prerequisite for fine-tuning
5. **Set up fine-tuning script** (`train_gliner_silver.py`) — ready to fire on 17 June
6. **`gliner_multi-v2.1` fine-tuned on silver labels** (1–2 days; tier-2 + 50K stratified tier-1; exclude 395 gold IDs; use label descriptions)
7. **Qwen2.5-14B zero-shot + few-shot** *(optional, time-permitting)* — not on critical path; run only after fine-tuned gliner_multi results are in
8. **Silver dataset release** (CC BY 4.0; confirm DDB data license)

**Hardware**: A100 80GB PCIe (CUDA 12.4, primary) + 2× Tesla V100S-PCIe 32GB (CUDA 11.6, fallback). Use A100 for all LLM inference and gliner_multi fine-tuning: bf16, flash attention 2, and vLLM are all available. Qwen2.5-14B (bf16, ~28GB) and Qwen2.5-32B (bf16, ~64GB) both fit on the A100 with headroom; use vLLM for batch inference. Qwen2.5-72B in 4-bit (~36GB) is feasible if a larger scaling point is wanted.

### 2.6 Pros

- `spiel_ner.md` §0–§8 is a near-complete paper skeleton; chain-of-thought outline directly produces §§1–6
- All corpus statistics are already computed and saved as CSVs; `fig_title_lengths.png` is publication-ready
- The "why off-the-shelf NER fails" argument (§3 of spiel\_ner) is original and the JCDL historical/multilingual DL audience will find it directly relevant
- Historical German bibliographic records are a known pain point in the LAM community; scale (4.48M records, 71.6% NER fallback) is compelling
- No architectural novelty required: methodology + dataset + evaluation is the contribution type
- The convergence of pilot (29% ISBD coverage) and full corpus (28.4%) validates generalizability within the paper

### 2.7 Cons

- Formal eval code does not yet exist (`sr09` is a sanity check, not a paper-ready evaluation script)
- GLiNER2 (Zaratiana et al., 2025) is English-only; use `urchade/gliner_multi-v2.1` (multilingual GLiNER v1 fine-tune) for both zero-shot and fine-tuned baselines — these are distinct from the GLiNER2 system paper
- LLM structured output parsing is fragile; need a robust parser for JSON span extraction from Qwen2.5-14B output; budget half a day for prompt iteration
- Historical German orthography variation is a known challenge; discussion of tokenization choices may consume page budget
- GeMeA is the downstream consumer of NER output, but GeMeA is not built — cannot claim end-to-end integration; describe as intended downstream use

### 2.8 Task list

- [ ] Complete gold annotation (395 records) — started 20 May, due 28 May
- [ ] Run formal NuNER Zero evaluation on gold set; save per-stratum F1 + CI to `data/processed/`
- [ ] Run `urchade/gliner_multi-v2.1` zero-shot evaluation on gold set; save per-stratum F1 + CI to `data/processed/`
- [ ] Run Qwen2.5-14B zero-shot on gold set (fp16, single V100S; HF Transformers; structured JSON prompt); save per-stratum F1 + CI to `data/processed/`
- [ ] Run Qwen2.5-14B few-shot (k=5, Tier 1 silver examples) on gold set; save per-stratum F1 + CI to `data/processed/`
- [ ] Fine-tune `urchade/gliner_multi-v2.1` on silver labels; evaluate on gold set; save per-stratum F1 + CI
- [ ] Write abstract (≤250 words) — basis: `spiel_ner.md` chain-of-thought + §1 Scale
- [ ] Draft §1 Introduction — 71.6% fallback framing; 1 p
- [ ] Draft §2 Corpus and Task — DDB, German-language filter, ISBD structure, label set; 1.5 p
- [ ] Draft §3 Challenges and Silver Labeling — register failures + ISBD signal extraction + tiered confidence; 2 p
- [ ] Draft §4 Evaluation Design — gold set (395 records, stratified by era + silver tier), exact span match, CI constraints; 1 p
- [ ] Draft §5 Results — NuNER Zero + fine-tuned baseline F1 per stratum; analysis of failures; 2 p
- [ ] Draft §6 Related Work — HIPE-2022, historical NER, bibliographic NLP; 0.5 p
- [ ] Draft §7 Conclusion and Future Work; 0.5 p
- [ ] Confirm silver dataset and gold set availability/license (CC BY 4.0 via DDB)
- [ ] Add reproducibility statement (anonymized dataset + eval scripts via anonymous.4open.science)
- [ ] Convert to ACM sigconf LaTeX
- [ ] Submit via EasyChair

**Estimated writing effort**: 10–13 days from 28 May (eval code + NuNER Zero + gliner_multi zero-shot: 2 days; Qwen2.5-14B zero-shot + few-shot + prompt engineering: 1–2 days; gliner_multi fine-tuning: 1–2 days; writing: 5–6 days; formatting: 1 day) → submission-ready by ~10 June; ~20 days of buffer before 30 June deadline.

---

## 3. Comparison

| Criterion | Framing 1 (Harmonization + Eval) | Framing 2 (NER) |
|---|---|---|
| Content readiness | ~85% exists; eval experiment is new | ~80%; gold annotation completing 28 May |
| New code needed | Retrieval eval script (required) | Eval script + gliner_multi fine-tuning |
| Dual-submission risk | None (TPDL cancelled) | None (TPDL cancelled) |
| JCDL reviewer fit | "Metadata standards + KG construction" — strong | "NLP + DL" — strong |
| Evaluation type | WEMI-type precision + retrieval@k | F1 per stratum (NuNER Zero + gliner_multi zero-shot + Qwen2.5-14B zero-shot + Qwen2.5-14B few-shot + gliner_multi fine-tuned) |
| Man-days remaining | 9–12 | ~18–21 total (8 d annotation + 10–13 d post-annotation) |
| Acceptance w/ evaluation | 35–45% | 40–50% |
| Acceptance w/o evaluation | 15–20% (pipeline stats only) | 20–30% (zero-shot only) |
| Key risk | Retrieval eval required; pipeline stats alone insufficient | LLM output parsing fragility; annotation must complete on schedule |

**Recommendation**: **Framing 2 is the primary submission.** Gold annotation due 28 May; five-system comparison (NuNER Zero, gliner_multi zero-shot, Qwen2.5-14B zero-shot, Qwen2.5-14B few-shot, gliner_multi fine-tuned) gives per-stratum F1 results with a 40–50% acceptance ceiling. Submission-ready by ~10 June with ~20 days of buffer. Framing 1 is a viable backup but requires a retrieval experiment to clear the JCDL bar; pipeline statistics alone will not pass review.

---

## 5. Acceptance Assessment #1 — 2026-06-10

**Baseline range: 35–45%** (plan's 40–50% is slightly optimistic)

### 5.1 Strengths

- Three failure dimensions argument (wrong label set, wrong domain, wrong register) is original and crisp; JCDL audience will recognise it
- Scale: 9.21M records; real downstream use in GeMeA
- Methodology + dataset + evaluation is the contribution type JCDL rewards
- Dataset release with human-human IAA strengthens reproducibility
- cc2llm prior work establishes DDB authority and gives reviewers context

### 5.2 Reviewer pushback expected

- **Gold set size (395 records)**: small by NLP standards; stratification + IAA defends it but needs pre-emptive justification in §5
- **Related Work at 0.75 p** (→ bumped to 1 p after this assessment): thin coverage risks "insufficient positioning" comment; Sorbi et al. 2026 differentiation must be explicit
- **3 entity types only**: needs crisp one-sentence answer in paper (FRBR Work identification requires exactly these three; E/M levels are future work)
- **No generalizability beyond DDB/German**: mention Europeana/broader applicability briefly in conclusion

### 5.3 Outcome depends on fine-tuned F1

| Fine-tuned F1 | Likely acceptance range |
|---|---|
| ≥ 0.65 | 42–48% |
| 0.45–0.64 | 35–42% |
| < 0.45 | 28–35% — needs reframing as "task is hard + here's why" |

### 5.4 Action items from this assessment

- [x] Related Work → 1 page (was 0.75)
- [ ] Draft §2 early — check if 1 page is actually enough for Sorbi differentiation
- [ ] Pre-emptive IAA + gold set size justification in §5
- [ ] One-sentence answer on 3-entity-type scope in §1 or §3
- [ ] Conclusion: one sentence on generalizability to other national aggregators

---

## 6. Acceptance Assessment #2 — Devil's Advocate — 2026-06-10

**Revised range after devil's advocate: 30–42%**

The framing has real strengths but seven structural vulnerabilities a hostile reviewer will find.
Each is stated as the attack, then the mitigation.

---

### 6.1 Method is application, not invention

**Attack**: GLiNER fine-tuning on domain-specific data is standard practice. The paper applies
an existing tool to a new domain. By NLP venue standards this is incremental — the novelty is
the domain, not the method.

**Mitigation**: JCDL rewards domain contributions. Reframe throughout as "domain expert
evaluation" — the contribution is the corpus characterisation, label set design, and
silver strategy, not a new model architecture. Never claim methodological novelty for GLiNER itself.

---

### 6.2 System comparison is too narrow

**Attack**: Three of four systems are GLiNER variants. NuNER Zero is included as a strawman
(F1 ≈ 0 already confirmed). There is no BIO tagger (XLM-R fine-tuned, deepset/gbert-large),
no generative baseline (Llama, Qwen), no rule-only baseline. The comparison looks designed
to favour GLiNER.

**Mitigation**: Add at least one BIO tagger as a fifth system — `deepset/gbert-large`
fine-tuned on silver is a natural choice and directly comparable. Without it, reviewers will
ask for it. Qwen2.5-14B (already planned as optional) would also address the generative gap.
If compute is tight, a BIO tagger is cheaper than an LLM and more defensible to add.

---

### 6.3 "Ceiling" claim is unjustified

**Attack**: The paper claims to "establish the zero-shot and fine-tuned performance ceiling."
A ceiling requires exhaustive comparison. Excluding LLMs (Qwen deferred) and BIO taggers
means the claimed ceiling is just the GLiNER family ceiling — a much weaker claim.

**Mitigation**: Replace "ceiling" with "baseline performance characterisation" or
"zero-shot and fine-tuned baselines" throughout. Reserve "ceiling" language only for the
fine-tuned model, where the argument is strongest.

---

### 6.4 Scale claim vs. evaluation scope mismatch

**Attack**: The hook is 9.21M records. But the gold evaluation covers 395 records, and the
silver labeling applies to only 7.56% of the corpus (tier-1 + tier-2). The paper claims
scale but evaluates on a sliver. A reviewer can write: "The interesting 92.44% (tier-0,
no ISBD signal) is entirely out of scope."

**Mitigation**: Address this directly in §3 (Corpus and Task): "We focus on the 7.56% of
records with ISBD structural signal, where silver labeling is tractable. Tier-0 records
(92.44%) require a different strategy and are left for future work." Frame it as a deliberate
scope decision, not a limitation. The tier-0 problem becomes the future work hook in §7.

---

### 6.5 Silver labeling is rule-based, not learned

**Attack**: The silver labeling "pipeline" is ISBD regex matching with a confidence tier.
Sorbi et al. (2026) use actual weak supervision. By comparison, this is simpler and less
novel — it is domain-specific rule extraction, not a silver labeling method.

**Mitigation**: Own it. Frame the ISBD rules as "operationalised domain knowledge" rather
than a learned system. The contribution is identifying WHICH rules are reliable enough for
silver labeling at WHICH confidence tier — that is the expert knowledge contribution.
Avoid claiming this as a general silver labeling method; it is a domain-specific strategy.

---

### 6.6 "WEM Model" name misleads — only W is implemented

**Attack**: The paper is named "WEM Model" (Work–Expression–Manifestation) but evaluates
only Work-level entities. A reviewer familiar with FRBR will note this immediately:
"This is the W Model."

**Mitigation**: Two options:
- (A) Keep the name, add a footnote in §1: "Phase 1 covers Work-level entities (TITLE,
  OTHER\_TITLE, PERSON). Expression and Manifestation levels are future phases."
- (B) Rename to "WEM-W" or "W-Model" for the paper.

Option A is lower risk (name is already established in the project) and the FRBR/LRM audience
at JCDL will understand phased development. Add the footnote.

---

### 6.7 Fine-tuned results are the make-or-break number

**Attack**: The entire narrative — "silver labeling validates" — depends on fine-tuned F1 > zero-shot.
If fine-tuned F1 is ≤ zero-shot (possible if silver labels are too noisy), the contribution
collapses to "here is a task that is hard." That is publishable as a short paper, not a full paper.

**Mitigation**: No mitigation before the results exist. This is a genuine risk.
Contingency: if fine-tuned F1 disappoints, reframe as a short paper (4 pp) centred on the
dataset and evaluation protocol; drop the fine-tuning narrative. Accept a lower acceptance ceiling.

---

### 6.8 Summary table

| Vulnerability | Severity | Fix available now |
|---|---|---|
| Application, not invention | Medium | Yes — reframe §1 |
| Narrow system comparison | High | Yes — add gbert/XLM-R BIO baseline |
| "Ceiling" overclaim | Medium | Yes — change wording |
| Scale vs. scope mismatch | Medium | Yes — address in §3 |
| Silver = rule-based | Low–Medium | Yes — reframe §4 |
| WEM name misleads | Low | Yes — add footnote |
| Fine-tuned F1 unknown | High | No — wait for A3a/B5 |

### 6.9 Revised acceptance estimate

| Scenario | Range |
|---|---|
| Fine-tuned F1 ≥ 0.65 + BIO baseline added | 42–50% |
| Fine-tuned F1 ≥ 0.65, no BIO baseline | 35–42% |
| Fine-tuned F1 0.45–0.64 | 30–38% |
| Fine-tuned F1 < 0.45 | short paper fallback |

### 6.10 Action items from this assessment

- [x] Add XLM-R large fine-tuned BIO tagger as 5th system → added to A5, D3, B5b, wem-eval-plan.md
- [ ] Replace all "ceiling" language with "baseline characterisation" → §1, §5, abstract
- [ ] Add tier-0 scope statement to §3 Corpus and Task
- [ ] Add BibNER-Werk Phase 1 footnote to §1 Introduction
- [ ] Reframe silver labeling as "domain knowledge operationalisation" → §4

---

## 4. What NOT to submit for JCDL

- **GeMeA resource paper** — this is the ISWC 2026 Resource Track submission; submitting the same resource to JCDL is a dual-publication violation
- **babel-ddb/MOCHO design paper** — this is the SemDH 2026 submission; a JCDL version must foreground MOCHO *evaluation* (competency questions, alignment coverage), not the design rationale already in the SemDH paper; only viable if T13 (full MOCHO validation) is completed
- **Full GeMeA pipeline paper** — covers the same ground as both ISWC (resource) and Framing 1 (methodology); submitting all three creates dual-submission and self-plagiarism risk
