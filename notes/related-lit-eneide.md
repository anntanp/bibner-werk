---
title: Related Literature — ENEIDE (Santini et al., 2026)
up: []
related: []
created: 2026-06-14T00:00:00
updates:
  - 2026-06-24
in: bibner-werk
tags:
  - related-work
  - ner
  - eneide
source: bibner-werk/references/santini-2026-eneide.pdf
---

# Related Literature: ENEIDE (Santini et al., 2026)

## 0. Checklist — what needs to be done

**Paper writing**
- [ ] Add ENEIDE to §Related Work: cite as closest structural parallel (silver standard, DH domain, GLiNER fine-tuned)
- [ ] State all four differentiators explicitly in §Related Work (register, language, entity types, silver source mechanism) — see §4.5
- [ ] Cross-check BibNER-Werk's quality table format against ENEIDE Table 1: report wrong/missing spans separately, not collapsed into F1

**Evaluation**
- [ ] Break F1 results by era stratum (pre-1750, 1750–1900, post-1900) × silver tier — mirror ENEIDE's DZ/AMD split (§4.2)
- [ ] Quantify NIL PERSON entities in gold set: what fraction cannot be matched to GND? Report alongside ENEIDE's 7%/2% figures (§4.3)
- [ ] Confirm zero-shot LLaMA baseline is included in experiments — needed to replicate the zero-shot vs. fine-tuned gap argument (§1.4)

**Future work section**
- [ ] Frame Tier-0 enhancement as future work using ENEIDE's AMD pipeline as precedent: model → string propagation → human sample review (§4.1)
- [ ] Frame GND entity linking (Phase 1b) as future work analogous to ENEIDE's Wikidata EL component (§3.5)

**Data**
- [ ] Check if silver spans pipeline produces per-stratum counts (for the stratum × tier breakdown table)
- [ ] Add NIL PERSON count script to `scripts/py/` once gold set is finalized

**Full citation**: Santini, C., Barzaghi, S., Sernani, P., Frontoni, E., Melosi, L., & Alam, M. (2026). *ENEIDE: A High Quality Silver Standard Dataset for Named Entity Recognition and Linking in Historical Italian*. arXiv:2603.29801v1.

**What it is**: First multi-domain, publicly available NERL (NER + Entity Linking) dataset for historical Italian, with train/dev/test splits. 2,111 documents, 8,000+ annotations. Semi-automatically extracted from two scholarly digital editions (SDEs): Digital Zibaldone (DZ, Leopardi 1798–1837) and Aldo Moro Digitale (AMD, Moro 1916–1978). Entity types: PER, LOC, ORG, WORK. All linked to Wikidata; NIL entities included. Released CC BY-NC-SA 4.0.

---

## 1. Similarities

**1.1 Silver standard from structured sources.** Both exploit pre-existing structural annotations rather than annotating from scratch. ENEIDE harvests expert hyperlinks from SDE markup (HTML spans/links linked to Wikidata). BibNER-Werk harvests structural signals from ISBD punctuation conventions. Neither is a fully manual gold dataset.

**1.2 Historical text, underrepresented in NLP.** Both address historical language varieties absent from standard NLP training corpora. ENEIDE: 19th–20th century Italian prose. BibNER-Werk: historical German bibliographic strings spanning pre-1750 to contemporary.

**1.3 GLiNER fine-tuning as the primary model.** Both use `urchade/gliner_multi-v2.1` as the fine-tuned baseline. ENEIDE achieves F1 0.782 (DZ) and 0.876 (AMD). These are the direct comparison points for interpreting BibNER-Werk's fine-tuned results.

**1.4 Zero-shot vs. fine-tuned gap confirmed.** Both confirm that zero-shot LLMs (LLaMA 3.1-8B, Minerva-7B, Mistral-8B) fail on historical domain-specific NER. ENEIDE: LLaMA F1 0.405 (DZ) vs. GLiNER fine-tuned 0.782 — a 37-point gap. BibNER-Werk expects the same pattern and uses this to motivate fine-tuning.

**1.5 Digital humanities infrastructure framing.** Both are positioned as enabling resources for a downstream pipeline — ENEIDE for historical Italian NLP research, BibNER-Werk for the GeMeA KG construction pipeline.

---

## 2. Strategy comparison

| Dimension | ENEIDE | BibNER-Werk |
|---|---|---|
| Silver source | Pre-existing expert hyperlinks in SDEs | ISBD punctuation rules operationalized as field-boundary proxies |
| Coverage | Near-complete within scope (annotation gaps patched) | 7.56% of records have sufficient structural signal (Tier 1+2); Tier-0 out of scope |
| Quality gate | Expert review of 100-record samples; F1 95.6 (DZ), 79.8 (AMD) | FP rate per ISBD field on stratified 200-record sample; exclude fields >15% FP |
| Enhancement | String propagation + StanzaNLP NER for missed entities (AMD) | Planned as future work for Tier-0 records |
| Evaluation | NER + EL (Wikidata entity linking) | NER only in Phase 1; EL is GeMeA Phase 1b |
| Annotation | Philologists curated original SDEs (upstream, not project-run) | Three direct human annotators (frankp 393/395 primary; maria 131, gundi 82 for IAA) |

**Core strategic difference**: ENEIDE *reuses* expert curation already embedded in SDEs. BibNER-Werk *operationalizes domain conventions* because DDB records carry no pre-existing named-entity annotations — ISBD punctuation is the only structural signal available.

---

## 3. Differences

**3.1 Text register.** ENEIDE works on running prose (philosophical diary, political speeches). BibNER-Werk works on ISBD-formatted bibliographic catalog strings — a formulaic, abbreviated register, not prose. This is BibNER-Werk's primary differentiation claim and explains why standard NER fails on three independent dimensions: wrong label set, wrong domain, wrong register.

**3.2 Entity types.** ENEIDE: PER/LOC/ORG/WORK — conventional NE types anchored in ACE guidelines and linked to Wikidata. BibNER-Werk: TITLE/OTHER_TITLE/PERSON — FRBR Work-level entities. No geographic or organizational entities. The label set is not comparable to any existing NER benchmark.

**3.3 WORK vs. TITLE.** ENEIDE's WORK captures *references to other literary works within prose* (e.g., "Iliad" mentioned in Leopardi's diary). BibNER-Werk's TITLE captures *the title of the cataloged item itself* — a fundamentally different annotation target.

**3.4 Coverage gap.** ENEIDE annotates within its scope near-completely (AMD missing annotations reduced recall to 70.6%, then patched with enhancement pipeline). BibNER-Werk's Tier-0 (92.4% of corpus) is entirely out of scope for the current silver pipeline — a deliberate scope decision framed as future work, not a defect.

**3.5 Entity linking.** ENEIDE is a full NERL system with Wikidata disambiguation benchmarked against mGENRE, BLINK-ita, and BELA (best: mGENRE 0.689 on AMD, BELA 0.598 on DZ). BibNER-Werk is NER only in Phase 1; linking to GND/Wikidata is a separate downstream phase (GeMeA Phase 1b).

**3.6 Language.** ENEIDE: Italian (+ Latin/French/Greek excerpts in DZ). BibNER-Werk: German with Early Modern orthography variants (pre-1750) and Latin mixed in for the earliest strata.

**3.7 NIL entities.** ENEIDE: 7% NIL in DZ, 2% in AMD — quantified and discussed as an EL-specific challenge. BibNER-Werk: analogous gap exists for PERSON entities not in GND, not yet quantified.

---

## 4. What BibNER-Werk can emulate

**4.1 Annotation enhancement pipeline (Tier-0 coverage).** ENEIDE's AMD enhancement is directly applicable to BibNER-Werk's Tier-0 records: (1) run the fine-tuned model on Tier-0, (2) propagate high-confidence predictions via exact string matching across similar records, (3) validate a sample with human review. ENEIDE demonstrates this approach improved AMD recall from 70.6% to acceptable quality. The same argument justifies planning this as BibNER-Werk Phase 2.

**4.2 Cross-stratum evaluation framing.** ENEIDE reports performance separately on DZ (philosophical-literary) vs. AMD (political-legal), showing different entity distributions and difficulty profiles. BibNER-Werk should mirror this: report F1 per era stratum (pre-1750, 1750–1900, post-1900) × silver tier — not a single aggregate F1. ENEIDE gives the template reviewers will expect.

**4.3 NIL entity analysis.** ENEIDE quantifies NIL entities (7% DZ, 2% AMD) as a meaningful statistic about KG coverage gaps. BibNER-Werk should include an analogous figure: what fraction of PERSON entities in the gold set cannot be matched to GND? This is a concrete downstream relevance claim for GeMeA and frames a gap in the authority file.

**4.4 Quality reporting format (Table 1).** ENEIDE's quality table — annotations sampled, wrong annotations, missing annotations, precision, recall, F1 per corpus — is clean and directly reproducible. Use the same format for BibNER-Werk's IAA section: report wrong and missing spans separately rather than collapsing into a single F1.

**4.5 Cite ENEIDE as closest structural parallel, then differentiate explicitly.** ENEIDE is the most directly comparable paper BibNER-Werk will cite: both silver-standard, DH-domain, GLiNER fine-tuned, cultural heritage source. The differentiation must be stated explicitly in §Related Work: (a) ISBD bibliographic register vs. running prose, (b) German historical orthography vs. historical Italian, (c) FRBR Work-level entities (TITLE/OTHER_TITLE/PERSON) vs. general NE types (PER/LOC/ORG/WORK), (d) structural convention operationalization vs. SDE hyperlink extraction.

---

## 5. Key numbers from ENEIDE for §Related Work

| Statistic | Value |
|---|---|
| Documents total | 2,111 (DZ: 1,050; AMD: 1,061) |
| Annotations total | >8,000 |
| Entity types | PER, LOC, ORG, WORK |
| NIL entities | 7% (DZ), 2% (AMD) |
| Identifier overlap train+dev vs. test | 93.19% (DZ), 75.38% (AMD) |
| DZ quality (expert review F1) | 95.6 |
| AMD quality (expert review F1) | 79.8 |
| GLiNER fine-tuned F1 | 0.782 (DZ), 0.876 (AMD) |
| LLaMA 3.1-8B zero-shot F1 | 0.405 (DZ), 0.656 (AMD) |
| Best EL accuracy | 0.598 BELA (DZ), 0.689 mGENRE (AMD) |
