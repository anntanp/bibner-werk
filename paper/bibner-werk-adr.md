---
up: "[[bibner-werk]]"
related: "[[sr08-gold-iaa-script-plan]]"
created: 2026-06-24
updates: []
in: bibner-werk/paper
tags: [adr, iaa, annotation, krippendorff, jcdl-2026]
---

# Architecture Decision Records — JCDL 2026

---

## ADR-01 Inter-Coder Agreement Metric

**Status**: Accepted  
**Date**: 2026-06-24

### Context

The BibNER-Werk gold corpus is annotated by three annotators (frankp, gundi, maria) with near-complete but uneven coverage:

| Annotator | Reviewed / Total | Notes |
|-----------|-----------------|-------|
| frankp    | 394 / 395       | 1 unreviewed |
| gundi     | 395 / 395       | 130 empty-label + comment ("Keine Annotation") |
| maria     | 392 / 395       | 98 empty-label + comment; 3 empty+no-comment treated as reviewed |

We need a chance-corrected inter-coder agreement coefficient to report annotation reliability in the paper.

### Decision

Use **Krippendorff's α** with the **nominal distance function**.

### Justification

Justified from Artstein and Poesio (2008), "Inter-Coder Agreement for Computational Linguistics," *Computational Linguistics* 34(4):555–596.

#### 1. Applies to three coders

> "The coefficient α (Krippendorff 1980, 2004a) is an extremely versatile agreement coefficient based on assumptions similar to π... **It applies to multiple coders**, and it allows for different magnitudes of disagreement." (§2.6.1, p. 564)

Cohen's κ is defined for two coders only; Fleiss's multi-π and Krippendorff's α both generalise to c > 2. α is preferred here (see point 3).

#### 2. Nominal NER labels

The paper defines a distance function for nominal categories, i.e. categories that are mutually exclusive and equally distinct:

> "the function for nominal categories, that is, a function which considers all distinct labels equally distant from one another: d_ab = 0 if a = b; 1 if a ≠ b" (§2.6.1, pp. 566–567)

NER entity labels are nominal. This distance function is the correct and unambiguous choice.

#### 3. Measures annotation procedure reliability, not individual annotator bias

The goal is to show that the annotation *procedure* is reliable (so the gold labels are trustworthy), not to characterise individual coders. The paper argues:

> "The most appropriate measures of reliability for this purpose are therefore single-distribution coefficients like π and **α**, which generalize over the individual coders and exclude marginal disagreements from the expected agreement." (§3.1, p. 571)

> "We agree with the view that K and α are more appropriate, as they abstract away from the bias of specific coders." (§5.2, p. 590)

κ-style coefficients incorporate individual annotator marginals into expected agreement; α (like multi-π) uses a pooled distribution, which is the right model when the goal is reproducibility.

#### 4. Near-complete coverage — missing data is negligible

α's formula uses n_ik (number of coders assigning item i to category k), which naturally handles items annotated by fewer than c coders. With 1–3 missing judgments out of 395 items, the effect on the coefficient is negligible.

### Alternatives considered

| Metric | Reason rejected |
|--------|----------------|
| Cohen's κ | Defined for 2 coders only |
| Fleiss's multi-π | Valid, but α is nearly identical for nominal data and preferred by Artstein & Poesio for reliability studies |
| Percentage agreement | Does not correct for chance; biased by category distribution (§2.3, p. 558) |

### Caveat

Artstein and Poesio warn that α with a **non-nominal** (weighted) distance metric can be difficult to interpret because results vary substantially with the choice of weights (§2.6.1, p. 567). This risk does not apply here: the nominal distance function is unambiguous for NER labels.

### Consequence

- Report α computed on the pre-adjudication annotations (raw labels before manual resolution).
- Use the nominal distance function (d_ab = 0 if a = b, else 1).
- Cite: Krippendorff (1980, 2004a) as the original source; Artstein and Poesio (2008) as the methodological justification.
- Bootstrap confidence intervals if reporting significance (Krippendorff 2004a, §11.4.2; Hayes and Krippendorff 2007).
