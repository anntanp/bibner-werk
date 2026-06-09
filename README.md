# WEM Model: Bibliographic NER for Historical German Catalogs

**WEM** = Work–Expression–Manifestation (FRBR/LRM levels); *wem* (German dative, "to whom") signals attribution and authorship.

Named entity recognition for FRBR Work-level entities (TITLE, OTHER_TITLE, PERSON) in historical German bibliographic catalog strings from the Deutsche Digitale Bibliothek (DDB).

## Overview

Standard NER models fail on ISBD-formatted bibliographic catalog strings across three dimensions: wrong label set, wrong domain, and wrong register. WEM Model addresses this via:

1. **Silver labeling**: ISBD-guided signal extraction covering 7.6% of 9.2M DDB records (tier-1/2)
2. **Gold evaluation**: 395-record stratified gold set (by era and ISBD tier)
3. **Fine-tuned model**: `urchade/gliner_multi-v2.1` trained on silver labels

## Datasets

- `wem-silver`: 689K tier-1 + 7K tier-2 silver-labeled records (CC BY 4.0, pending DDB confirmation)
- `wem-gold`: 395-record manually annotated gold set

## Related

- [GeMeA](https://github.com/anntanp/gemea) — German Memory Atlas KG; WEM is GeMeA Phase 0 NER component
- JCDL 2026 paper: *WEM Model: Bibliographic NER for Historical German Catalogs*
