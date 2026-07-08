# BibNER-Werk: Bibliographic NER for Historical German Catalogs

**BibNER-Werk** = Bibliographic NER for FRBR Work-level entities. *Werk* (German for "work") denotes the FRBR/LRM Work level.

Named entity recognition for FRBR Work-level entities (TITLE, OTHER_TITLE, PERSON) in historical German bibliographic catalog strings from the Deutsche Digitale Bibliothek (DDB).

## Overview

Standard NER models fail on ISBD-formatted bibliographic catalog strings across three dimensions: wrong label set, wrong domain, and wrong register. BibNER-Werk addresses this via:

1. **Silver labeling**: ISBD-guided signal extraction covering 7.6% of 9.2M DDB records (tier-1/2)
2. **Gold evaluation**: 395-record stratified gold set (by era and ISBD tier)
3. **Fine-tuned model**: `urchade/gliner_multi-v2.1` trained on silver labels

## Setup

```bash
cd /Users/mta/Documents/claude/bibner-werk
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow torch transformers gliner flair
```

Dedicated venv — does not share or depend on `gemea/.venv`, per the 2026-06-10 project split.

## Datasets

- `wem-silver`: 689K tier-1 + 7K tier-2 silver-labeled records (CC BY 4.0, pending DDB confirmation)
- `wem-gold`: 395-record manually annotated gold set

## Related

- [GeMeA](https://github.com/anntanp/gemea) — German Memory Atlas KG; BibNER-Werk is GeMeA Phase 0 NER component
- JCDL 2026 paper: *BibNER-Werk: Bibliographic NER for Historical German Catalogs*
