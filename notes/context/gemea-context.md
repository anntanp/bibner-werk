# GeMeA Context

WEM Model is the NER component of GeMeA Phase 0. This note documents the upstream relationship.

## WEM's role in the GeMeA pipeline

```
DDB parquet (9.21M records)
  └─ gemea/data/
       ├─ Tier-2 (0.08%): ISBD structural → direct silver labels
       ├─ Tier-1 (7.48%): ISBD heuristic → silver labels (this is WEM's training data)
       └─ Tier-0 (92.44%): no ISBD signal → WEM NER inference ← WEM's target
```

WEM output (predicted TITLE / OTHER_TITLE / PERSON spans) feeds back into GeMeA Phase 0 as input to `link_gnd_works.py`, which uses extracted titles and authors to look up GND Werk entries via lobid-gnd.

## Upstream data source

DDB parquet files live in `~/Documents/claude/gemea/data/`. Pass the parquet path as a CLI argument to WEM scripts — do not hardcode paths or copy parquet files into this repo.

## Sector and mediatype reference

See `gemea/.claude/CLAUDE.md` for the full sector (sparte) and mediatype (mt) code tables.

## Related GeMeA papers

- ISWC 2026 Resource Track: GeMeA KG resource description
- SemDH 2026 (babel-ddb): MOCHO hub ontology design
- JCDL 2026 (this project): WEM Model bibliographic NER

## Silver tier assignments

Tier assignments (tier-0/1/2 per record) are computed in the GeMeA pipeline using ISBD signal detection. The tier column in the parquet is the entry point for building `silver_spans.jsonl` (see `scripts/py/build_silver_spans.py`).
