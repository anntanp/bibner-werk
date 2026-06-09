# WEM Model — Claude Project Instructions

**WEM** = Work–Expression–Manifestation (FRBR/LRM levels).
NER system for FRBR Work-level entity extraction from historical German bibliographic catalog strings.

## Task

Extract TITLE, OTHER_TITLE, and PERSON entities from ISBD-formatted DDB catalog title strings using ISBD-guided silver labeling and fine-tuned `urchade/gliner_multi-v2.1`.

Phase 1 scope: Work-level entities (TITLE, OTHER_TITLE, PERSON). E and M levels are future phases.

## Data

- **Gold set** (395 records, Doccano JSONL): `data/annotation/doccano/export-20260608-1221/`
  - Primary annotator: `frankp.jsonl` (393/395)
  - IAA subsets: `maria.jsonl` (131), `gundi.jsonl` (82)
  - Claude pre-annotation (seeding): `ann.jsonl` (183) — not used for IAA
- **Silver spans** (to be built): `data/processed/silver_spans.jsonl`
- **DDB parquet** (upstream, stays in gemea): `~/Documents/claude/gemea/data/`; pass path as CLI arg to scripts

## Models

- `urchade/gliner_multi-v2.1` — primary (zero-shot, few-shot, fine-tuned)
- `numind/NuNer_Zero` — zero-shot baseline
- Hardware: A100 80GB PCIe (bf16, flash attention 2); V100S 32GB fallback

## Paper

- **Title**: *WEM Model: Bibliographic NER for Historical German Catalogs*
- **Venue**: JCDL 2026, ACM sigconf, double-blind
- **Location**: `paper/jcdl-2026/`
- **Deadline**: 30 June 2026
- **Plan**: `~/Documents/claude/notes/project-plans/jcdl-plan-202606.md`

## GeMeA relationship

WEM is the NER component of GeMeA Phase 0. See `notes/context/gemea-context.md`.

## Available role skills

| Skill | When to use |
|---|---|
| `/senior_python_developer` | NER scripts, eval pipeline, fine-tuning |
| `/senior_researcher` | Evaluation design, paper writing, model selection |
| `/expert_bash_scripter` | Shell scripts, data wrangling |

## Conventions

- Scripts → `scripts/py/`; every script has a header block (purpose, usage, inputs/outputs, dependencies, assumptions)
- `mypy --strict` + `ruff` clean before committing Python
- Data outputs → `data/processed/`; annotation data → `data/annotation/`
- Notes → `notes/ner/`; paper notes → `notes/paper/`
- Transcripts → `~/Documents/claude/notes/transcripts-wem-model/`
- No hardcoded paths; pass parquet location as CLI argument
