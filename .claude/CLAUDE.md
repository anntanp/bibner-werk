# BibNER-Werk — Claude Project Instructions

**BibNER-Werk** = Bibliographic NER for FRBR Work-level entities (*Werk* = German for "work", i.e. FRBR Work level).
NER system for FRBR Work-level entity extraction from historical German bibliographic catalog strings.

## Task

Extract TITLE, OTHER_TITLE, and PERSON entities from ISBD-formatted DDB catalog title strings using ISBD-guided silver labeling and fine-tuned `urchade/gliner_multi-v2.1`.

Phase 1 scope: Work-level entities (TITLE, OTHER_TITLE, PERSON). E and M levels are future phases.

## Data

- **Gold set** (395 records, Doccano JSONL): current export is `data/annotation/doccano/export-20260623-1110/` (annotator completion counts and IAA change each round — see `docs/inter-annotator-agreement.md` for current figures rather than hardcoding them here); adjudication status tracked in `notes/fiz/bibner-werk/adjudication-plan.md`
- **Silver spans**: done — `data/silver/silver_spans.jsonl` (2,384 tier-2 + 49,999 tier-1 = 52,383 records); see `notes/fiz/bibner-werk/A2-silver-spans-script-plan.md`
- **DDB parquet** (upstream, stays in gemea): `~/Documents/claude/gemea/data/`; pass path as CLI arg to scripts

## Models

- `urchade/gliner_multi-v2.1` — primary (zero-shot, few-shot, fine-tuned)
- `numind/NuNer_Zero` — zero-shot baseline
- Hardware: A100 80GB PCIe (bf16, flash attention 2); V100S 32GB fallback

## Paper

- **Title**: *BibNER-Werk: Bibliographic NER for Historical German Catalogs*
- **Venue**: JCDL 2026, ACM sigconf, double-blind
- **Location**: `paper/jcdl-2026/`
- **Deadline**: 30 June 2026
- **Plan**: `~/Documents/claude/notes/fiz/bibner-werk/jcdl-plan-202606.md` (tracks current task status — check here rather than this file for what's done vs. outstanding)

## GeMeA relationship

BibNER-Werk is the NER component of GeMeA Phase 0. See `notes/gemea-context.md`.

## Available role skills

| Skill | When to use |
|---|---|
| `/senior_python_developer` | NER scripts, eval pipeline, fine-tuning |
| `/senior_researcher` | Evaluation design, paper writing, model selection |
| `/expert_bash_scripter` | Shell scripts, data wrangling |

## Conventions

- Scripts → `scripts/py/`; every script has a header block (purpose, usage, inputs/outputs, dependencies, assumptions)
- Every script is also registered in `scripts/README.md` — one row per script, updated in the same change that adds or renames it
- `mypy --strict` + `ruff` clean before committing Python
- Data outputs → topic-named top-level dirs (`data/gold/`, `data/iaa/`, `data/adjudication/`, `data/silver/`, `data/title-length/`, `data/eval/`, `data/heuristics/`); annotation data → `data/annotation/`; a residual `data/processed/` still exists for a handful of not-yet-generated sr01/03/05/06/08/09/10/11 outputs. `data/gold/`, `data/iaa/`, `data/adjudication/`, `data/heuristics/` are git-tracked (manually reviewed ground truth, not reproducible from scripts); `data/processed/`, `data/silver/`, `data/title-length/`, `data/eval/`, and `data/annotation/doccano/` are gitignored (see notes/fiz/bibner-werk/data-reorg-plan.md)
- Notes → `notes/ner/`; paper notes → `notes/paper/`
- Transcripts → `~/Documents/claude/notes/transcripts-wem-model/`
- No hardcoded paths; pass parquet location as CLI argument
