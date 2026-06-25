# BibNER-Werk — Scripts

All scripts are standalone Python files in `scripts/py/`. Each has a header block: purpose, usage, inputs/outputs, dependencies, assumptions.

## Planned scripts

| Script | Status | Description |
|---|---|---|
| `build_silver_spans.py` | TODO (A2) | Stage 5 silver pipeline: extract char-offset spans from tier-1/2 parquet records |
| `eval_ner_gold.py` | TODO (A3) | Paper-ready evaluation: exact span match, per-stratum F1, 95% CI bootstrap, IAA |
| `train_gliner_silver.py` | TODO (A5) | Fine-tune `urchade/gliner_multi-v2.1` on silver spans; exclude gold IDs |

## Conventions

- No hardcoded paths — pass parquet location and data dirs as CLI arguments
- `mypy --strict` + `ruff` clean before commit
- Save all numeric outputs to `data/processed/` as CSV before referencing in notes
