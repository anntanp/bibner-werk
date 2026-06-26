# BibNER-Werk — Scripts

All scripts are standalone Python files in `scripts/py/`. Each has a header block: purpose, usage, inputs/outputs, dependencies, assumptions.

## Planned scripts

| Script | Status | Description |
|---|---|---|
| `build_silver_spans.py` | TODO (A2) | Stage 5 silver pipeline: extract char-offset spans from tier-1/2 parquet records |
| `eval_ner_gold.py` | TODO (A3) | Paper-ready evaluation: exact span match, per-stratum F1, 95% CI bootstrap, IAA |
| `train_gliner_silver.py` | TODO (A5) | Fine-tune `urchade/gliner_multi-v2.1` on silver spans; exclude gold IDs |

## NER Study Record Scripts

Scripts in `ner/` implement study records sr01–sr11. Each script has a header block covering purpose, usage, inputs/outputs, dependencies, and assumptions.

| Script | Source Note | Purpose | Created | Updates |
|---|---|---|---|---|
| `ner/sr01_check_isbd_titles.py` | [sr01_isbd-applicability.md](../notes/ner/sr01_isbd-applicability.md) | Check how many DE-title strings contain ISBD punctuation marks | 2026-04-09T23:17:52 | — |
| `ner/sr01_rate_isbd_fields.py` | [sr01_isbd-field-rating.md](../notes/ner/sr01_isbd-field-rating.md) | Rate DDB title strings for ISBD field presence and assign silver tiers | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr03_extract_fp_examples.py` | [sr03_silver-label-fp-review.md](../notes/ner/sr03_silver-label-fp-review.md) | Extract FP/FN examples from heuristic validation CSV for documentation | 2026-04-09T23:17:52 | — |
| `ner/sr03_fp_review.py` | [sr03_silver-label-fp-review.md](../notes/ner/sr03_silver-label-fp-review.md) | Apply automated and manual FP rules to sr03_heuristic_validation_sample.csv | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr03_validate_fp_review.py` | [sr03_silver-label-fp-review.md](../notes/ner/sr03_silver-label-fp-review.md) | Validate FP review output — row count, field names, flag consistency | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr03_validate_heuristic_fields.py` | [sr03_silver-label-fp-review.md](../notes/ner/sr03_silver-label-fp-review.md) | Sample 200 heuristic-tier records and write FP review sheet | 2026-04-09T23:17:52 | — |
| `ner/sr04_evaluate_translator_heuristic.py` | [sr04_translator-person-disambiguation.md](../notes/ner/sr04_translator-person-disambiguation.md) | Evaluate translator keyword heuristic (precision/recall/F1) | 2026-04-09T23:17:52 | — |
| `ner/sr04_validate_translator_disambiguation.py` | [sr04_translator-person-disambiguation.md](../notes/ner/sr04_translator-person-disambiguation.md) | Sample 100 f_person records and apply translator heuristic for manual review | 2026-04-09T23:17:52 | — |
| `ner/sr05_trailing_period_review.py` | [sr05_trailing-period-noise.md](../notes/ner/sr05_trailing-period-noise.md) | Annotate sr05_trailing_period_sample.csv with true_class (ISBD_CLOSE/ABBREV/ORDINAL) | 2026-04-09T23:17:52 | — |
| `ner/sr05_validate_trailing_period.py` | [sr05_trailing-period-noise.md](../notes/ner/sr05_trailing-period-noise.md) | Sample 200 trailing-period records and write classification review sheet | 2026-04-09T23:17:52 | — |
| `ner/sr06_evaluate_historical.py` | [sr06_historical-scope.md](../notes/ner/sr06_historical-scope.md) | Evaluate heuristic accuracy for historical/Latin title classification | 2026-04-09T23:17:52 | — |
| `ner/sr06_historical_scope.py` | [sr06_historical-scope.md](../notes/ner/sr06_historical-scope.md) | Estimate proportion of pre-1800 records that are Latin or Early Modern German | 2026-04-09T23:17:52 | — |
| `ner/sr06_plot_silver_tiers.py` | [sr06_historical-scope.md](../notes/ner/sr06_historical-scope.md) | Plot silver-tier composition by era as stacked bar chart | 2026-04-09T23:17:52 | — |
| `ner/sr06_spotcheck_results.py` | [sr06_historical-scope.md](../notes/ner/sr06_historical-scope.md) | Print spot-check tables from sr06_historical_evaluated.csv | 2026-04-09T23:17:52 | — |
| `ner/sr08_add_meta_to_export.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Inject htype + DDB URL meta into Doccano annotator JSONL exports | 2026-06-10T00:07:30 | — |
| `ner/sr08_check_agent_coverage.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Report dc:creator/dc:contributor presence rates overall and by era | 2026-04-09T23:17:52 | — |
| `ner/sr08_check_person_in_title.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Report ner_person hit rate per era to assess PERSON extraction utility | 2026-04-09T23:17:52 | — |
| `ner/sr08_ci_sample_size.py` | [sr08_evaluation-design.md](../notes/ner/sr08_evaluation-design.md) | Compute minimum entity instance counts for target CI half-width on F1 | 2026-04-09T23:17:52 | — |
| `ner/sr08_corpus_cell_sizes.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Compute era × silver_tier corpus cell sizes from actual data | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr08_count_comments.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Count and list annotator comments across all Doccano export JSONL files | 2026-06-10T00:07:30 | — |
| `ner/sr08_doccano_import.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Extract title and spans from sr08_gold_prefilled.jsonl for Doccano import | 2026-04-09T23:17:52 | — |
| `ner/sr08_gold_composition_audit.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Audit gold set composition (era × tier) against original allocation targets | 2026-04-09T23:17:52 | — |
| `ner/sr08_gold_dctype_breakdown.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Show dc_type composition of gold sample broken down by era stratum | 2026-04-09T23:17:52 | — |
| `ner/sr08_make_meta_sample.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Generate a 10-record Doccano import sample with meta for UI verification | 2026-06-10T00:07:30 | — |
| `ner/sr08_prefill_spans.py` | [sr08_annotation-guide.md](../notes/ner/sr08_annotation-guide.md) | Pre-fill NER annotation spans for the gold sample using ISBD rules | 2026-04-09T23:17:52 | — |
| `ner/sr08_sample_gold.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Draw stratified ~500-record gold sample from DF_DE_TITLES for annotation | 2026-04-09T23:17:52 | — |
| `ner/sr08_verify_meta_export.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Verify that meta enrichment preserved all annotations and comments exactly | 2026-06-10T00:07:30 | — |
| `ner/sr08_verify_spans.py` | [sr08_annotation-guide.md](../notes/ner/sr08_annotation-guide.md) | Spot-check pre-filled annotation span character offsets | 2026-04-09T23:17:52 | — |
| `ner/sr09_eval_nuner_tier2.py` | [sr09_nuner-tier2-sanity.md](../notes/ner/sr09_nuner-tier2-sanity.md) | Sanity-check NuNER on tier-2 records using ISBD silver spans as pseudo-gold | 2026-04-09T23:17:52 | — |
| `ner/sr10_analyse_title_lengths.py` | [sr10_de-titles-distribution.md](../notes/ner/sr10_de-titles-distribution.md) | Plot title-length distribution (BPE or word tokens) per year bucket | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet<br>2026-05-28: data-path update |
| `ner/sr10_era_length_summary.py` | [sr10_title-length-thresholds.md](../notes/ner/sr10_title-length-thresholds.md) | Summarise title-length distribution (short/medium/long, median) per era stratum | 2026-04-09T23:17:52 | — |
| `ner/sr10_explore_token_distribution.py` | [sr10_de-titles-distribution.md](../notes/ner/sr10_de-titles-distribution.md) | Plot raw BPE token-count distribution to identify short/medium/long breakpoints | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr10_render_title_viz.py` | [sr10_de-titles-distribution.md](../notes/ner/sr10_de-titles-distribution.md) | Render fig_title_lengths.jsx and fig_title_lengths_bw.html from title-length JSON | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr10_word_token_era_summary.py` | [sr10_tracing-df-de-titles.md](../notes/ner/sr10_tracing-df-de-titles.md) | Compute per-era median whitespace word count from de_titles_tokenized.parquet | 2026-05-28T20:56:19 | — |
| `ner/sr11_annotate.py` | [sr11_labeling-strategy.md](../notes/ner/sr11_labeling-strategy.md) | Interactive helper for manual span labeling of sr11_prompt_validation_manual.jsonl | 2026-04-09T23:17:52 | — |
| `ner/sr11_dctype_by_era.py` | [sr11_labeling-strategy.md](../notes/ner/sr11_labeling-strategy.md) | Show dc_type prevalence and stacked bar chart per era stratum | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr11_sample_validation.py` | [sr11_labeling-strategy.md](../notes/ner/sr11_labeling-strategy.md) | Sample 50 pre-1750 tier-0 records for manual prompt validation (T11.1a) | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet<br>2026-04-20: remote export scripts |

## Conventions

- No hardcoded paths — pass parquet location and data dirs as CLI arguments
- `mypy --strict` + `ruff` clean before commit
- Save all numeric outputs to `data/processed/` as CSV before referencing in notes
