# BibNER-Werk — Scripts

All scripts are standalone Python files in `scripts/py/`. Each has a header block: purpose, usage, inputs/outputs, dependencies, assumptions.

## Planned scripts

| Script | Status | Description |
|---|---|---|
| `build_silver_spans.py` | Done (A2), 2026-07-06 | Stage 5 silver pipeline: extract char-offset spans (TITLE/OTHER_TITLE/PERSON) from ISBD field-flag ratings; see `notes/fiz/bibner-werk/A2-silver-spans-script-plan.md` |
| `verify_resp_classifier.py` | Done, 2026-07-06 | Layer-2 precision check (plan §8.2) for `build_silver_spans.py`'s `f_resp_*` classifier against the SR-04 ground-truth sample. Imports `classify_resp_batch` from `build_silver_spans.py` — a deliberate exception to the "standalone" convention above, since the point is to test the live classifier |
| `prepare_gliner_dataset.py` | Done, 2026-07-07 | Convert Doccano char-offset NER records (gold or silver) to GLiNER's word-indexed training format via GLiNER's own `WordsSplitter`; excludes and reports any span that doesn't align to a token boundary. Tests: `tests/test_prepare_gliner_dataset.py` |
| `eval_ner_gold.py` | Done (A3), 2026-07-07 | Paper-ready evaluation: split-folds/score/aggregate subcommands; exact span match, per-stratum F1, 95% bootstrap CI; tested against `tests/fixtures/` synthetic gold since `gold_reconciled.jsonl` (A7) doesn't exist yet; see `notes/fiz/bibner-werk/A3-eval-ner-gold-script-plan.md` |
| `train_gliner_silver.py` | TODO (A5) | Fine-tune `urchade/gliner_multi-v2.1` on `data/silver/gliner_dataset_silver.jsonl` (already built); exclude gold IDs; see `notes/fiz/bibner-werk/A5-finetuning-script-plan.md` |
| `train_xlmr_silver.py` | TODO (A5, second half) | Fine-tune `xlm-roberta-large` (BIO tagger) on silver spans; needs a char-offset → BIO conversion step not yet built; exclude gold IDs; sequenced after `train_gliner_silver.py` per §10 risk fallback; see `notes/fiz/bibner-werk/A5-finetuning-script-plan.md` |
| `build_data_provenance.py` | Done, 2026-07-07 | Build a cross-pipeline provenance table (`doc_id`, `is_gold`, `obj_id`, `parquet_row_id`) for every gold and silver title, joined to the s2 DDB parquet metadata file; asserts gold/silver are disjoint (leakage avoidance). Output: `data/provenance/bibner-data-provenance.parquet` (moved 2026-07-08 from `data/gold/` — the table covers both gold and silver records, not gold alone). Tests: `tests/test_build_data_provenance.py` |
| `compute_gold_label_stats.py` | Done, 2026-07-08 | Per-label entity instance counts and doc-level prevalence for `gold_reconciled.jsonl`, plus O_ONLY (zero-label docs) and TOTAL summary rows. Output: `data/gold/gold_label_stats.csv` |

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
| `ner/sr08_adjudication_to_doccano.py` | [adjudication-plan.md](../../notes/fiz/bibner-werk/adjudication-plan.md) §2.4 | Convert the full gold corpus (tiers 1+2+3) into EN/DE Doccano-importable JSONL for PI review | 2026-07-06 | — |
| `ner/sr08_apply_rationale_translations.py` | [adjudication-plan.md](../../notes/fiz/bibner-werk/adjudication-plan.md) §2.4 | Merge translated `llm_rationale_de` back into the `batch_NN.json` files in place | 2026-07-06 | — |
| `ner/sr08_build_blind_sample.py` | [adjudication-llm-justification.md](../../notes/fiz/bibner-werk/adjudication-llm-justification.md) §2 | Build a stratified random sample of tier-3 docs (excluding contested/precedent-referenced ones) as a Doccano import with no LLM suggestion/confidence, for a blind re-adjudication comparison | 2026-07-07 | — |
| `ner/sr08_compare_blind_sample.py` | [adjudication-llm-justification.md](../../notes/fiz/bibner-werk/adjudication-llm-justification.md) §2 | Compare the blind re-adjudication export against the PI's original (LLM-visible) decision and the LLM's suggestion itself, per doc and per conflict_type, with Wilson CIs | 2026-07-07 | — |
| `ner/sr08_check_agent_coverage.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Report dc:creator/dc:contributor presence rates overall and by era | 2026-04-09T23:17:52 | — |
| `ner/sr08_check_person_in_title.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Report ner_person hit rate per era to assess PERSON extraction utility | 2026-04-09T23:17:52 | — |
| `ner/sr08_ci_sample_size.py` | [sr08_evaluation-design.md](../notes/ner/sr08_evaluation-design.md) | Compute minimum entity instance counts for target CI half-width on F1 | 2026-04-09T23:17:52 | — |
| `ner/sr08_classify_disagreements.py` | [adjudication-plan.md](../../notes/fiz/bibner-werk/adjudication-plan.md) §1–2.1 | Classify each gold doc into adjudication tier (1 auto-accept / 2 majority-auto / 3 PI-adjudicated) | 2026-07-03 | — |
| `ner/sr08_compute_iaa.py` | [inter-annotator-agreement.md](../docs/inter-annotator-agreement.md) | Compute Krippendorff's α (nominal) for gold annotation | 2026-06-10T00:07:30 | — |
| `ner/sr08_compute_llm_acceptance.py` | [adjudication-llm-justification.md](../../notes/fiz/bibner-werk/adjudication-llm-justification.md) §3.1 | Compute PI acceptance/override rate for LLM-drafted tier-3 adjudication suggestions, cross-tabbed by conflict_type and llm_confidence; cross-refs contested-tokens/precedent comment tags | 2026-07-07 | — |
| `ner/sr08_corpus_cell_sizes.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Compute era × silver_tier corpus cell sizes from actual data | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr08_count_comments.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Count and list annotator comments across all Doccano export JSONL files | 2026-06-10T00:07:30 | — |
| `ner/sr08_doccano_import.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Extract title and spans from sr08_gold_prefilled.jsonl for Doccano import | 2026-04-09T23:17:52 | — |
| `ner/sr08_draft_adjudication.py` | [adjudication-plan.md](../../notes/fiz/bibner-werk/adjudication-plan.md) §2.2 | Batch tier-3 docs into paste-able LLM prompts | 2026-07-03 | — |
| `ner/sr08_extract_contested_tokens.py` | [adjudication-rulings.md](../docs/adjudication-rulings.md) | Parse the contested-tokens note into a flat (doc_id, obj_id, contested_token) list | 2026-07-06 | — |
| `ner/sr08_extract_rationales_for_translation.py` | [adjudication-plan.md](../../notes/fiz/bibner-werk/adjudication-plan.md) §2.4 | Extract `llm_rationale` texts needing translation into one consolidated JSON | 2026-07-06 | — |
| `ner/sr08_gold_composition_audit.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Audit gold set composition (era × tier) against original allocation targets | 2026-04-09T23:17:52 | — |
| `ner/sr08_gold_dctype_breakdown.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Show dc_type composition of gold sample broken down by era stratum | 2026-04-09T23:17:52 | — |
| `ner/sr08_ingest_llm_drafts.py` | [adjudication-plan.md](../../notes/fiz/bibner-werk/adjudication-plan.md) §2.3 | Ingest pasted LLM adjudication drafts into the canonical tier-3 sheet | 2026-07-03 | — |
| `ner/sr08_lookup_record_provenance.py` | [adjudication-rulings.md](../docs/adjudication-rulings.md) | Look up cross-pipeline identifiers (obj_id, gemea parquet row, silver doc id) for gold adjudication doc_ids | 2026-07-06 | — |
| `ner/sr08_make_meta_sample.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Generate a 10-record Doccano import sample with meta for UI verification | 2026-06-10T00:07:30 | — |
| `ner/sr08_prefill_spans.py` | [annotation-guide.md](../docs/annotation-guide.md) | Pre-fill NER annotation spans for the gold sample using ISBD rules | 2026-04-09T23:17:52 | — |
| `ner/sr08_sample_gold.py` | [sr08_gold-set-composition.md](../notes/ner/sr08_gold-set-composition.md) | Draw stratified ~500-record gold sample from DF_DE_TITLES for annotation | 2026-04-09T23:17:52 | — |
| `ner/sr08_spotcheck_doccano_import.py` | [adjudication-plan.md](../../notes/fiz/bibner-werk/adjudication-plan.md) §2.4 | Spot-check the tier 1/2/3 EN/DE Doccano conversion — print one EN+DE record per tier | 2026-07-06 | — |
| `ner/sr08_verify_adjudication_batch.py` | [adjudication-plan.md](../../notes/fiz/bibner-werk/adjudication-plan.md) §2.1 note | Validate a drafted `batch_NN.json` against its source prompt (replaces ad-hoc verification snippets) | 2026-07-03 | — |
| `ner/sr08_verify_meta_export.py` | [sr08_doccano-meta.md](../notes/ner/sr08_doccano-meta.md) | Verify that meta enrichment preserved all annotations and comments exactly | 2026-06-10T00:07:30 | — |
| `ner/sr08_verify_spans.py` | [annotation-guide.md](../docs/annotation-guide.md) | Spot-check pre-filled annotation span character offsets | 2026-04-09T23:17:52 | — |
| `ner/sr09_eval_nuner_tier2.py` | [sr09_nuner-tier2-sanity.md](../notes/ner/sr09_nuner-tier2-sanity.md) | Sanity-check NuNER on tier-2 records using ISBD silver spans as pseudo-gold | 2026-04-09T23:17:52 | — |
| `ner/sr10_analyse_title_lengths.py` | [sr10_de-titles-distribution.md](../notes/ner/sr10_de-titles-distribution.md) | Plot title-length distribution (BPE or word tokens) per year bucket | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet<br>2026-05-28: data-path update<br>2026-07-03: fixed `length_thresholds` in saved JSON always reporting BPE-mode constants regardless of `--word-tokens` |
| `ner/sr10_era_length_summary.py` | [sr10_title-length-thresholds.md](../notes/ner/sr10_title-length-thresholds.md) | Summarise title-length distribution (short/medium/long, median) per era stratum | 2026-04-09T23:17:52 | — |
| `ner/sr10_era_length_summary_v2.py` | [dataset.md](../docs/dataset.md) §2.1 | Era-bucket (pre-1700/1700-1800/19th-c/modern/unknown) short/medium/long summary computed directly from `de_titles_tokenized.parquet`; BPE tokens (`--short-max`/`--medium-max`, default p25/p75=8/27) or `--word-tokens` (whitespace, default 3/12) | 2026-07-03 | — |
| `ner/sr10_explore_token_distribution.py` | [sr10_de-titles-distribution.md](../notes/ner/sr10_de-titles-distribution.md) | Plot raw BPE token-count distribution to identify short/medium/long breakpoints | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr10_render_title_viz.py` | [sr10_de-titles-distribution.md](../notes/ner/sr10_de-titles-distribution.md) | Render fig_title_lengths.jsx and fig_title_lengths_bw.html from title-length JSON | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr10_word_token_era_summary.py` | [sr10_tracing-df-de-titles.md](../notes/ner/sr10_tracing-df-de-titles.md) | **BROKEN, never successfully run** — hardcoded path outside this repo, no CLI override, wrong `dates`-as-list schema assumption, no title-regex year fallback. Superseded by `ner/sr10_era_length_summary_v2.py --word-tokens`. Kept for history. | 2026-05-28T20:56:19 | 2026-07-03: marked broken/superseded (see script header) |
| `ner/sr11_annotate.py` | [sr11_labeling-strategy.md](../notes/ner/sr11_labeling-strategy.md) | Interactive helper for manual span labeling of sr11_prompt_validation_manual.jsonl | 2026-04-09T23:17:52 | — |
| `ner/sr11_dctype_by_era.py` | [sr11_labeling-strategy.md](../notes/ner/sr11_labeling-strategy.md) | Show dc_type prevalence and stacked bar chart per era stratum | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet |
| `ner/sr11_sample_validation.py` | [sr11_labeling-strategy.md](../notes/ner/sr11_labeling-strategy.md) | Sample 50 pre-1750 tier-0 records for manual prompt validation (T11.1a) | 2026-04-09T23:17:52 | 2026-04-15: ported from .pkl to parquet<br>2026-04-20: remote export scripts |

## Conventions

- No hardcoded paths — pass parquet location and data dirs as CLI arguments
- `mypy --strict` + `ruff` clean before commit
- Save all numeric outputs to `data/processed/` as CSV before referencing in notes
