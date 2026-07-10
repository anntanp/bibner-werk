#!/usr/bin/env python3
# Purpose:      A5 — fine-tune urchade/gliner_multi-v2.1 on silver-labeled BibNER-Werk
#               spans (TITLE, OTHER_TITLE, PERSON). One of k replicate runs (--fold-index)
#               for cross-fold variance estimation, per the k-run semantics in
#               notes/fiz/bibner-werk/A5-finetuning-script-plan.md §4. Training data is
#               identical across all k runs (gold IDs already excluded regardless of
#               fold); only the seed differs.
#               See notes/fiz/bibner-werk/A5-finetuning-script-plan.md §3/§5,
#               A5-finetuning-gliner.md (API research), A5-finetuning-gliner-actual.md
#               (decision log), A5-person-imbalance.md (oversampling rationale).
# Usage:        python3 scripts/py/train_gliner_silver.py --fold-index 0
#                   [--silver-data PATH] [--gold PATH] [--provenance PATH]
#                   [--seed-base INT] [--base-model STR] [--precision {fp16,bf16}]
#                   [--dev-fraction FLOAT] [--dev-split-seed INT]
#                   [--epochs INT] [--learning-rate FLOAT] [--others-learning-rate FLOAT]
#                   [--warmup-ratio FLOAT] [--batch-size INT]
#                   [--oversample-tier2-factor FLOAT]
#                   [--focal-loss-alpha FLOAT] [--focal-loss-gamma FLOAT]
#                   [--output-dir PATH] [--dry-run]
# Inputs:       data/silver/gliner_dataset_silver.jsonl (GLiNER schema:
#                   {"id","tokenized_text","ner","silver_tier","era"} — silver_tier/era
#                   added 2026-07-10, see A5-finetuning-gliner-actual.md #1-2)
#               data/gold/gold_reconciled.jsonl (leakage guard)
#               data/provenance/bibner-data-provenance.parquet (leakage guard,
#                   gold obj_id resolution)
# Outputs:      models/gliner_fold{fold-index}/ (GLiNER checkpoint, via save_pretrained)
# Dependencies: gliner, transformers, torch, pyarrow, scikit-learn
# Assumptions:  Training labels are short canonical strings (TITLE/OTHER_TITLE/PERSON),
#               matching gold data and eval_ner_gold.py exactly — resolved 2026-07-10
#               after reading the actual GLiNER paper; see A5-finetuning-gliner.md §2.
#               Dev split (90/10, era x silver_tier-stratified) is for best-epoch
#               checkpoint selection only; the gold folds (B6) remain the untouched
#               held-out test set. Negative-sampling ratio and the exact
#               urchade/gliner_multi-v2.1 architecture subclass are NOT yet resolved
#               (A5-finetuning-gliner-actual.md "Still open") — left at gliner's class
#               defaults rather than guessed at.
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SILVER_DATA = ROOT / "data" / "silver" / "gliner_dataset_silver.jsonl"
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_PROVENANCE = ROOT / "data" / "provenance" / "bibner-data-provenance.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "models"

LABELS = {"TITLE", "OTHER_TITLE", "PERSON"}  # matches eval_ner_gold.py's LABELS; ADR D-10


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Stream a JSONL file into a list of dicts, one per non-empty line."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def validate_labels(records: list[dict[str, Any]]) -> None:
    """Fail loudly if any span uses a label outside the fixed TITLE/OTHER_TITLE/PERSON set.

    Catches schema drift (e.g. an un-rebuilt silver file, or a stray label
    from an unrelated pipeline change) before it silently trains a model on
    the wrong label vocabulary.
    """
    seen = {label for rec in records for _, _, label in rec["ner"]}
    unexpected = seen - LABELS
    if unexpected:
        raise ValueError(
            f"Training data contains labels outside {LABELS}: {unexpected}. "
            "Refusing to train — check --silver-data matches the expected schema."
        )


def leakage_guard(silver_records: list[dict[str, Any]], gold_path: Path, provenance_path: Path) -> None:
    """Assert no silver record's id is a gold obj_id. Raises, does not warn.

    Resolves gold's 395 Doccano ids to obj_id via the provenance parquet
    (built by build_data_provenance.py, which already asserts gold/silver
    obj_id disjointness at build time) — re-checked here defensively since
    that assertion was computed by a different script at a different time.
    See A5-finetuning-script-plan.md §2.
    """
    gold_ids = {rec["id"] for rec in read_jsonl(gold_path)}
    table = pq.read_table(provenance_path, columns=["doc_id", "is_gold", "obj_id"])
    prov = table.to_pylist()
    gold_obj_ids = {row["obj_id"] for row in prov if row["is_gold"] and row["doc_id"] in gold_ids}

    silver_ids = {rec["id"] for rec in silver_records}
    leaked = silver_ids & gold_obj_ids
    if leaked:
        raise AssertionError(
            f"Leakage guard failed: {len(leaked)} silver record(s) share an obj_id with a gold "
            f"record: {sorted(leaked)[:5]}{'...' if len(leaked) > 5 else ''}. Training on leaked "
            "gold records would silently inflate every downstream F1 number — refusing to proceed."
        )
    logging.info(
        "Leakage guard passed: 0 of %d silver records overlap with %d gold obj_ids.",
        len(silver_ids), len(gold_obj_ids),
    )


def oversample_tier2(records: list[dict[str, Any]], factor: float) -> list[dict[str, Any]]:
    """Duplicate tier-2 records to partially offset PERSON scarcity.

    Bounded, not full rebalancing — see A5-person-imbalance.md §2.1 for why
    (only 2,384 unique tier-2 texts; aggressive oversampling risks the model
    memorizing tier-2's narrow "/ Name" surface pattern rather than
    generalizing). factor=1.0 is a no-op; factor=3.0 triples tier-2's count
    by adding 2 extra copies of each tier-2 record.
    """
    if factor <= 1.0:
        return records
    tier2 = [r for r in records if r.get("silver_tier") == 2]
    extra_copies = int(round(factor - 1.0))
    expanded = records + tier2 * extra_copies
    logging.info(
        "Oversampled tier-2: %d tier-2 records x%d extra copies = %d added (pool: %d -> %d).",
        len(tier2), extra_copies, len(tier2) * extra_copies, len(records), len(expanded),
    )
    return expanded


def stratified_dev_split(
    records: list[dict[str, Any]], dev_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """90/10 (by default) train/dev split, stratified by era x silver_tier.

    Single fixed split — call once, reuse across all --fold-index runs (not
    re-derived per fold). Falls back to era-only stratification, then to an
    unstratified split, if the composite key has strata too small for
    sklearn's stratified split to run (each class needs >= 2 members) —
    logged, not silent, since it changes what "stratified" actually means
    for this run.
    """
    keys = [f"{r.get('era')}_{r.get('silver_tier')}" for r in records]
    try:
        train, dev = train_test_split(
            records, test_size=dev_fraction, random_state=seed, stratify=keys
        )
        logging.info("Dev split: era x silver_tier-stratified, %d train / %d dev.", len(train), len(dev))
        return train, dev
    except ValueError as e:
        logging.warning("era x silver_tier stratification failed (%s); falling back to era-only.", e)
    era_keys = [str(r.get("era")) for r in records]
    try:
        train, dev = train_test_split(
            records, test_size=dev_fraction, random_state=seed, stratify=era_keys
        )
        logging.info("Dev split: era-only-stratified (fallback), %d train / %d dev.", len(train), len(dev))
        return train, dev
    except ValueError as e:
        logging.warning("era-only stratification also failed (%s); falling back to unstratified.", e)
    train, dev = train_test_split(records, test_size=dev_fraction, random_state=seed)
    logging.info("Dev split: unstratified (fallback), %d train / %d dev.", len(train), len(dev))
    return train, dev


def to_training_example(rec: dict[str, Any]) -> dict[str, Any]:
    """Strip pass-through fields (silver_tier, era) — GLiNER's Dataset only expects id/tokenized_text/ner."""
    return {"id": rec["id"], "tokenized_text": rec["tokenized_text"], "ner": rec["ner"]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Fine-tune urchade/gliner_multi-v2.1 on BibNER-Werk silver spans."
    )
    parser.add_argument("--silver-data", type=Path, default=DEFAULT_SILVER_DATA)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--fold-index", type=int, required=True, help="0..k-1; determines seed and checkpoint name")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--base-model", type=str, default="urchade/gliner_multi-v2.1")
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="bf16",
                        help="bf16 on A100 (decided, see A5-finetuning-script-plan.md §0/§8c); "
                             "override to fp16 only if running on V100S")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--dev-split-seed", type=int, default=42,
                        help="Fixed once, shared across all --fold-index runs — not re-derived per fold")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Paper used num_steps=30000 (~5 epochs on Pile-NER's 44,889 passages, "
                             "a different-sized pool than ours) — starting point, not verified for "
                             "this dataset size, see A5-finetuning-gliner-actual.md 'Still open'")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Encoder LR — paper §3.2/Table 7 lr_encoder. Maps to the base "
                             "TrainingArguments field, NOT gliner's others_lr (verified by tracing "
                             "create_optimizer(), see A5-finetuning-gliner-actual.md #6)")
    parser.add_argument("--others-learning-rate", type=float, default=5e-5,
                        help="New-layers LR — paper §3.2/Table 7 lr_others. Maps to gliner's others_lr field")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Paper §3.2/Table 7")
    parser.add_argument("--batch-size", type=int, default=8, help="Paper §3.2/Table 7")
    parser.add_argument("--oversample-tier2-factor", type=float, default=3.0,
                        help="~3-4x per A5-person-imbalance.md §2.1; 1.0 disables oversampling")
    parser.add_argument("--focal-loss-alpha", type=float, default=0.25,
                        help="GLiNER's built-in focal loss, global (not per-label — see "
                             "A5-person-imbalance.md §2.3a); worth enabling as general practice regardless")
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: models/gliner_fold{fold-index}/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train on a ~50-record subset for a fast smoke test, per A5-finetuning-script-plan.md §7")
    args = parser.parse_args()

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"gliner_fold{args.fold_index}")
    seed = args.seed_base + args.fold_index

    logging.info("Loading silver data from %s", args.silver_data)
    silver_records = read_jsonl(args.silver_data)
    validate_labels(silver_records)
    leakage_guard(silver_records, args.gold, args.provenance)

    train_records, dev_records = stratified_dev_split(silver_records, args.dev_fraction, args.dev_split_seed)
    train_records = oversample_tier2(train_records, args.oversample_tier2_factor)

    if args.dry_run:
        train_records = train_records[:50]
        dev_records = dev_records[:10]
        logging.info("--dry-run: truncated to %d train / %d dev records.", len(train_records), len(dev_records))

    train_examples = [to_training_example(r) for r in train_records]
    dev_examples = [to_training_example(r) for r in dev_records]

    # Imports deferred past this point so --dry-run's data-prep logic (leakage guard, split,
    # oversampling, label validation) is testable without gliner/torch installed or a GPU present.
    from gliner import GLiNER  # type: ignore[import-untyped]
    from gliner.training import Trainer, TrainingArguments  # type: ignore[import-untyped]

    logging.info("Loading base model %s", args.base_model)
    model = GLiNER.from_pretrained(args.base_model)
    data_collator = model.data_collator_class(
        model.config, data_processor=model.data_processor, prepare_labels=True
    )

    # Negative-type sampling: config.max_neg_type_ratio (data-level, GLiNERConfig) is the paper's
    # Table 5 ablation knob ("50% negative-entity-type sampling most effective"), NOT
    # TrainingArguments.negatives/.masking (a separate, loss-level mechanism — see
    # A5-negative-sampling.md). Package default (1) already approximates the paper's optimum in
    # expectation; log it so a checkpoint that overrides this default doesn't go unnoticed.
    logging.info("model.config.max_neg_type_ratio = %s (paper's ablation optimum: ~1)",
                 getattr(model.config, "max_neg_type_ratio", "MISSING — gliner API may have changed"))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        others_lr=args.others_learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
        seed=seed,
        max_steps=5 if args.dry_run else -1,
        # Dev split (era x silver_tier-stratified) exists for best-epoch checkpoint selection
        # (see module docstring) — without these four, eval_dataset above is computed but never
        # used, and save_pretrained() below just saves whatever epoch happened to run last.
        # No compute_metrics is wired in, so eval_loss is the only metric available; matches the
        # dev split's stated purpose (checkpoint selection) without conflating it with the
        # span-F1 evaluation eval_ner_gold.py does separately against held-out gold.
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=dev_examples,
        data_collator=data_collator,
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    logging.info("Saved checkpoint -> %s", output_dir)


if __name__ == "__main__":
    main()
