#!/usr/bin/env python3
# Purpose:      A5 — fine-tune xlm-roberta-large as a BIO sequence tagger on
#               silver-labeled BibNER-Werk spans (TITLE, OTHER_TITLE, PERSON).
#               Second half of A5 (train_gliner_silver.py was first), same
#               k-run semantics: one of k replicate runs (--fold-index) for
#               cross-fold variance estimation, per notes/fiz/bibner-werk/
#               A5-finetuning-script-plan.md §4. Training data is identical
#               across all k runs (gold IDs already excluded regardless of
#               fold); only the seed differs.
#               See A5-finetuning-script-plan.md §3b/§5,
#               A5-person-imbalance.md §3.1/§3.2 (why XLM-R uses class-weighted
#               focal loss instead of GLiNER's tier-2 oversampling — the two
#               systems' imbalance handling is decoupled, not required to match).
# Usage:        python3 scripts/py/train_xlmr_silver.py --fold-index 0
#                   [--bio-data PATH] [--gold PATH] [--provenance PATH]
#                   [--seed-base INT] [--model-name STR] [--precision {fp16,bf16}]
#                   [--dev-fraction FLOAT] [--dev-split-seed INT] [--max-length INT]
#                   [--epochs INT] [--learning-rate FLOAT] [--batch-size INT]
#                   [--warmup-ratio FLOAT] [--weight-decay FLOAT]
#                   [--focal-gamma FLOAT] [--class-weights {auto,none}]
#                   [--output-dir PATH] [--dry-run]
# Inputs:       data/silver/bio_dataset_silver.jsonl (word-level BIO schema:
#                   {"id","tokens","tags","silver_tier","era"} — produced by
#                   prepare_bio_dataset.py from data/silver/silver_spans.jsonl;
#                   §2/§9's standalone-vs-inline conversion question resolved in
#                   favor of standalone, see prepare_bio_dataset.py's header)
#               data/gold/gold_reconciled.jsonl (leakage guard)
#               data/provenance/bibner-data-provenance.parquet (leakage guard,
#                   gold obj_id resolution)
# Outputs:      models/xlmr_fold{fold-index}/ (HF Trainer checkpoint, via
#                   save_pretrained, tokenizer included)
# Dependencies: transformers, torch, sentencepiece, pyarrow, scikit-learn
# Assumptions:  Word-level BIO tags are aligned to XLM-R subwords via the
#                   standard "label only the first subword" HF convention
#                   (tokenizer.word_ids(), label_all_tokens=False) — non-first
#                   subwords and special tokens get label -100, ignored by the
#                   loss. Dev split (90/10, era x silver_tier-stratified) is for
#                   best-epoch checkpoint selection only; the gold folds (B6)
#                   remain the untouched held-out test set — same policy as
#                   train_gliner_silver.py, duplicated here rather than shared
#                   (see A5-finetuning-script-plan.md §1: the two scripts share
#                   almost no code, so a shared module would mostly be
#                   plumbing). Class weights are computed per BIO tag (7-way),
#                   not per entity type, and from the actual training split
#                   actually used (post any --dry-run truncation), not the raw
#                   corpus — see A5-person-imbalance.md §3.1. Hyperparameters
#                   (epochs, LR, batch size, warmup) are standard XLM-R-large
#                   token-classification starting points, NOT tuned for this
#                   dataset — same "flagged default, not a verified value"
#                   posture as train_gliner_silver.py, pending PI approval
#                   (A5-finetuning-script-plan.md §0).
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BIO_DATA = ROOT / "data" / "silver" / "bio_dataset_silver.jsonl"
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_PROVENANCE = ROOT / "data" / "provenance" / "bibner-data-provenance.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "models"

# Matches prepare_bio_dataset.py's BIO_TAGS — must stay in sync.
BIO_TAGS = ["O", "B-TITLE", "I-TITLE", "B-OTHER_TITLE", "I-OTHER_TITLE", "B-PERSON", "I-PERSON"]
LABEL2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
ID2LABEL = {i: tag for tag, i in LABEL2ID.items()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Stream a JSONL file into a list of dicts, one per non-empty line."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def validate_bio_tags(records: list[dict[str, Any]]) -> None:
    """Fail loudly if any record uses a tag outside the fixed 7-tag BIO_TAGS vocabulary.

    Catches schema drift (e.g. an un-rebuilt BIO file, or a stray tag from an
    unrelated pipeline change) before it silently trains a token-classification
    head with the wrong label vocabulary.
    """
    seen = {tag for rec in records for tag in rec["tags"]}
    unexpected = seen - set(BIO_TAGS)
    if unexpected:
        raise ValueError(
            f"Training data contains BIO tags outside {BIO_TAGS}: {unexpected}. "
            "Refusing to train — check --bio-data matches the expected schema."
        )


def leakage_guard(bio_records: list[dict[str, Any]], gold_path: Path, provenance_path: Path) -> None:
    """Assert no BIO record's id is a gold obj_id. Raises, does not warn.

    Same defensive re-check as train_gliner_silver.py's leakage_guard — resolves
    gold's 395 Doccano ids to obj_id via the provenance parquet, then asserts
    disjointness against the training file's id column, rather than trusting
    build_data_provenance.py's own assertion blindly (computed by a different
    script at a different time). See A5-finetuning-script-plan.md §2.
    """
    gold_ids = {rec["id"] for rec in read_jsonl(gold_path)}
    table = pq.read_table(provenance_path, columns=["doc_id", "is_gold", "obj_id"])
    prov = table.to_pylist()
    gold_obj_ids = {row["obj_id"] for row in prov if row["is_gold"] and row["doc_id"] in gold_ids}

    bio_ids = {rec["id"] for rec in bio_records}
    leaked = bio_ids & gold_obj_ids
    if leaked:
        raise AssertionError(
            f"Leakage guard failed: {len(leaked)} training record(s) share an obj_id with a gold "
            f"record: {sorted(leaked)[:5]}{'...' if len(leaked) > 5 else ''}. Training on leaked "
            "gold records would silently inflate every downstream F1 number — refusing to proceed."
        )
    logging.info(
        "Leakage guard passed: 0 of %d training records overlap with %d gold obj_ids.",
        len(bio_ids), len(gold_obj_ids),
    )


def stratified_dev_split(
    records: list[dict[str, Any]], dev_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """90/10 (by default) train/dev split, stratified by era x silver_tier.

    Single fixed split — call once, reuse across all --fold-index runs (not
    re-derived per fold). Falls back to era-only stratification, then to an
    unstratified split, if the composite key has strata too small for
    sklearn's stratified split to run — logged, not silent. Duplicated from
    train_gliner_silver.py rather than shared, see module Assumptions.
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


def compute_class_weights(train_records: list[dict[str, Any]]) -> list[float]:
    """Inverse-sqrt-frequency weight per BIO tag, normalized to mean 1.0.

    Computed from the actual training split's tag distribution (including any
    --dry-run truncation), not the raw corpus, so it reflects what training
    will actually see. Per-tag (7-way), not per entity type — O dominates even
    within TITLE/OTHER_TITLE-heavy records. See A5-person-imbalance.md §3.1.
    Kept torch-free (returns a plain list) so it's testable without torch
    installed; converted to a tensor by the caller after the deferred import.
    """
    counts = Counter(tag for rec in train_records for tag in rec["tags"])
    freqs = [max(counts.get(tag, 0), 1) for tag in BIO_TAGS]  # avoid /0 for an absent tag
    weights = [1.0 / (freq**0.5) for freq in freqs]
    mean_weight = sum(weights) / len(weights)
    normalized = [w / mean_weight for w in weights]
    logging.info(
        "Class weights (tag: count, weight): %s",
        {tag: (counts.get(tag, 0), round(w, 3)) for tag, w in zip(BIO_TAGS, normalized)},
    )
    return normalized


def tokenize_and_align_labels(
    record: dict[str, Any], tokenizer: "PreTrainedTokenizerFast", max_length: int
) -> dict[str, Any]:
    """Convert one word-level BIO record into subword input_ids/attention_mask/labels.

    Standard "label only the first subword" alignment (label_all_tokens=False)
    via the fast tokenizer's word_ids() — the canonical HF token-classification
    pattern. Non-first subwords and special tokens (CLS/SEP/PAD) get label
    -100, ignored by cross-entropy/focal loss.
    """
    label_ids = [LABEL2ID[tag] for tag in record["tags"]]
    tokenized = tokenizer(record["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
    word_ids = tokenized.word_ids()
    aligned: list[int] = []
    previous_word_idx: int | None = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(-100)
        elif word_idx != previous_word_idx:
            aligned.append(label_ids[word_idx])
        else:
            aligned.append(-100)
        previous_word_idx = word_idx
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": aligned,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Fine-tune xlm-roberta-large as a BIO tagger on BibNER-Werk silver spans."
    )
    parser.add_argument("--bio-data", type=Path, default=DEFAULT_BIO_DATA,
                        help="Produced by prepare_bio_dataset.py from data/silver/silver_spans.jsonl")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--fold-index", type=int, required=True, help="0..k-1; determines seed and checkpoint name")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="xlm-roberta-large",
                        help="Fallback: xlm-roberta-base if D3 shows per-run time > 3h, see "
                             "A5-finetuning-script-plan.md §3b")
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="bf16",
                        help="bf16 on A100 (decided, see A5-finetuning-script-plan.md §0/§8c); "
                             "override to fp16 only if running on V100S")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--dev-split-seed", type=int, default=42,
                        help="Fixed once, shared across all --fold-index runs — not re-derived per fold")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Subword sequence length cap; corpus median is ~15 BPE tokens "
                             "(jcdl-plan-202606.md §4) — generous headroom, not expected to truncate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Standard XLM-R-large token-classification starting point, NOT tuned "
                             "for this dataset — see module Assumptions / A5-finetuning-script-plan.md §0")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Kept low (large pretrained encoders are unstable at 5e-5) — starting "
                             "point, not verified for this dataset size")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Fixed constant per Lin et al. 2017's default, not tuned — see "
                             "A5-person-imbalance.md §3.1")
    parser.add_argument("--class-weights", choices=["auto", "none"], default="auto",
                        help="auto: per-BIO-tag inverse-sqrt-frequency weights from the training "
                             "split (§3.1); none: plain (unweighted) focal loss")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: models/xlmr_fold{fold-index}/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train on a ~50-record subset for a fast smoke test, mirroring "
                             "train_gliner_silver.py's --dry-run, per A5-finetuning-script-plan.md §7")
    args = parser.parse_args()

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"xlmr_fold{args.fold_index}")
    seed = args.seed_base + args.fold_index

    logging.info("Loading BIO data from %s", args.bio_data)
    bio_records = read_jsonl(args.bio_data)
    validate_bio_tags(bio_records)
    leakage_guard(bio_records, args.gold, args.provenance)

    train_records, dev_records = stratified_dev_split(bio_records, args.dev_fraction, args.dev_split_seed)

    if args.dry_run:
        train_records = train_records[:50]
        dev_records = dev_records[:10]
        logging.info("--dry-run: truncated to %d train / %d dev records.", len(train_records), len(dev_records))

    class_weights_raw = compute_class_weights(train_records) if args.class_weights == "auto" else None

    # Imports deferred past this point so --dry-run's data-prep logic (leakage guard, split,
    # tag validation, class-weight computation) is testable without torch/transformers installed
    # or a GPU present — same posture as train_gliner_silver.py.
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    class FocalLossTrainer(Trainer):
        """Trainer override applying class-weighted focal loss to the token-classification head.

        Per A5-finetuning-script-plan.md §3b / A5-person-imbalance.md §3.1: PERSON's
        severe token-level rarity is handled here via loss weighting, instead of
        GLiNER's tier-2 oversampling — the two systems' fixes are decoupled, not
        stacked (A5-person-imbalance.md §3.1's "don't combine" rule).
        """

        def __init__(
            self, *targs: Any, class_weights: "torch.Tensor | None" = None, gamma: float = 2.0, **tkwargs: Any
        ) -> None:
            super().__init__(*targs, **tkwargs)
            self.class_weights = class_weights
            self.gamma = gamma

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: "torch.Tensor | None" = None,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # (batch, seq_len, num_labels)

            active = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.size(-1))[active]
            active_labels = labels.view(-1)[active]

            weight = self.class_weights.to(active_logits.device) if self.class_weights is not None else None
            ce_loss = F.cross_entropy(active_logits, active_labels, weight=weight, reduction="none")
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** self.gamma * ce_loss).mean()

            return (loss, outputs) if return_outputs else loss

    logging.info("Loading tokenizer/model %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # type: ignore[no-untyped-call]
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(BIO_TAGS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_examples = [tokenize_and_align_labels(r, tokenizer, args.max_length) for r in train_records]
    dev_examples = [tokenize_and_align_labels(r, tokenizer, args.max_length) for r in dev_records]

    class_weights = (
        torch.tensor(class_weights_raw, dtype=torch.float32) if class_weights_raw is not None else None
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        seed=seed,
        max_steps=5 if args.dry_run else -1,
        report_to=[],
        # eval_strategy/save_strategy/load_best_model_at_end wire the dev split (built for
        # best-epoch checkpoint selection, see module Assumptions) into an actual selection
        # decision — omitting these was a real bug caught in train_gliner_silver.py's first
        # real run (fold-0 saved its unconditional end-of-training state instead of the
        # dev-best one; A5-finetuning-gliner-actual.md #16). metric_for_best_model="eval_loss"
        # since no compute_metrics is wired in here either — dev-loss is the only metric the
        # Trainer produces; span/entity-level F1 against held-out gold folds is a separate,
        # later step (B6, eval_ner_gold.py).
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=dev_examples,
        data_collator=data_collator,
        class_weights=class_weights,
        gamma=args.focal_gamma,
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logging.info("Saved checkpoint -> %s", output_dir)


if __name__ == "__main__":
    main()
