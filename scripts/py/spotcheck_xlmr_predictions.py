#!/usr/bin/env python3
# Purpose:      Qualitative spot-check of a trained BIO checkpoint (train_xlmr_silver.py's output)
#               against SILVER BIO tags (not the human-adjudicated gold set — see Assumptions) on N
#               sampled dev-split records — a quick sanity check on whether a model's predictions
#               look right, complementing the loss-curve numbers alone. Written after
#               A5-finetuning-xlmr-workflow.md §6.2 flagged fold-0's unusually low train/eval loss
#               (0.00462/0.00442) as worth a deliberate look before firing the remaining k-fold
#               replicates on the same recipe. This is a manual qualitative read, not a formal
#               metric — that's B6 (not yet built), which evaluates against held-out *gold* folds.
# Usage:        python3 scripts/py/spotcheck_xlmr_predictions.py --checkpoint models/xlmr_fold0
#                   [--bio-data PATH] [--n-samples INT] [--sample-seed INT]
#                   [--dev-fraction FLOAT] [--dev-split-seed INT] [--max-length INT] [--device STR]
# Inputs:       data/silver/bio_dataset_silver.jsonl (word-level BIO schema, same file
#                   train_xlmr_silver.py trains on)
#               A checkpoint dir saved by train_xlmr_silver.py (model + tokenizer via
#                   save_pretrained)
# Outputs:      Printed word-by-word silver-vs-predicted comparison to stdout, one block per sampled
#                   record, plus an aggregate mismatch-rate summary — no file output; redirect if a
#                   saved copy is wanted (e.g. `> spotcheck_fold0.txt`)
# Dependencies: torch, transformers
# Assumptions:  **The reference tags here are SILVER (ISBD-heuristic-derived), not the true
#                   395-record human-adjudicated GOLD set (data/gold/gold_reconciled.jsonl) — those
#                   are disjoint datasets.** A "mismatch" against silver means the model disagrees
#                   with the heuristic labeling pipeline, which is not automatically the same as
#                   disagreeing with bibliographic ground truth; the silver heuristic itself can be
#                   wrong on hard cases (see A5-xlmr-spotcheck.md for a worked example). Only B6's
#                   eval against gold folds measures correctness against actual ground truth — this
#                   script is a fast qualitative read of self-consistency with the training signal,
#                   not a substitute for that. Reproduces the exact dev split train_xlmr_silver.py
#                   used at training time (--dev-fraction/--dev-split-seed must match what that run
#                   actually used — defaults are 0.1/42 for both scripts) by importing read_jsonl/
#                   stratified_dev_split/ID2LABEL directly from train_xlmr_silver.py rather than
#                   reimplementing them — a deliberate exception to the "standalone scripts"
#                   convention (scripts/README.md), since correctness here depends on bit-for-bit
#                   reproducing that split, not just approximating it (same rationale as
#                   verify_resp_classifier.py's import of build_silver_spans.py). Predictions use
#                   the same "label only the first subword" word-alignment convention as training
#                   (tokenizer.word_ids()) — the inverse direction of tokenize_and_align_labels().
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BIO_DATA = ROOT / "data" / "silver" / "bio_dataset_silver.jsonl"

sys.path.insert(0, str(ROOT / "scripts" / "py"))
from train_xlmr_silver import ID2LABEL, read_jsonl, stratified_dev_split  # noqa: E402


def sample_dev_records(
    bio_data: Path,
    dev_fraction: float,
    dev_split_seed: int,
    n_samples: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Reproduce train_xlmr_silver.py's exact dev split, then draw a random subset for review."""
    records = read_jsonl(bio_data)
    _, dev_records = stratified_dev_split(records, dev_fraction, dev_split_seed)
    rng = random.Random(sample_seed)
    return rng.sample(dev_records, min(n_samples, len(dev_records)))


def align_word_predictions(word_ids: list[int | None], pred_ids: list[int]) -> list[str]:
    """First-subword-only alignment, the inverse of tokenize_and_align_labels()'s label assignment.

    Mirrors the training-time convention (first token occurrence per word_ids() value), so
    predictions are compared against silver tags at the same word-level granularity.
    """
    num_words = max((w for w in word_ids if w is not None), default=-1) + 1
    tags = ["O"] * num_words
    seen: set[int] = set()
    for tok_idx, w_idx in enumerate(word_ids):
        if w_idx is None or w_idx in seen:
            continue
        seen.add(w_idx)
        tags[w_idx] = ID2LABEL[pred_ids[tok_idx]]
    return tags


def format_comparison(
    record_id: object, words: list[str], silver_tags: list[str], pred_tags: list[str]
) -> tuple[str, int]:
    """Render one record's word-by-word silver-vs-predicted comparison.

    Labeled "silver=", not "gold=" — these reference tags come from the ISBD-heuristic
    silver-labeling pipeline (bio_dataset_silver.jsonl), not the human-adjudicated gold set. A
    mismatch means disagreement with the training signal, not necessarily with ground truth.

    Returns:
        (report_text, mismatch_count).
    """
    lines = [f"{'=' * 80}", f"id: {record_id}"]
    mismatches = 0
    for word, silver, pred in zip(words, silver_tags, pred_tags):
        flag = "" if silver == pred else "  <-- MISMATCH"
        mismatches += bool(flag)
        lines.append(f"{word:30s} silver={silver:16s} pred={pred:16s}{flag}")
    lines.append(f"-> {mismatches}/{len(words)} word-level mismatches")
    return "\n".join(lines), mismatches


def predict_word_tags(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerFast",
    words: list[str],
    max_length: int,
    device: str,
) -> list[str]:
    """Run one record's words through the model and align predictions back to word-level tags."""
    import torch

    encoded = tokenizer(
        words, is_split_into_words=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    word_ids = tokenizer(
        words, is_split_into_words=True, truncation=True, max_length=max_length
    ).word_ids()
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        pred_ids: list[int] = model(**encoded).logits[0].argmax(-1).tolist()
    return align_word_predictions(word_ids, pred_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spot-check a train_xlmr_silver.py checkpoint's predictions against SILVER "
        "(not gold) BIO tags on sampled dev-split records."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="e.g. models/xlmr_fold0")
    parser.add_argument("--bio-data", type=Path, default=DEFAULT_BIO_DATA)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Different seeds draw different samples for repeat spot-checks")
    parser.add_argument("--dev-fraction", type=float, default=0.1,
                        help="Must match the value train_xlmr_silver.py actually used for this checkpoint")
    parser.add_argument("--dev-split-seed", type=int, default=42,
                        help="Must match the value train_xlmr_silver.py actually used for this checkpoint")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default=None,
                        help="Default: cuda if available, else cpu")
    args = parser.parse_args()

    sample = sample_dev_records(
        args.bio_data, args.dev_fraction, args.dev_split_seed, args.n_samples, args.sample_seed
    )

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))  # type: ignore[no-untyped-call]
    model = AutoModelForTokenClassification.from_pretrained(str(args.checkpoint))
    model.eval()
    model.to(device)

    total_mismatches = 0
    total_words = 0
    for rec in sample:
        pred_tags = predict_word_tags(model, tokenizer, rec["tokens"], args.max_length, device)
        report, mismatches = format_comparison(rec["id"], rec["tokens"], rec["tags"], pred_tags)
        print(report)
        total_mismatches += mismatches
        total_words += len(rec["tokens"])

    print(f"{'=' * 80}\nOverall: {total_mismatches}/{total_words} word-level mismatches "
          f"across {len(sample)} sampled records ({100 * total_mismatches / total_words:.2f}%)")


if __name__ == "__main__":
    main()
