#!/usr/bin/env python3
# Purpose:      B6 (XLM-R side) — generate a `models/xlmr_fold{i}/` checkpoint's
#               predictions on its paired gold fold (`data/eval/folds_k10.json`),
#               in the `{"id","spans"}` schema `eval_ner_gold.py score` expects.
#               XLM-R-equivalent of b6_generate_gliner_predictions.py: same 1:1
#               (training-replicate, gold-fold) pairing convention, adapted for
#               BIO sequence-tagging output instead of GLiNER's native span API.
#               The fold-index -> gold-fold pairing is a bookkeeping convention,
#               not a leakage requirement — train_xlmr_silver.py excludes all 395
#               gold records from every fold's training regardless of
#               --fold-index (see its module docstring), so any checkpoint could
#               validly score against any gold fold; pairing fold N with gold
#               fold N just gives one clean, reproducible assignment.
# Usage:        python3 scripts/py/generate_xlmr_predictions.py --fold-index 0
#                   [--checkpoint-dir PATH] [--gold PATH] [--folds PATH]
#                   [--max-length INT] [--device STR] [--output PATH]
#               Run once per fold (0..9) against models/xlmr_fold{i}/, matching
#               the pairing eval_ner_gold.py's score/aggregate subcommands assume.
# Inputs:       models/xlmr_fold{fold-index}/ (checkpoint via train_xlmr_silver.py,
#                   save_pretrained model + tokenizer)
#               data/gold/gold_reconciled.jsonl, data/eval/folds_k10.json
# Outputs:      data/eval/predictions_xlmr_fold{fold-index}.jsonl
#                   ({"id": doc_id, "spans": [[start,end,label], ...]})
# Dependencies: torch, transformers
# Assumptions:  Word tokenization for inference must match prepare_bio_dataset.py's
#                   word-boundary convention (WORD_PATTERN regex) exactly, since
#                   predicted spans are reconstructed as character offsets from
#                   word boundaries — duplicated here rather than imported, same
#                   "no gliner/no cross-script data-prep dependency" posture as
#                   prepare_bio_dataset.py itself. Predictions use the "label
#                   only the first subword" alignment (tokenizer.word_ids()), the
#                   inverse of train_xlmr_silver.py's tokenize_and_align_labels()
#                   — duplicated from spotcheck_xlmr_predictions.py's
#                   align_word_predictions()/predict_word_tags() rather than
#                   imported, to avoid a script-importing-a-script chain (that
#                   script itself imports from train_xlmr_silver.py already).
#                   ID2LABEL IS imported from train_xlmr_silver.py, not
#                   duplicated — same deliberate exception as
#                   spotcheck_xlmr_predictions.py, since this must match the
#                   trained label vocabulary bit-for-bit, not just approximate
#                   it. BIO decoding is lenient: an I-LABEL tag with no matching
#                   open span (none open, or a different label open) starts a
#                   new span rather than being dropped — model output at
#                   inference time isn't guaranteed to respect strict BIO
#                   grammar the way training data does by construction; the
#                   orphan count is logged as a diagnostic, not treated as an
#                   error. The model-inference call itself isn't unit tested,
#                   same posture as spotcheck_xlmr_predictions.py's tests.
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"
DEFAULT_FOLDS = ROOT / "data" / "eval" / "folds_k10.json"
DEFAULT_CHECKPOINT_ROOT = ROOT / "models"

sys.path.insert(0, str(ROOT / "scripts" / "py"))
from train_xlmr_silver import ID2LABEL  # noqa: E402

# Matches prepare_bio_dataset.py's WORD_PATTERN / GLiNER's WordsSplitter — must
# stay in sync, since predicted spans are reconstructed from these word
# boundaries.
WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S", re.UNICODE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Stream a JSONL file into a list of dicts, one per non-empty line."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def tokenize_words(text: str) -> list[tuple[str, int, int]]:
    """Split text into (word, char_start, char_end) triples via the shared word-boundary regex."""
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]


def select_fold_records(
    gold_records: list[dict[str, Any]], folds: dict[str, Any], fold_index: int
) -> list[dict[str, Any]]:
    """Return gold records whose folds.json assignment matches fold_index."""
    fold_ids = {int(doc_id) for doc_id, info in folds.items() if info["fold"] == fold_index}
    selected = [rec for rec in gold_records if rec["id"] in fold_ids]
    if not selected:
        raise ValueError(f"No gold records assigned to fold {fold_index} in folds file.")
    return selected


def decode_bio_spans(
    words_with_offsets: list[tuple[str, int, int]], tags: list[str]
) -> tuple[list[list[Any]], int]:
    """Merge word-level BIO tags into char-offset span triples.

    Standard "B-LABEL starts, I-LABEL continues" merge. Decoded leniently: an
    I-LABEL tag that doesn't continue a matching open span (none open, or a
    different label open) starts a new span of that label instead of being
    dropped, since model predictions aren't guaranteed to respect strict BIO
    grammar the way training data does by construction (see module Assumptions).

    Args:
        words_with_offsets: (word, char_start, char_end) triples, end exclusive.
        tags: Predicted BIO tag per word, same length/order as words_with_offsets.

    Returns:
        (spans, orphan_count) — spans as [start, end, label] triples (Doccano
        convention, end exclusive, char offsets); orphan_count is how many I-
        tags needed the lenient fallback, a diagnostic for decode quality.
    """
    spans: list[list[Any]] = []
    orphan_count = 0
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def close_current() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is not None:
            spans.append([current_start, current_end, current_label])
        current_label, current_start, current_end = None, None, None

    for (word, w_start, w_end), tag in zip(words_with_offsets, tags):
        if tag == "O":
            close_current()
        elif tag.startswith("B-"):
            close_current()
            current_label, current_start, current_end = tag[2:], w_start, w_end
        elif tag.startswith("I-"):
            label = tag[2:]
            if current_label == label:
                current_end = w_end
            else:
                close_current()
                current_label, current_start, current_end = label, w_start, w_end
                orphan_count += 1
        else:
            raise ValueError(f"Unexpected BIO tag {tag!r} for word {word!r} — not in ID2LABEL's vocabulary")
    close_current()
    return spans, orphan_count


def align_word_predictions(word_ids: list[int | None], pred_ids: list[int]) -> list[str]:
    """First-subword-only alignment, the inverse of tokenize_and_align_labels()'s label assignment.

    Duplicated from spotcheck_xlmr_predictions.py (module Assumptions) — mirrors the
    training-time convention (first subword occurrence per word_ids() value).
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


def predict_spans(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerFast",
    text: str,
    max_length: int,
    device: str,
) -> tuple[list[list[Any]], int]:
    """Run one record's raw text through the model, return (spans, orphan_count)."""
    words_with_offsets = tokenize_words(text)
    if not words_with_offsets:
        return [], 0
    words = [word for word, _, _ in words_with_offsets]
    pred_tags = predict_word_tags(model, tokenizer, words, max_length, device)
    return decode_bio_spans(words_with_offsets, pred_tags)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate XLM-R fine-tuned (BIO) predictions on gold, for eval_ner_gold.py score."
    )
    parser.add_argument("--fold-index", type=int, required=True, help="0..9, matches models/xlmr_fold{i}/")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Default: models/xlmr_fold{fold-index}/")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--max-length", type=int, default=256,
                        help="Must match the value train_xlmr_silver.py actually used for this checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Default: cuda if available, else cpu")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: data/eval/predictions_xlmr_fold{fold-index}.jsonl")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or (DEFAULT_CHECKPOINT_ROOT / f"xlmr_fold{args.fold_index}")
    output = args.output or (ROOT / "data" / "eval" / f"predictions_xlmr_fold{args.fold_index}.jsonl")

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_dir} — run this on the server where "
            "train_xlmr_silver.py's output actually lives, not this machine."
        )

    logging.info("Loading gold from %s", args.gold)
    gold_records = read_jsonl(args.gold)
    folds = json.loads(args.folds.read_text())
    fold_records = select_fold_records(gold_records, folds, args.fold_index)
    logging.info("Fold %d: %d gold records to predict on.", args.fold_index, len(fold_records))

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading XLM-R checkpoint from %s", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))  # type: ignore[no-untyped-call]
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_dir))
    model.eval()
    model.to(device)

    total_orphans = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for rec in fold_records:
            spans, orphan_count = predict_spans(model, tokenizer, rec["text"], args.max_length, device)
            total_orphans += orphan_count
            f.write(json.dumps({"id": rec["id"], "spans": spans}, ensure_ascii=False) + "\n")

    logging.info("Wrote %d predictions -> %s", len(fold_records), output)
    if total_orphans:
        logging.info(
            "%d orphan I- tag(s) across this fold's predictions (lenient BIO decode fallback, "
            "see module Assumptions) — not an error, but worth a glance if the count is large "
            "relative to %d records.", total_orphans, len(fold_records),
        )


if __name__ == "__main__":
    main()
