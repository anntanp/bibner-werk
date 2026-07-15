# Purpose:      Tests for scripts/py/generate_xlmr_predictions.py's pure-logic pieces —
#               word tokenization, BIO-to-span decoding, and fold selection. No model/GPU
#               needed; predict_word_tags() (the one function that calls a real model) is
#               untested here, same "full runs aren't unit tested" posture as
#               test_spotcheck_xlmr_predictions.py / train_xlmr_silver.py's trainer.train().
# Usage:        pytest tests/test_generate_xlmr_predictions.py
# Dependencies: pytest

from __future__ import annotations

import pytest

from generate_xlmr_predictions import decode_bio_spans, select_fold_records, tokenize_words


def test_tokenize_words_returns_word_char_offset_triples() -> None:
    result = tokenize_words("Der Titel.")

    words = [w for w, _, _ in result]
    assert words == ["Der", "Titel", "."]
    # Offsets are exclusive-end char spans into the original string.
    assert result[0] == ("Der", 0, 3)
    assert result[1] == ("Titel", 4, 9)
    assert result[2] == (".", 9, 10)


def test_decode_bio_spans_merges_b_i_into_one_span() -> None:
    words = [("Der", 0, 3), ("Titel", 4, 9)]
    tags = ["B-TITLE", "I-TITLE"]

    spans, orphan_count = decode_bio_spans(words, tags)

    assert spans == [[0, 9, "TITLE"]]
    assert orphan_count == 0


def test_decode_bio_spans_separates_adjacent_spans_of_same_label() -> None:
    words = [("A", 0, 1), ("B", 2, 3), ("C", 4, 5), ("D", 6, 7)]
    tags = ["B-TITLE", "I-TITLE", "B-TITLE", "I-TITLE"]

    spans, orphan_count = decode_bio_spans(words, tags)

    assert spans == [[0, 3, "TITLE"], [4, 7, "TITLE"]]
    assert orphan_count == 0


def test_decode_bio_spans_all_o_yields_no_spans() -> None:
    words = [("A", 0, 1), ("B", 2, 3)]
    tags = ["O", "O"]

    spans, orphan_count = decode_bio_spans(words, tags)

    assert spans == []
    assert orphan_count == 0


def test_decode_bio_spans_orphan_i_tag_with_no_open_span_starts_new_span() -> None:
    # I-PERSON with nothing open before it — not valid strict BIO, decoded leniently.
    words = [("Der", 0, 3), ("Autor", 4, 9)]
    tags = ["O", "I-PERSON"]

    spans, orphan_count = decode_bio_spans(words, tags)

    assert spans == [[4, 9, "PERSON"]]
    assert orphan_count == 1


def test_decode_bio_spans_orphan_i_tag_with_mismatched_open_label_closes_and_starts_new() -> None:
    # I-PERSON right after a B-TITLE/I-TITLE run — label mismatch, not a continuation.
    words = [("Der", 0, 3), ("Titel", 4, 9), ("Schmidt", 10, 17)]
    tags = ["B-TITLE", "I-TITLE", "I-PERSON"]

    spans, orphan_count = decode_bio_spans(words, tags)

    assert spans == [[0, 9, "TITLE"], [10, 17, "PERSON"]]
    assert orphan_count == 1


def test_decode_bio_spans_multiple_labels_in_sequence() -> None:
    words = [("Titel", 0, 5), (":", 5, 6), ("von", 7, 10), ("Schmidt", 11, 18)]
    tags = ["B-TITLE", "O", "O", "B-PERSON"]

    spans, orphan_count = decode_bio_spans(words, tags)

    assert spans == [[0, 5, "TITLE"], [11, 18, "PERSON"]]
    assert orphan_count == 0


def test_decode_bio_spans_span_open_at_end_of_sequence_is_closed() -> None:
    words = [("A", 0, 1), ("B", 2, 3)]
    tags = ["B-PERSON", "I-PERSON"]

    spans, orphan_count = decode_bio_spans(words, tags)

    assert spans == [[0, 3, "PERSON"]]
    assert orphan_count == 0


def test_select_fold_records_filters_by_fold_assignment() -> None:
    gold_records = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]
    folds = {
        "1": {"fold": 0, "era": "modern", "silver_tier": 1, "flagged_small_cell": False},
        "2": {"fold": 1, "era": "modern", "silver_tier": 1, "flagged_small_cell": False},
        "3": {"fold": 0, "era": "modern", "silver_tier": 1, "flagged_small_cell": False},
    }

    selected = select_fold_records(gold_records, folds, fold_index=0)

    assert {r["id"] for r in selected} == {1, 3}


def test_select_fold_records_raises_if_fold_is_empty() -> None:
    gold_records = [{"id": 1, "text": "a"}]
    folds = {"1": {"fold": 0, "era": "modern", "silver_tier": 1, "flagged_small_cell": False}}

    with pytest.raises(ValueError, match="No gold records assigned to fold 5"):
        select_fold_records(gold_records, folds, fold_index=5)
