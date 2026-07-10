# Purpose:      Tests for scripts/py/prepare_bio_dataset.py — the Doccano
#               char-offset -> word-level BIO conversion for XLM-R fine-tuning.
# Usage:        pytest tests/test_prepare_bio_dataset.py
# Dependencies: pytest

from __future__ import annotations

import json
from pathlib import Path

from prepare_bio_dataset import convert, read_jsonl, to_bio_example, tokenize_words, write_jsonl

# Real gold-adjudication record (sr08_adjudication_import_en_export-20260623-1110.jsonl,
# line 45) — hyphenated PERSON name, colon-delimited TITLE, slash-delimited trailing PERSON.
# Same fixture text as test_prepare_gliner_dataset.py, reused here so both conversion
# scripts are exercised against the same known-tricky real-world string.
SAINTE_PALLAYE_TEXT = (
    "Sainte-Pallaye, Jean Baptiste de LaCurne de: Das Ritterwesen des Mittelalters "
    "nach seiner politischen und militärischen Verfassung / Aus dem Französischen "
    "des Herrn de la Curne de Sainte-Palaye mit Anmerkungen, Zusätzen und Vorrede "
    "von Johann Ludwig Klüber. - Nürnberg : Grattenauer Bd. 3. - 1791"
)
SAINTE_PALLAYE_SPANS = [
    (0, 43, "PERSON"),
    (45, 130, "TITLE"),
    (236, 256, "PERSON"),
]

DIASKEPSIS_TEXT = (
    "Diaskepsis Theologica, Das ist: Christliche Leichpredigt/ In welcher der 42. "
    "Psalm bey Volckreicher Versamlung"
)
_DIASKEPSIS_TITLE_START = DIASKEPSIS_TEXT.find("Christliche")
_DIASKEPSIS_TITLE_END = DIASKEPSIS_TEXT.find("Leichpredigt") + len("Leichpredigt")
DIASKEPSIS_SPANS = [(_DIASKEPSIS_TITLE_START, _DIASKEPSIS_TITLE_END, "TITLE")]


def test_real_example_produces_correct_bio_tags() -> None:
    example, bad = to_bio_example("t1", SAINTE_PALLAYE_TEXT, SAINTE_PALLAYE_SPANS)

    assert bad is None
    assert example is not None
    tokens = tokenize_words(SAINTE_PALLAYE_TEXT)
    assert example["tokens"] == [w for w, _, _ in tokens]

    tags = example["tags"]
    assert len(tags) == len(tokens)

    # First PERSON span: "Sainte-Pallaye, Jean Baptiste de LaCurne de:" — B- on the
    # first word-token, I- on every subsequent token through the closing ':'.
    first_person_word_indices = [i for i, (_, s, e) in enumerate(tokens) if s >= 0 and e <= 43]
    assert tags[first_person_word_indices[0]] == "B-PERSON"
    assert all(tags[i] == "I-PERSON" for i in first_person_word_indices[1:])

    # Everything outside the three annotated spans stays "O".
    assert tags[-1] == "O"  # trailing "1791" is outside all spans


def test_slash_glued_punctuation_aligns_without_leading_whitespace() -> None:
    example, bad = to_bio_example("t2", DIASKEPSIS_TEXT, DIASKEPSIS_SPANS)

    assert bad is None
    assert example is not None
    tokens = tokenize_words(DIASKEPSIS_TEXT)
    title_word_indices = [
        i for i, (_, s, e) in enumerate(tokens) if s >= _DIASKEPSIS_TITLE_START and e <= _DIASKEPSIS_TITLE_END
    ]
    assert example["tags"][title_word_indices[0]] == "B-TITLE"
    assert all(example["tags"][i] == "I-TITLE" for i in title_word_indices[1:])
    # The token immediately after the span must be the bare '/', tagged O, not glued to a word.
    assert tokens[title_word_indices[-1] + 1][0] == "/"
    assert example["tags"][title_word_indices[-1] + 1] == "O"


def test_misaligned_span_is_reported_not_raised() -> None:
    # Shift the TITLE end offset by 1 char, into the middle of "Leichpredigt".
    shifted_end = _DIASKEPSIS_TITLE_END - 1
    bad_spans = [(_DIASKEPSIS_TITLE_START, shifted_end, "TITLE")]
    example, bad = to_bio_example("t3", DIASKEPSIS_TEXT, bad_spans)

    assert example is None
    assert bad is not None
    assert bad["id"] == "t3"
    assert bad["bad_spans"] == [[_DIASKEPSIS_TITLE_START, shifted_end, "TITLE"]]


def test_zero_span_record_is_kept_as_all_o() -> None:
    example, bad = to_bio_example("t4", "Some untitled fragment", [])

    assert bad is None
    assert example is not None
    assert example["tokens"] == ["Some", "untitled", "fragment"]
    assert example["tags"] == ["O", "O", "O"]


def test_unexpected_label_is_reported_as_misaligned() -> None:
    # A label outside {TITLE, OTHER_TITLE, PERSON} (e.g. schema drift) must not
    # silently produce a BIO tag outside prepare_bio_dataset.py's BIO_TAGS vocabulary.
    example, bad = to_bio_example("t5", "Some fragment", [(0, 4, "UNKNOWN_LABEL")])

    assert example is None
    assert bad is not None
    assert bad["bad_spans"] == [[0, 4, "UNKNOWN_LABEL"]]


def test_passthrough_fields_carried_and_none_omitted() -> None:
    example, bad = to_bio_example(
        "t6", "Some fragment", [], passthrough={"silver_tier": 2, "era": None}
    )

    assert bad is None
    assert example is not None
    assert example["silver_tier"] == 2
    assert "era" not in example


def test_read_jsonl_streams_records_in_order(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text(
        '{"id": "a", "text": "x", "label": []}\n'
        '{"id": "b", "text": "y", "label": []}\n',
        encoding="utf-8",
    )

    records = list(read_jsonl(path))

    assert [r["id"] for r in records] == ["a", "b"]


def test_convert_excludes_misaligned_and_assigns_index_id_when_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    lines = [
        {"text": SAINTE_PALLAYE_TEXT, "label": [list(s) for s in SAINTE_PALLAYE_SPANS]},
        {"text": DIASKEPSIS_TEXT, "label": [[_DIASKEPSIS_TITLE_START, _DIASKEPSIS_TITLE_END - 1, "TITLE"]]},  # misaligned
    ]
    input_path.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n", encoding="utf-8")

    examples, misaligned, tag_counts = convert(input_path)

    assert len(examples) == 1
    assert examples[0]["id"] == 0  # no "id" key in input -> falls back to line index
    assert len(misaligned) == 1
    assert tag_counts["B-PERSON"] == 2
    assert tag_counts["B-TITLE"] == 1
    assert tag_counts["O"] > 0

    write_jsonl(examples, output_path)
    written = list(read_jsonl(output_path))
    assert len(written) == 1
