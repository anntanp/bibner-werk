# Purpose:      Tests for scripts/py/prepare_gliner_dataset.py — the Doccano
#               char-offset -> GLiNER word-indexed conversion.
# Usage:        pytest tests/test_prepare_gliner_dataset.py
# Dependencies: pytest, gliner

from __future__ import annotations

import json
from pathlib import Path

from gliner.data_processing.tokenizer import WordsSplitter

from prepare_gliner_dataset import convert, read_jsonl, to_gliner_example, write_jsonl

SPLITTER = WordsSplitter(splitter_type="whitespace")

# Real gold-adjudication record (sr08_adjudication_import_en_export-20260623-1110.jsonl,
# line 45) — hyphenated PERSON name, colon-delimited TITLE, slash-delimited trailing PERSON.
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

# Real gold-adjudication text with no space before '/' (Early Modern German
# convention) — the case that motivated verifying WordsSplitter's regex at all.
DIASKEPSIS_TEXT = (
    "Diaskepsis Theologica, Das ist: Christliche Leichpredigt/ In welcher der 42. "
    "Psalm bey Volckreicher Versamlung"
)
_DIASKEPSIS_TITLE_START = DIASKEPSIS_TEXT.find("Christliche")
_DIASKEPSIS_TITLE_END = DIASKEPSIS_TEXT.find("Leichpredigt") + len("Leichpredigt")
DIASKEPSIS_SPANS = [(_DIASKEPSIS_TITLE_START, _DIASKEPSIS_TITLE_END, "TITLE")]  # ends right before '/'


def reconstruct_char_span(
    text: str, word_start: int, word_end: int
) -> tuple[int, int]:
    """Decode a word-index span back to char offsets via an independent tokenization."""
    tokens = list(SPLITTER(text))
    return tokens[word_start][1], tokens[word_end][2]


def test_real_example_round_trips_to_original_char_offsets() -> None:
    example, bad = to_gliner_example("t1", SAINTE_PALLAYE_TEXT, SAINTE_PALLAYE_SPANS, SPLITTER)

    assert bad is None
    assert example is not None
    assert len(example["ner"]) == len(SAINTE_PALLAYE_SPANS)

    for (c_start, c_end, label), (w_start, w_end, ner_label) in zip(
        SAINTE_PALLAYE_SPANS, example["ner"]
    ):
        assert ner_label == label
        assert reconstruct_char_span(SAINTE_PALLAYE_TEXT, w_start, w_end) == (c_start, c_end)


def test_slash_glued_punctuation_aligns_without_leading_whitespace() -> None:
    example, bad = to_gliner_example("t2", DIASKEPSIS_TEXT, DIASKEPSIS_SPANS, SPLITTER)

    assert bad is None
    assert example is not None
    (w_start, w_end, label) = example["ner"][0]
    assert label == "TITLE"
    assert reconstruct_char_span(DIASKEPSIS_TEXT, w_start, w_end) == (
        _DIASKEPSIS_TITLE_START,
        _DIASKEPSIS_TITLE_END,
    )
    # The token immediately after the span must be the bare '/', not glued to a word.
    tokens = list(SPLITTER(DIASKEPSIS_TEXT))
    assert tokens[w_end + 1][0] == "/"


def test_misaligned_span_is_reported_not_raised() -> None:
    # Shift the TITLE end offset by 1 char, into the middle of "Leichpredigt".
    shifted_end = _DIASKEPSIS_TITLE_END - 1
    bad_spans = [(_DIASKEPSIS_TITLE_START, shifted_end, "TITLE")]
    example, bad = to_gliner_example("t3", DIASKEPSIS_TEXT, bad_spans, SPLITTER)

    assert example is None
    assert bad is not None
    assert bad["id"] == "t3"
    assert bad["bad_spans"] == [[_DIASKEPSIS_TITLE_START, shifted_end, "TITLE"]]


def test_zero_span_record_is_kept_as_a_negative_example() -> None:
    example, bad = to_gliner_example("t4", "Some untitled fragment", [], SPLITTER)

    assert bad is None
    assert example is not None
    assert example["ner"] == []
    assert example["tokenized_text"] == ["Some", "untitled", "fragment"]


def test_read_jsonl_streams_records_in_order(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text(
        '{"id": "a", "text": "x", "label": []}\n'
        '{"id": "b", "text": "y", "label": []}\n',
        encoding="utf-8",
    )

    records = list(read_jsonl(path))

    assert [r["id"] for r in records] == ["a", "b"]


def test_convert_excludes_misaligned_and_assigns_index_id_when_missing(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    lines = [
        {"text": SAINTE_PALLAYE_TEXT, "label": [list(s) for s in SAINTE_PALLAYE_SPANS]},
        {"text": DIASKEPSIS_TEXT, "label": [[_DIASKEPSIS_TITLE_START, _DIASKEPSIS_TITLE_END - 1, "TITLE"]]},  # misaligned, excluded
    ]
    input_path.write_text(
        "\n".join(json.dumps(rec) for rec in lines) + "\n", encoding="utf-8"
    )

    examples, misaligned, label_counts = convert(input_path, "whitespace")

    assert len(examples) == 1
    assert examples[0]["id"] == 0  # no "id" key in input -> falls back to line index
    assert len(misaligned) == 1
    assert label_counts == {"PERSON": 2, "TITLE": 1}

    write_jsonl(examples, output_path)
    written = list(read_jsonl(output_path))
    assert len(written) == 1
