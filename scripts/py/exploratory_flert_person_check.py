#!/usr/bin/env python3
# Purpose:      Exploratory, one-off check: does flair/ner-german-large (already used
#               elsewhere in this pipeline for f_resp_person/org sub-classification,
#               build_silver_spans.py's compute_resp_flags()) correctly detect the
#               specific PERSON spans GLiNER's fine-tuned checkpoint missed (B6 gold-eval,
#               B6-gliner-eval.md), for both short/abbreviated names and long
#               honorific-laden royal titulature? Curiosity check, not registered as a
#               pipeline script -- not in scripts/README.md, no tests.
#
#               Revision: feeds FLERT the FULL gold record text each false-negative span
#               came from (not the isolated span string) -- an isolated span is not a
#               fair comparison, since GLiNER's fine-tuned model saw the full catalog
#               string too. doc_id/text pulled directly from data/gold/gold_reconciled.jsonl
#               via the doc_ids recorded in data/processed/b6_person_error_analysis.csv.
# Usage:        python3 scripts/py/exploratory_flert_person_check.py
# Dependencies: flair (model already cached locally, no network needed)
from __future__ import annotations

import json
from pathlib import Path

from flair.data import Sentence
from flair.models import SequenceTagger

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD = ROOT / "data" / "gold" / "gold_reconciled.jsonl"

# (kind, doc_id, expected gold PERSON span) -- doc_ids and spans taken from
# b6_person_error_analysis.csv's false-negative rows for the same examples discussed
# in the isolated-span version of this check.
EXAMPLES = [
    ("short", 229, "M. Hess"),
    ("short", 11, "W. Slaje"),
    ("short", 148, "W. Dürig"),
    ("short", 384, "Büsching"),
    ("short", 29, "C. Höfler"),
    ("long", 36, "Georg der Dritte, von Gottes Gnaden König von Groß-Britannien, Franckreich und "
                 "Irrland, Beschützer des Glaubens, Hertzog zu Braunschweig und Lüneburg"),
    ("long", 68, "Friedrich August, von Gottes Gnaden, König in Pohlen, Groß-Hertzog in Litthauen, "
                 "Reussen, Preussen, Mazovien, Samogitien, Kyovien, Vollhinien, Podolien, "
                 "Podlachien, Liefland, Smolensco, Severien und Zchernicovien, [et]c"),
    ("long", 282, "Ioannes Avenarius, Past. und Superint. des Hochgräfl. Gemeinschafftl. Consistorii "
                  "Assess. Prim. des Gymnasii Illustris Inspector und Theol. Profess. Publicus"),
    ("long", 220, "D. Franz Dominicus Häberlins, Herzogl. Braunschweig-Lüneburg. Geheimen-Justitz-Raths, "
                  "ersten Lehrers der Rechte, des Teutschen Staats-Rechtes und der Geschichte auf der "
                  "Julius-Carls-Universität zu Helmstedt"),
    ("long", 147, "Vnser von Gotts gnaden Johans Friedrichen Hertzogen zu Sachssen vnd Churfürsten etc. "
                  "Burggrauen zu Magdeburg Vnd Philipsen Landgrauen zu Hessen, Grauen zu CatzenElnbogen"),
]


def load_gold_texts(doc_ids: set[int]) -> dict[int, str]:
    texts: dict[int, str] = {}
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] in doc_ids:
                texts[rec["id"]] = rec["text"]
    return texts


def main() -> None:
    doc_ids = {doc_id for _, doc_id, _ in EXAMPLES}
    texts = load_gold_texts(doc_ids)

    print("Loading flair/ner-german-large (cached locally)...")
    tagger = SequenceTagger.load("flair/ner-german-large")

    for kind, doc_id, gold_span in EXAMPLES:
        text = texts[doc_id]
        sentence = Sentence(text)
        tagger.predict(sentence)
        spans = [(span.text, span.get_label("ner").value) for span in sentence.get_spans("ner")]
        print(f"\n[{kind}] doc_id={doc_id}")
        print(f"  full text: {text!r}")
        print(f"  gold PERSON span: {gold_span!r}")
        print(f"  FLERT spans: {spans if spans else '(none found)'}")


if __name__ == "__main__":
    main()
