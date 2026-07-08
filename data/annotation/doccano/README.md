# Doccano exports

`export-20260623-1110/` is the **live/canonical** annotation round — the 395-record gold set used for IAA (`docs/inter-annotator-agreement.md` §1) and adjudication (`docs/adjudication-method.md`). All current scripts default to it.

`export-20260516-1808/` (dry-run) and `export-20260608-1221/` (partial round) are earlier, historical rounds kept for provenance — not live, but still read by `sr08_add_meta_to_export.py`, `sr08_verify_meta_export.py`, `sr08_count_comments.py`, and `sr08_make_meta_sample.py`.

`adjucated-20260707-1557.jsonl` is the PI's completed Doccano export for all 395 docs from the live round (the actual final-decision file — see `docs/adjudication-method.md` §3–4). `adjudiced-blind-20260707-1914.jsonl` is the blind re-adjudication comparison sample (`docs/adjudication-method.md` §5, `sr08_compare_blind_sample.py`).
