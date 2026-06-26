---
up: "[[sr08_gold-set-composition]]"
related: "[[sr08_doccano-meta]]"
created: 2026-06-23T00:00:00
updates:
  - 2026-06-25
  - 2026-06-26
in: bibner-werk
status: finalized
tags: [ner, iaa, annotation, script-plan]
---

# SR-08: IAA Computation Script Plan

## 1. Context

The IAA metric is **Krippendorff's α** (nominal distance function), as specified in `notes/fiz/bibner-werk/iaa-score-walkthrough.md` (source: Artstein & Poesio 2008, §2.6.1). α is preferred over pairwise span F1 because it (a) handles all three annotators simultaneously, (b) corrects for chance agreement, and (c) treats "no entity" records as valid O-category assignments rather than missing data. The recommended threshold is α ≥ 0.800 (Krippendorff's minimum for reliable data). This script makes α reportable for the paper.

## 2. Output file

`scripts/ner/sr08_compute_iaa.py`

## 3. Data inputs

- `--export-dir` (required): path to a doccano export directory, e.g. `data/annotation/doccano/export-20260623-1110` (latest as of 2026-06-23)
- Active annotator files in that dir (auto-discovered): `frankp.jsonl`, `gundi.jsonl`, `maria.jsonl`
- All `.jsonl` files are **read-only** — the script never writes to, moves, or modifies any file in `--export-dir`
**Skip unconditionally:** `dummy.doccano.jsonl`, `ann.jsonl` (Claude pre-annotation seed, not used for IAA), any `*.meta.jsonl`, any `sr08_*` prefixed file

**JSONL record format** (one JSON object per line):
```json
{"id": 4, "text": "Zur Frage...", "label": [[0,45,"TITLE"],[48,93,"OTHER_TITLE"]], "Comments": []}
```
Spans are `[start, end, label]` character offsets (Python slice notation).

**Label extraction and deduplication:** for each annotator and document, extract the *set* of entity types present — not individual spans. Two different TITLE spans from the same annotator count as one TITLE assignment. This ensures n_ik ≤ 1 per annotator per label, consistent with Artstein & Poesio's formula ("number of coders who assigned category k").

```python
def get_labels(record: dict) -> list[str] | None:
    """Return deduplicated label types for a reviewed record, or None if unreviewed."""
    reviewed = bool(record["label"]) or bool(record.get("Comments"))
    if not reviewed:
        return None
    types = set(span[2] for span in record["label"])  # deduplicate by type
    return list(types) if types else ["O"]
```

## 4. Algorithm

### 4.1 Unit definition

- **Item** = one document (395 records).
- **Label** per annotator per document = the set of entity types they assigned (TITLE, OTHER_TITLE, PERSON, etc.), plus one **O** label if they found no entities. A document with TITLE + PERSON contributes two label assignments from that annotator.
- A record counts as **reviewed** if `label != []` OR `Comments != []`. Unreviewed records (n_ik = 0 for all k) are excluded from the sums and do not contribute to i or c. See §8 for counts.

### 4.2 Data loading and n_ik matrix

Load each annotator's JSONL file; apply `get_labels()` from §3 to build `data`:

```python
from collections import Counter, defaultdict
import json
from pathlib import Path

def load_annotator(path: Path) -> dict[int, list[str]]:
    """Return {doc_id: [label_types]} for reviewed records only."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels = get_labels(rec)
            if labels is not None:
                result[rec["id"]] = labels
    return result

# data[annotator_name] = {doc_id: [label_types]}
data: dict[str, dict[int, list[str]]] = {
    path.stem: load_annotator(path)
    for path in sorted(export_dir.glob("*.jsonl"))
    if path.name not in SKIP_NAMES
    and not any(path.name.endswith(s) for s in SKIP_SUFFIXES)
    and not any(path.name.startswith(p) for p in SKIP_PREFIXES)
}

# Skip annotators with no reviewed records (e.g. herald.jsonl)
data = {a: recs for a, recs in data.items() if recs}
```

Build n_ik from `data`:

```python
n = defaultdict(Counter)   # n[doc_id][label] = number of annotators who assigned label
n_k = Counter()            # pooled label counts across all items and annotators

for annotator, records in data.items():
    for doc_id, labels in records.items():   # labels already deduplicated by type
        for lbl in labels:
            n[doc_id][lbl] += 1
            n_k[lbl] += 1
```

### 4.3 Krippendorff's α (nominal, walkthrough §6)

For the nominal distance function d(a, b) = 0 if a == b, 1 otherwise, the cross-terms simplify:

$$\sum_{j \neq l} n_{ik_j} n_{ik_l} \, d_{k_jk_l} = c_i^2 - \sum_k n_{ik}^2$$

where $c_i = \sum_k n_{ik}$ is the total label assignments for item i.

```python
# Observed disagreement D_o
Do_num = sum(ci**2 - sum(v**2 for v in n[i].values())
             for i in n if (ci := sum(n[i].values())) > 1)
Do_den = sum(ci * (ci - 1)
             for i in n if (ci := sum(n[i].values())) > 1)
if Do_den == 0:
    raise ValueError("Do_den=0: every reviewed document was seen by only one annotator — α undefined")
D_o = Do_num / Do_den

# Expected disagreement D_e
N = sum(n_k.values())
D_e = (N**2 - sum(v**2 for v in n_k.values())) / (N * (N - 1))

if D_e == 0:
    # All annotations are the same label — perfect agreement by convention
    alpha = 1.0
    # emit warning: f"D_e=0 for scope {label}: all annotations identical, α set to 1.0"
else:
    alpha = 1 - D_o / D_e
```

Compute overall α (all labels) and per-label α (restrict items and labels to {label, O} pairs).

### 4.4 Per-label α

For label L (e.g. TITLE):
- **Item set**: all documents reviewed by at least one annotator (same as global — do not restrict by whether L appears).
- **O assignment**: an annotator contributes O to a document's n_ik if they reviewed the document but did not assign label L — regardless of whether they assigned other labels (PERSON, OTHER_TITLE, etc.). This captures genuine disagreement: an annotator who assigned PERSON but missed a TITLE is in disagreement with one who assigned TITLE.
- **n_ik** for scope L: `{L: count_of_L_assignments, O: count_of_reviewers_without_L}` per document.
- Apply the same D_o / D_e formula (including the D_e = 0 convention above).

## 5. CLI

```bash
python scripts/ner/sr08_compute_iaa.py \
    --export-dir data/annotation/doccano/export-20260623-1110 \
    [--output-dir DIR]    # default: data/processed/ner/
```

## 6. Outputs

Two CSVs, written in order (counts first, scores second):

**`sr08_iaa_counts_<export>.csv`** — intermediate raw counts; one row per label assignment:

| column | description |
|---|---|
| `doc_id` | Doccano document id (1–395) |
| `annotator` | annotator filename stem (frankp, gundi, maria) |
| `label` | assigned label: TITLE, OTHER_TITLE, PERSON, … or O (no entity) |

One row per label type per annotator per document (types are deduplicated — two TITLE spans from the same annotator produce one TITLE row, not two). A reviewed document with no entities produces one O row. Unreviewed documents produce no rows.

**`sr08_iaa_<export>.csv`** — final scores; one row per label scope:

| column | description |
|---|---|
| `label` | ALL, then one row per unique label found in n_k (e.g. TITLE, OTHER_TITLE, PERSON) — dynamic, not hardcoded |
| `n_items` | documents included in this scope |
| `N_judgments` | total label assignments across all annotators and items |
| `D_o` | observed disagreement |
| `D_e` | expected disagreement |
| `alpha` | Krippendorff's α |

**Stdout:** reviewed-doc counts per annotator; α table (overall + per label); pass/fail against threshold α ≥ 0.800.

## 7. Code conventions (follow `sr08_count_comments.py`)

- `BASE = Path(__file__).resolve().parents[2]`
- `SKIP_NAMES = {"dummy.doccano.jsonl", "ann.jsonl"}`, `SKIP_SUFFIXES = (".meta.jsonl",)`, `SKIP_PREFIXES = ("sr08_",)`
- stdlib only — no pandas, no seqeval
- `--export-dir` required (no hardcoded default); `--output-dir` defaults to `BASE / "data/processed/ner/"`
- `out_path.parent.mkdir(parents=True, exist_ok=True)` before writing
- All writes go to `--output-dir` only; never write to `--export-dir` or any `.jsonl` file

### Header block

```python
#!/usr/bin/env python3
# Purpose:      Compute Krippendorff's α (nominal) for BibNER-Werk gold annotation
# Usage:        python scripts/ner/sr08_compute_iaa.py \
#                   --export-dir data/annotation/doccano/export-20260623-1110
# Inputs:       *.jsonl files in --export-dir (skips dummy.doccano.jsonl, ann.jsonl, *.meta.jsonl, sr08_*)
# Outputs:      data/processed/ner/sr08_iaa_counts_<export>.csv  (raw label assignments)
#               data/processed/ner/sr08_iaa_<export>.csv          (Krippendorff's α per scope)
#               stdout: annotator coverage, α table (overall + per label), pass/fail
# Dependencies: none (stdlib only)
# Assumptions:  reviewed = label != [] OR Comments != []; empty-label+comment = O assignment
```

## 8. Verification

### 8.1 Run

```bash
python scripts/ner/sr08_compute_iaa.py \
    --export-dir data/annotation/doccano/export-20260623-1110
```

Both output files must exist before proceeding:

```bash
ls data/processed/ner/sr08_iaa_counts_export-20260623-1110.csv
ls data/processed/ner/sr08_iaa_export-20260623-1110.csv
```

### 8.2 Counts CSV — spot-check a known document

Doc id=1 (`"16-30, Deutschlands sittliche Sendung - Bismark"`) has known annotations:
- frankp: `[7,38,"TITLE"]` + `[40,47,"PERSON"]` → two rows: TITLE, PERSON
- gundi annotated it with a comment ("Keine Annotation") → one row: O
- maria: check her export directly

```bash
python3 -c "
import csv
rows = [r for r in csv.DictReader(open('data/processed/ner/sr08_iaa_counts_export-20260623-1110.csv'))
        if r['doc_id'] == '1']
for r in rows: print(r)
"
```

Expected: at minimum two rows for frankp (TITLE, PERSON) and one row for gundi (O).

### 8.3 Formula sanity — perfect agreement

Construct a minimal synthetic case where all annotators agree: D_o must be 0, α must be 1.

```python
# Inline check — paste into a Python REPL
from collections import Counter, defaultdict

n = defaultdict(Counter)
n_k = Counter()

# 2 items, 2 coders, both assign TITLE to both items
for doc_id in [1, 2]:
    for annotator in ["A", "B"]:
        n[doc_id]["TITLE"] += 1
        n_k["TITLE"] += 1

Do_num = sum(ci**2 - sum(v**2 for v in n[i].values())
             for i in n if (ci := sum(n[i].values())) > 1)
N = sum(n_k.values())
D_e = (N**2 - sum(v**2 for v in n_k.values())) / (N * (N - 1))
D_o = Do_num / sum(ci * (ci - 1) for i in n if (ci := sum(n[i].values())) > 1)

assert D_o == 0.0, f"expected D_o=0, got {D_o}"
alpha = 1 - D_o / D_e if D_e > 0 else 1.0
assert alpha == 1.0, f"expected α=1, got {alpha}"
print("perfect agreement: PASS")
```

### 8.4 Scores CSV — range and structure checks

```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('data/processed/ner/sr08_iaa_export-20260623-1110.csv')))
labels = [r['label'] for r in rows]
assert 'ALL' in labels, 'missing ALL row'
for r in rows:
    a = float(r['alpha'])
    assert -1.0 <= a <= 1.0, f\"{r['label']}: α={a} out of range\"
    assert float(r['D_e']) >= 0, f\"{r['label']}: D_e must be non-negative\"
print('Range checks: PASS')
print()
for r in rows:
    print(f\"{r['label']:<15} α={float(r['alpha']):.3f}  n_items={r['n_items']}\")
"
```

Overall α must be a real number; flag if α < 0.800 (below Krippendorff's minimum for reliable data).

### 8.5 Expected annotation counts (from 2026-06-23 export)

| Annotator | Reviewed / Total | Notes |
|-----------|-----------------|-------|
| frankp    | 394 / 395       | 1 unreviewed |
| gundi     | 395 / 395       | 130 empty-label + comment ("Keine Annotation") |
| maria     | 392 / 395       | 98 empty-label + comment; 3 empty+no-comment treated as reviewed |
