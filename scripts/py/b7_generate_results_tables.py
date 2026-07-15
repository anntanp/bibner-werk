#!/usr/bin/env python3
# Purpose:      Generate the per-era, three-system (zero-shot / GLiNER fine-tuned /
#               XLM-R fine-tuned) F1 comparison used in 60-results.tex's Tables 7-9
#               (tab:title-per-era, tab:other-title-per-era, tab:person-per-era) --
#               written to replace a hand-transcribed version of these tables (copied
#               from B3-gliner-zeroshot-eval.md/B6-gliner-eval.md/B6-xlmr-eval.md by
#               reading, not from eval_summary.csv directly) with a real, rerunnable
#               artifact, per this project's data-derived-tables convention (never
#               hand-write a number that a script could produce).
# Usage:        python3 scripts/py/b7_generate_results_tables.py
#                   [--eval-summary PATH] [--output PATH] [--latex-output PATH]
# Inputs:       data/eval/eval_summary.csv (eval_ner_gold.py aggregate output; must
#                   contain rows for systems gliner_zero, gliner, xlmr)
# Outputs:      data/processed/b7_results_per_era_by_system.csv (label, era,
#                   gliner_zero_f1, gliner_zero_std, gliner_f1, gliner_std, xlmr_f1,
#                   xlmr_std, n_gold_spans, best_system) and, by default, a .tex
#                   snippet (data/processed/b7_results_tables.tex) with the three
#                   ready-to-paste \begin{table}...\end{table} blocks, \textbf{}
#                   already applied to each row's best_system column.
# Dependencies: none (stdlib csv only)
# Assumptions:  n_gold_spans is a property of the gold set, not the system under
#                   evaluation, so it must match across all three systems at every
#                   (label, era) -- asserted explicitly, same posture as
#                   compare_gliner_zero_vs_finetuned.py, since a mismatch would mean
#                   the systems were scored against different fold/gold slices.
#                   Era display order/labels are this project's fixed five strata
#                   (30-corpus.tex Table tab:corpus-era); a era value in
#                   eval_summary.csv outside this set is treated as a hard error,
#                   not silently dropped, since it would mean the gold stratification
#                   changed without this script being updated.
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVAL_SUMMARY = ROOT / "data" / "eval" / "eval_summary.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "b7_results_per_era_by_system.csv"
DEFAULT_LATEX_OUTPUT = ROOT / "data" / "processed" / "b7_results_tables.tex"

CORE_LABELS = ("TITLE", "OTHER_TITLE", "PERSON")
SYSTEMS = ("gliner_zero", "gliner", "xlmr")
SYSTEM_COLUMN_HEADER = {"gliner_zero": "Zero-shot", "gliner": "GLiNER-FT", "xlmr": "XLM-R-FT"}

# (eval_summary.csv era value, LaTeX display string) -- fixed display order per
# 30-corpus.tex's Table tab:corpus-era, not alphabetical or CSV row order.
ERA_ORDER = (
    ("pre-1700", "Pre-1700"),
    ("1700-1800", "1700--1800"),
    ("19th-c", "19th-c"),
    ("modern", "Modern"),
    ("unknown", "Unknown"),
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_comparison(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """One row per (label, era): all three systems' F1 side by side, plus the winner."""
    by_key: dict[tuple[str, str, str], dict[str, str]] = {
        (r["system"], r["label"], r["era"]): r for r in rows
    }

    known_eras = {era for era, _ in ERA_ORDER}
    seen_eras = {r["era"] for r in rows if r["label"] in CORE_LABELS}
    unexpected = seen_eras - known_eras
    if unexpected:
        sys.exit(
            f"b7_generate_results_tables.py: eval_summary.csv has era value(s) "
            f"{sorted(unexpected)} not in this script's fixed ERA_ORDER {sorted(known_eras)} "
            "-- update ERA_ORDER before trusting this output."
        )

    out: list[dict[str, object]] = []
    for label in CORE_LABELS:
        for era, era_display in ERA_ORDER:
            per_system: dict[str, dict[str, str]] = {}
            for system in SYSTEMS:
                key = (system, label, era)
                if key not in by_key:
                    sys.exit(
                        f"b7_generate_results_tables.py: no eval_summary.csv row for "
                        f"system={system!r} label={label!r} era={era!r}"
                    )
                per_system[system] = by_key[key]

            n_gold_spans = {int(per_system[s]["n_gold_spans"]) for s in SYSTEMS}
            if len(n_gold_spans) != 1:
                sys.exit(
                    f"b7_generate_results_tables.py: n_gold_spans mismatch for "
                    f"label={label!r} era={era!r} across systems ("
                    + ", ".join(f"{s}={per_system[s]['n_gold_spans']}" for s in SYSTEMS)
                    + ") -- systems scored against different gold/fold slices, refusing to compare."
                )

            f1_by_system = {s: float(per_system[s]["mean_f1"]) for s in SYSTEMS}
            best_system = max(f1_by_system, key=lambda s: f1_by_system[s])

            row: dict[str, object] = {"label": label, "era": era, "era_display": era_display}
            for s in SYSTEMS:
                row[f"{s}_f1"] = per_system[s]["mean_f1"]
                row[f"{s}_std"] = per_system[s]["std_f1"]
            row["n_gold_spans"] = n_gold_spans.pop()
            row["best_system"] = best_system
            out.append(row)
    return out


def render_latex(comparison: list[dict[str, object]]) -> str:
    """Emit the three \\begin{table}...\\end{table} blocks, bolding each row's best system."""
    label_meta = {
        "TITLE": ("tab:title-per-era", r"\texttt{TITLE}"),
        "OTHER_TITLE": ("tab:other-title-per-era", r"\texttt{OTHER\_TITLE}"),
        "PERSON": ("tab:person-per-era", r"\texttt{PERSON}"),
    }
    blocks: list[str] = []
    for label in CORE_LABELS:
        table_label, tex_name = label_meta[label]
        label_rows = [r for r in comparison if r["label"] == label]
        body_lines = []
        for r in label_rows:
            cells = [str(r["era_display"])]
            for s in SYSTEMS:
                cell = f"{r[f'{s}_f1']} $\\pm$ {r[f'{s}_std']}"
                if r["best_system"] == s:
                    cell = f"\\textbf{{{r[f'{s}_f1']}}} $\\pm$ {r[f'{s}_std']}"
                cells.append(cell)
            cells.append(str(r["n_gold_spans"]))
            body_lines.append("    " + " & ".join(cells) + r" \\")
        body = "\n".join(body_lines)
        blocks.append(
            "\\begin{table}\n"
            f"  \\caption{{{tex_name} F1 by era: GLiNER-multi zero-shot, GLiNER-multi "
            "fine-tuned, and XLM-R-large fine-tuned (mean $\\pm$ std across ten folds).}\n"
            f"  \\label{{{table_label}}}\n"
            "  \\begin{tabular}{lrrrr}\n"
            "    \\toprule\n"
            "    Era & Zero-shot & GLiNER-FT & XLM-R-FT & Gold spans \\\\\n"
            "    \\midrule\n"
            f"{body}\n"
            "    \\bottomrule\n"
            "  \\end{tabular}\n"
            "\\end{table}"
        )
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-summary", type=Path, default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latex-output", type=Path, default=DEFAULT_LATEX_OUTPUT)
    args = parser.parse_args()

    all_rows = load_rows(args.eval_summary)
    for system in SYSTEMS:
        if not any(r["system"] == system for r in all_rows):
            known = sorted({r["system"] for r in all_rows})
            sys.exit(
                f"b7_generate_results_tables.py: no rows for system {system!r} in "
                f"{args.eval_summary} (systems present: {known})"
            )

    comparison = build_comparison(all_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label", "era", "era_display",
        "gliner_zero_f1", "gliner_zero_std",
        "gliner_f1", "gliner_std",
        "xlmr_f1", "xlmr_std",
        "n_gold_spans", "best_system",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison)
    print(f"Wrote {args.output}")

    latex = render_latex(comparison)
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.latex_output, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Wrote {args.latex_output}")
    print()
    print(latex)


if __name__ == "__main__":
    main()
