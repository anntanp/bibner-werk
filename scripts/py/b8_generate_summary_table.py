#!/usr/bin/env python3
# Purpose:      Generate the compact system-level summary table (label x system,
#               era pooled) for 60-results.tex's "main result table" TODO -- the
#               single at-a-glance table that Tables 7-9 (tab:title-per-era/
#               tab:other-title-per-era/tab:person-per-era) deliberately don't
#               provide, since those are the paper's primary per-era reporting
#               unit (54-evalprotocol.tex). This table reuses the same
#               support-weighted "overall F1" figures already stated in prose
#               throughout 60-63.tex/70-conclusion.tex (e.g. "F1 = 0.087"), now
#               computed once from eval_summary.csv instead of living only as
#               scattered hand-written numbers.
# Usage:        python3 scripts/py/b8_generate_summary_table.py
#                   [--eval-summary PATH] [--output PATH] [--latex-output PATH]
# Inputs:       data/eval/eval_summary.csv (eval_ner_gold.py aggregate output;
#                   must contain rows for gliner_zero, gliner_fewshot_k1,
#                   gliner_fewshot_k3, gliner_fewshot_k5, gliner, xlmr)
# Outputs:      data/processed/b8_summary_table.csv (system, label, weighted_f1,
#                   n_gold_spans -- long format, one row per system x
#                   {TITLE,OTHER_TITLE,PERSON,OVERALL}) and
#                   data/processed/b8_summary_table.tex (one ready-to-paste
#                   \begin{table}...\end{table} block, \textbf{} applied to each
#                   row's best-performing system)
# Dependencies: none (stdlib csv only)
# Assumptions:  The three few-shot k values collapse to one "Few-shot" column,
#                   since 61-fewshot.tex's finding is F1=0.000 uniformly across
#                   every (label, era, k) -- asserted explicitly here, not
#                   assumed, so a future re-run with a nonzero few-shot result
#                   fails loudly instead of silently reporting k=1's number as
#                   if it still represented all three. Reuses
#                   weighted_f1_by_label from b6_summarize_gliner_eval.py rather
#                   than a frozen copy, same cross-script-import convention as
#                   compare_gliner_zero_vs_finetuned.py.
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b6_summarize_gliner_eval import load_rows, weighted_f1_by_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVAL_SUMMARY = ROOT / "data" / "eval" / "eval_summary.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "b8_summary_table.csv"
DEFAULT_LATEX_OUTPUT = ROOT / "data" / "processed" / "b8_summary_table.tex"

ROW_LABELS = ("TITLE", "OTHER_TITLE", "PERSON", "OVERALL")
FEWSHOT_K_SYSTEMS = ("gliner_fewshot_k1", "gliner_fewshot_k3", "gliner_fewshot_k5")

# (internal system key, eval_summary.csv system value(s) to pool, LaTeX column header)
COLUMNS = (
    ("zero_shot", ("gliner_zero",), "Zero-shot"),
    ("few_shot", FEWSHOT_K_SYSTEMS, "Few-shot"),
    ("gliner_ft", ("gliner",), "GLiNER-FT"),
    ("xlmr_ft", ("xlmr",), "XLM-R-FT"),
)


def assert_fewshot_k_uniform(all_rows: list[dict[str, str]]) -> None:
    """Refuse to collapse k=1/3/5 into one column unless every core-label F1 is 0.0 at every k."""
    for k_system in FEWSHOT_K_SYSTEMS:
        rows = [r for r in all_rows if r["system"] == k_system]
        if not rows:
            sys.exit(f"b8_generate_summary_table.py: no eval_summary.csv rows for system={k_system!r}")
        nonzero = [r for r in rows if r["label"] in ("TITLE", "OTHER_TITLE", "PERSON") and float(r["mean_f1"]) != 0.0]
        if nonzero:
            sys.exit(
                f"b8_generate_summary_table.py: {k_system!r} has nonzero F1 cells "
                f"({[(r['label'], r['era'], r['mean_f1']) for r in nonzero]}) -- "
                "few-shot no longer collapses to a single 'Few-shot' column as 61-fewshot.tex "
                "assumes; update this script (and 61-fewshot.tex/62-finetuned.tex) before trusting "
                "a pooled column."
            )


def build_summary(all_rows: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    """weighted_f1_by_label per summary column, pooling the three few-shot k's into one system key."""
    out: dict[str, list[dict[str, object]]] = {}
    for col_key, systems, _ in COLUMNS:
        rows = [r for r in all_rows if r["system"] in systems]
        if not rows:
            sys.exit(f"b8_generate_summary_table.py: no eval_summary.csv rows for system(s) {systems!r}")
        out[col_key] = weighted_f1_by_label(rows)
    return out


def render_latex(summary: dict[str, list[dict[str, object]]]) -> str:
    by_col_label = {
        col_key: {r["label"]: r for r in summary[col_key]} for col_key, _, _ in COLUMNS
    }
    row_display = {
        "TITLE": r"\texttt{TITLE}",
        "OTHER_TITLE": r"\texttt{OTHER\_TITLE}",
        "PERSON": r"\texttt{PERSON}",
        "OVERALL": r"\textbf{Overall}",
    }
    body_lines = []
    for label in ROW_LABELS:
        f1_by_col = {
            col_key: cast(float, by_col_label[col_key][label]["weighted_f1"]) for col_key, _, _ in COLUMNS
        }
        best_col = max(f1_by_col, key=lambda c: f1_by_col[c])
        cells = [row_display[label]]
        for col_key, _, _ in COLUMNS:
            # 3-decimal display to match Tables 7-9's realized precision in 62-finetuned.tex
            # (b7's raw round(...,4) output was hand-trimmed to 3 decimals when pasted there).
            cell = f"{f1_by_col[col_key]:.3f}"
            if col_key == best_col:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        n_gold_spans = by_col_label["zero_shot"][label]["n_gold_spans"]
        cells.append(str(n_gold_spans))
        body_lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(body_lines)
    col_headers = " & ".join(header for _, _, header in COLUMNS)
    return (
        "\\begin{table}\n"
        "  \\caption{Support-weighted F1 by label, pooled across era strata, all four systems "
        "(Section~\\ref{sec:systems}). A summary only: the per-era point estimates in "
        "Tables~\\ref{tab:title-per-era}--\\ref{tab:person-per-era} are the paper's primary "
        "reporting unit (Section~\\ref{sec:evalprotocol}).}\n"
        "  \\label{tab:summary}\n"
        "  \\setlength{\\tabcolsep}{3pt}\n"
        "  \\begin{tabular}{lrrrrr}\n"
        "    \\toprule\n"
        f"    Label & {col_headers} & $n$ \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-summary", type=Path, default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latex-output", type=Path, default=DEFAULT_LATEX_OUTPUT)
    args = parser.parse_args()

    all_rows = load_rows(args.eval_summary)
    assert_fewshot_k_uniform(all_rows)
    summary = build_summary(all_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "label", "weighted_f1", "n_gold_spans"])
        writer.writeheader()
        for col_key, _, header in COLUMNS:
            for row in summary[col_key]:
                writer.writerow({"system": header, **row})
    print(f"Wrote {args.output}")

    latex = render_latex(summary)
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.latex_output, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Wrote {args.latex_output}")
    print()
    print(latex)


if __name__ == "__main__":
    main()
