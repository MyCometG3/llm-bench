"""Generate PNG charts from an LLM bench summary markdown file.

Modes:
  default        one line chart per metric (x=condition, series=model)
  --per-model    one chart per model with all metrics overlaid
                 (left axis = characters as bars, right axis = time as line)

Usage:
  uv run --with matplotlib python3 plot_summary_charts.py <summary.md> <outdir> [--per-model]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHAR_METRICS = [
    ("thinking_chars", "Thinking chars", "tab:blue"),
    ("answer_chars", "Answer chars", "tab:orange"),
]
TIME_METRICS = [
    ("total_s", "total_s", "tab:red"),
]

EFFORT_ORDER = ["off", "low", "medium", "high", "xhigh"]


def _budget_sort(budget: int | None) -> tuple[bool, int]:
    """Sort key placing the budget-free slot (None) before integer budgets."""
    return (budget is not None, budget or 0)


def build_conditions(
    rows: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, int | None]], list[str]]:
    """Derive (thinking, effort, budget) conditions present in the data, in
    canonical order: off/off first, then on rows by effort (off, low, medium,
    high, xhigh; unknown efforts alphabetically last) and, within an effort,
    by budget (no budget first, then ascending). Labels append the budget
    segment only when one is present; segments are stacked on separate lines
    ("on\\nmedium\\n2048") so many conditions stay readable on one axis."""
    present = {r["cond"] for r in rows}
    ordered = sorted(
        present,
        key=lambda c: (
            0 if c[0] == "off" else 1,
            EFFORT_ORDER.index(c[1]) if c[1] in EFFORT_ORDER else len(EFFORT_ORDER),
            c[1],
            _budget_sort(c[2]),
        ),
    )
    labels = [
        "\n".join(
            segment
            for segment in (thinking, effort, None if b is None else str(b))
            if segment is not None
        )
        for thinking, effort, b in ordered
    ]
    return ordered, labels


def parse(path: str) -> list[dict[str, Any]]:
    """Parse Table 1 rows, preserving conditions with unavailable means.

    A dash in a metric cell becomes ``math.nan`` so failed or incomplete
    conditions remain on the common condition axis without being plotted as
    zero.

    Columns are resolved by NAME from the Table 1 header row instead of by
    fixed index: run_bench has changed this table's layout twice (budget
    column, then the answer_chars / no_ans columns) and an index-based reader
    silently plots the wrong neighbour when it does."""
    # header label -> row key. Summaries written before Table 1 split the
    # echoed thinking block out of content_chars report that position as
    # content_chars; those runs had no echoed rows, so it is the answer length.
    key_of_label = {
        "total_s": "total_s",
        "thinking_chars": "thinking_chars",
        "answer_chars": "answer_chars",
        "content_chars": "answer_chars",
    }
    rows: list[dict[str, Any]] = []
    col: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            labels = [c.removesuffix(" mean/med").strip() for c in cells]
            if "model" in labels and "thinking_chars" in labels:
                col = {name: i for i, name in enumerate(labels)}
                continue
            # data row of the last seen header only when the width matches, so
            # tables 2-5 are not read as Table 1 rows
            if not col or len(cells) != len(col) or set(cells[0]) <= {"-", ":"}:
                continue

            def cell(label: str) -> str:
                """The data cell under one Table 1 header label ('-' if absent)."""
                idx = col.get(label)
                return cells[idx] if idx is not None else "-"

            def mean_value(text: str) -> float:
                """Parse a mean cell, retaining a dash as NaN."""
                value = text.split("/", 1)[0]
                return math.nan if value in ("-", "") else float(value)

            try:
                budget: int | None = None
                if cell("budget") != "-":
                    budget = int(float(cell("budget")))
                row: dict[str, Any] = {
                    "model": cell("model"),
                    "cond": (cell("thinking"), cell("effort"), budget),
                    "total_s": math.nan,
                    "thinking_chars": math.nan,
                    "answer_chars": math.nan,
                }
                for label, key in key_of_label.items():
                    if label in col:
                        row[key] = mean_value(cell(label))
                if not row["model"]:
                    continue
                rows.append(row)
            except ValueError:
                continue
    return rows


def grouped_charts(
    rows: list[dict[str, Any]],
    models: list[str],
    conditions: list[tuple[str, str, int | None]],
    cond_labels: list[str],
    outdir: str,
) -> None:
    """One line chart per metric, models as series."""
    metrics = [
        ("total_s", "total_s", "Total response time (s)"),
        ("thinking_chars", "thinking_chars", "Thinking characters"),
        ("answer_chars", "answer_chars", "Answer characters"),
    ]
    axis_label = (
        "thinking / effort / budget"
        if any(c[2] is not None for c in conditions)
        else "thinking / effort"
    )
    lookup = {(r["model"], r["cond"]): r for r in rows}
    n_conds = len(conditions)
    x = list(range(n_conds))

    for key, fname, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m in models:
            vals = [lookup.get((m, c), {}).get(key, math.nan) for c in conditions]
            ax.plot(x, vals, marker="o", label=m, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_xlabel(axis_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by model (mean over prompts x runs)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = f"{outdir}/{fname}_by_model.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")


def per_model_charts(
    rows: list[dict[str, Any]],
    models: list[str],
    conditions: list[tuple[str, str, int | None]],
    cond_labels: list[str],
    outdir: str,
) -> None:
    """One chart per model: char metrics as bars (left axis), time as line (right axis)."""
    axis_label = (
        "thinking / effort / budget"
        if any(c[2] is not None for c in conditions)
        else "thinking / effort"
    )
    lookup = {(r["model"], r["cond"]): r for r in rows}
    n_conds = len(conditions)
    bar_w = 0.8 / (1 + len(CHAR_METRICS))
    x = list(range(n_conds))
    group_centers = [xi + bar_w * (len(CHAR_METRICS) - 1) / 2 for xi in x]

    # NOTE [refactor]: model names are sanitized into filenames here; distinct
    # raw names can collide after sanitizing (e.g. "foo/bar" and "foo?bar" both
    # become "foo_bar"). Colliding models are skipped with a stderr warning
    # rather than silently overwriting. If a unique name is required, derive it
    # from a content hash of the raw model identifier instead.
    seen = set()
    for m in models:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", m)
        if safe in seen:
            print(f"warning: model '{m}' maps to an already-used filename "
                  f"'{safe}'; skipping its per-model chart", file=sys.stderr)
            continue
        seen.add(safe)

        fig, ax1 = plt.subplots(figsize=(10, 5.5))
        ax2 = ax1.twinx()

        for i, (key, label, color) in enumerate(CHAR_METRICS):
            vals = [lookup.get((m, c), {}).get(key, math.nan) for c in conditions]
            ax1.bar(
                [xi + i * bar_w for xi in x],
                vals,
                width=bar_w,
                label=label,
                color=color,
                alpha=0.85,
            )

        for key, label, color in TIME_METRICS:
            vals = [lookup.get((m, c), {}).get(key, math.nan) for c in conditions]
            ax2.plot(
                group_centers,
                vals,
                marker="o",
                label=label,
                color=color,
                linewidth=2,
                markersize=6,
            )

        ax1.set_xticks(group_centers)
        ax1.set_xticklabels(cond_labels)
        ax1.set_xlabel(axis_label)
        ax1.set_ylabel("characters")
        ax2.set_ylabel("time (s)")
        ax1.set_title(f"{m} (mean over prompts x runs)")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        ax1.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = f"{outdir}/per_model_{safe}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")


def main() -> None:
    """Parse CLI arguments and generate the requested charts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary")
    ap.add_argument("outdir", help="directory to write PNGs to (created if missing)")
    ap.add_argument(
        "--per-model",
        action="store_true",
        help="one chart per model with all metrics (chars left / time right)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = parse(args.summary)
    if not rows:
        ap.error("no data rows parsed from summary")
    models = sorted({r["model"] for r in rows})
    conditions, cond_labels = build_conditions(rows)

    if args.per_model:
        per_model_charts(rows, models, conditions, cond_labels, args.outdir)
    else:
        grouped_charts(rows, models, conditions, cond_labels, args.outdir)


if __name__ == "__main__":
    main()
