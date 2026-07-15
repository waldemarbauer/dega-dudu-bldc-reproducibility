"""Figures A and B: acquisition-level performance and evaluation-level contrast.

Both figures consume the frozen per-assignment metric tables. The central
constraint they encode visually is that the 16 assignment views are dependent
exhaustive views of the same 8 acquisitions, so no aggregate error bar implying
independent replication is drawn anywhere.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _m3_common import (
    BLUE,
    CLASSIFIER_COLOR,
    CLASSIFIER_MARKER,
    NEUTRAL,
    FigureRecord,
    attach_sources,
    export,
    read_csv,
    style_for,
    write_figure_data,
)

ACQ_METRICS = "Output/ArticleV1/Tables/article_acquisition_metrics_by_assignment.csv"
WIN_METRICS = "Output/ArticleV1/Tables/article_window_metrics_by_assignment.csv"
ASSIGN_SUMMARY = "Output/ArticleV1/Tables/acquisition_assignment_summary.csv"


def _stack_offsets(values: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Return a vertical offset index for each value, stacking exact ties.

    Acquisition-level scores over 4 held-out acquisitions are coarsely quantised
    and tie heavily. Stacking ties keeps every one of the 16 views visible
    instead of hiding them under one another.
    """
    offsets = np.zeros(len(values))
    seen: dict[float, int] = {}
    for i, v in enumerate(values):
        key = next((k for k in seen if abs(k - v) < tol), v)
        offsets[i] = seen.get(key, 0)
        seen[key] = seen.get(key, 0) + 1
    return offsets


def _verify(rows: list[dict], cfg: dict, level: str) -> None:
    ev = cfg["evaluation_units"]
    assert len(rows) == ev["assignment_views"] * ev["classifier_count"], f"{level}: row count"
    assert {r["assignment_id"] for r in rows} == {f"A{i:02d}" for i in range(16)}
    assert {r["interpretation"] for r in rows} == {ev["interpretation_flag"]}
    expected = ev["held_out_acquisitions_per_view"] if level == "acquisition" else ev["held_out_windows_per_view"]
    assert {int(r["row_count"]) for r in rows} == {expected}, f"{level}: unit count"


def build_fig_a(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_A"]
    order = cfg["orderings"]["classifier"]
    labels = cfg["orderings"]["classifier_label"]
    rows = read_csv(ACQ_METRICS)
    _verify(rows, cfg, "acquisition")

    record = FigureRecord("FIG_A", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [ACQ_METRICS, ASSIGN_SUMMARY])
    record.display_transforms = [
        "selection of macro_f1 and balanced_accuracy columns",
        "vertical stacking of exactly tied values for visibility (display only)",
    ]

    metrics = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy")]
    plotted: list[dict] = []

    with style_for(spec["export_profile"]):
        fig, axes = plt.subplots(1, 2, sharey=True)
        for ax, (col, xlabel) in zip(axes, metrics):
            for ci, clf in enumerate(order):
                vals = np.array([float(r[col]) for r in rows if r["classifier_id"] == clf])
                asg = [r["assignment_id"] for r in rows if r["classifier_id"] == clf]
                med = float(np.median(vals))
                # Median rule spans the row, drawn behind the dots.
                ax.plot([med, med], [ci - 0.34, ci + 0.46], color=NEUTRAL["near_black"],
                        lw=1.1, zorder=2, solid_capstyle="butt")
                off = _stack_offsets(vals)
                ax.scatter(
                    vals, ci + off * 0.055, s=11,
                    marker=CLASSIFIER_MARKER[clf],
                    facecolor=CLASSIFIER_COLOR[clf],
                    edgecolor=NEUTRAL["near_black"], linewidth=0.35, zorder=3,
                )
                # Median value parked in a clear column to the right of the data.
                ax.annotate(f"med {med:.2f}", xy=(1.09, ci), ha="left", va="center",
                            fontsize=6.4, color=NEUTRAL["near_black"])
                for a, v in zip(asg, vals):
                    plotted.append(
                        {"figure": "FIG_A", "metric": col, "classifier_id": clf,
                         "assignment_id": a, "value": v}
                    )
            ax.set_xlim(-0.02, 1.34)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlabel(xlabel)
            ax.set_ylim(2.85, -0.55)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_bounds(0.0, 1.0)
            ax.tick_params(axis="y", length=0)

        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels([labels[c] for c in order])
        axes[1].tick_params(axis="y", labelleft=False)
        # The evaluation unit and the dependence of the 16 views are stated in the
        # caption, not in the plot area.
        fig.subplots_adjust(left=0.20, right=0.985, top=0.94, bottom=0.17, wspace=0.12)

    write_figure_data("FIG_A", plotted, record)
    record.checks = {
        "rows_read": len(rows),
        "assignment_views": len({r["assignment_id"] for r in rows}),
        "marks_plotted": len(plotted),
        "held_out_acquisitions_per_view": sorted({int(r["row_count"]) for r in rows}),
    }
    return fig, record


def build_fig_b(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_B"]
    order = cfg["orderings"]["classifier"]
    labels = cfg["orderings"]["classifier_label"]
    win, acq = read_csv(WIN_METRICS), read_csv(ACQ_METRICS)
    _verify(win, cfg, "window")
    _verify(acq, cfg, "acquisition")

    # Exact pairing on (assignment_id, classifier_id): the same fitted model in
    # the same view, scored at two evaluation levels. No interpolation.
    wkey = {(r["assignment_id"], r["classifier_id"]): float(r["macro_f1"]) for r in win}
    akey = {(r["assignment_id"], r["classifier_id"]): float(r["macro_f1"]) for r in acq}
    assert set(wkey) == set(akey), "window/acquisition pairing keys differ"

    record = FigureRecord("FIG_B", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [WIN_METRICS, ACQ_METRICS])
    record.display_transforms = [
        "selection of macro_f1 column at both evaluation levels",
        "exact join on (assignment_id, classifier_id); no aggregation",
    ]

    plotted: list[dict] = []
    with style_for(spec["export_profile"]):
        fig, axes = plt.subplots(1, 3, sharey=True)
        for ax, clf in zip(axes, order):
            keys = sorted(k for k in wkey if k[1] == clf)
            for k in keys:
                w, a = wkey[k], akey[k]
                ax.plot(
                    [0, 1],
                    [w, a],
                    color=CLASSIFIER_COLOR[clf],
                    lw=0.8,
                    alpha=0.75,
                    marker=CLASSIFIER_MARKER[clf],
                    markersize=3.2,
                    markerfacecolor=CLASSIFIER_COLOR[clf],
                    markeredgecolor=NEUTRAL["near_black"],
                    markeredgewidth=0.3,
                    zorder=3,
                )
                plotted += [
                    {"figure": "FIG_B", "classifier_id": clf, "assignment_id": k[0],
                     "level": "window", "macro_f1": w},
                    {"figure": "FIG_B", "classifier_id": clf, "assignment_id": k[0],
                     "level": "acquisition", "macro_f1": a},
                ]
            mw = float(np.median([wkey[k] for k in keys]))
            ma = float(np.median([akey[k] for k in keys]))
            ax.plot([0, 1], [mw, ma], color=NEUTRAL["near_black"], lw=1.8, zorder=5)
            ax.annotate(f"{mw:.2f}", xy=(-0.06, mw), ha="right", va="center", fontsize=6.5,
                        color=NEUTRAL["near_black"])
            ax.annotate(f"{ma:.2f}", xy=(1.06, ma), ha="left", va="center", fontsize=6.5,
                        color=NEUTRAL["near_black"])
            ax.set_xlim(-0.42, 1.42)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Window\n(100 corr.)", "Acquisition\n(4 held-out)"], fontsize=7)
            ax.set_title(labels[clf], fontsize=8, pad=4)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0)

        axes[0].set_ylabel("Macro-F1")
        axes[0].set_ylim(0.0, 1.05)
        # Line meanings (per-view pairing, median) and window correlation are in
        # the caption, not the plot area.
        fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.19, wspace=0.30)

    write_figure_data("FIG_B", plotted, record)
    record.checks = {
        "paired_keys": len(wkey),
        "marks_plotted": len(plotted),
        "window_units_per_view": sorted({int(r["row_count"]) for r in win}),
        "acquisition_units_per_view": sorted({int(r["row_count"]) for r in acq}),
    }
    return fig, record
