"""Figures C and D: spline (trend) evidence and risk-component evidence.

Figure C is deliberately a *reduced* figure. ArticleV1's TrendEvidence record is
a scalar-only dataclass (src/trustworthy_agent/evidence/trend.py): it holds
trend_value, first/second derivative, curvature, maximum_curvature and related
scalars, but no ordered feature sequence, no spline evaluation grid, no
derivative array and no uncertainty band. A trajectory figure therefore cannot
be produced without refitting splines and extending the evidence schema, which
would be a new experiment. Figure C shows exactly what was persisted.

Figure D exposes, rather than hides, the saturation of the OOD and
trend-uncertainty components. No weight, scale or threshold was retuned.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _m3_common import (
    BLUE,
    NEUTRAL,
    YELLOW,
    FigureRecord,
    attach_sources,
    export,
    read_csv,
    style_for,
    write_figure_data,
)

TREND_VALUES = "Output/ArticleV1/Figures/v1/Figure_05_v6_temporal_spline_comparison_data.csv"
TREND_DESCRIPTORS = "Output/ArticleV1/Figures/v1/Figure_06_v6_derivatives_curvature_data.csv"
RISK = "Output/ArticleV1/Figures/v1/Figure_10_combined_risk_acquisitions_data.csv"

TREND_LABEL = {
    "RollingSmoothingSpline": "Rolling smoothing spline",
    "RollingBSpline": "Rolling B-spline",
    "RollingPSpline": "Rolling P-spline",
    "RollingHealthyRelativeSpline": "Rolling healthy-relative spline",
}
DESCRIPTOR_LABEL = {
    "trend_value": "Trend value",
    "first_derivative": "First derivative",
    "second_derivative": "Second derivative",
    "curvature": "Curvature",
    "maximum_curvature": "Maximum curvature",
}
RISK_LABEL = {
    "classifier_uncertainty": "Classifier uncertainty",
    "healthy_deviation": "Healthy deviation",
    "trend_uncertainty": "Trend uncertainty",
    "ood_contribution": "OOD contribution",
    "combined_risk": "Combined risk",
}


def build_fig_c(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_C"]
    order = cfg["orderings"]["trend_model"]
    vals = read_csv(TREND_VALUES)
    desc = read_csv(TREND_DESCRIPTORS)
    assert len(vals) == 256, f"expected 256 persisted trend records, got {len(vals)}"
    assert {d["component"] for d in desc} == set(DESCRIPTOR_LABEL)

    record = FigureRecord("FIG_C", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [TREND_VALUES, TREND_DESCRIPTORS, "src/trustworthy_agent/evidence/trend.py"])
    record.display_transforms = [
        "symmetric-log x axis (descriptors span ~1e-9 to ~1e6); axis transform only",
        "vertical jitter of tied trend values for visibility (display only)",
    ]

    plotted: list[dict] = []
    with style_for(spec["export_profile"]):
        fig, (axl, axr) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1.0]})

        # Panel (a): every persisted scalar trend value, by spline representation.
        rng = np.random.default_rng(0)  # display jitter only; never touches values
        for i, model in enumerate(order):
            v = np.array([float(r["trend_value"]) for r in vals if r["trend_model_id"] == model])
            assert len(v) == 64, f"{model}: expected 64 records, got {len(v)}"
            y = i + rng.uniform(-0.16, 0.16, len(v))
            axl.scatter(v, y, s=9, facecolor=BLUE["mid"], edgecolor=BLUE["dark"],
                        linewidth=0.3, alpha=0.85, zorder=3)
            for x in v:
                plotted.append({"figure": "FIG_C", "panel": "a", "trend_model_id": model,
                                "quantity": "trend_value", "value": x})
        axl.set_xscale("symlog", linthresh=1.0)
        axl.set_xticks([-1e7, -1e4, -1e1, 0, 1e1, 1e4, 1e7])
        axl.xaxis.set_minor_locator(plt.NullLocator())
        axl.axvline(0, color=NEUTRAL["light"], lw=0.7, zorder=1)
        axl.set_yticks(range(len(order)))
        axl.set_yticklabels([TREND_LABEL[m] for m in order], fontsize=7)
        axl.set_ylim(len(order) - 0.5, -0.5)
        axl.set_xlabel("Persisted scalar trend value (symlog)")
        axl.spines["left"].set_visible(False)
        axl.tick_params(axis="y", length=0)
        axl.tick_params(axis="x", labelsize=6.5)
        axl.set_title("All 256 persisted trend records\n(64 per representation)", fontsize=7.2, pad=5)

        # Panel (b): the five scalar descriptors persisted for one record. This is
        # the complete spline evidence for that record -- there is nothing else.
        dvals = {d["component"]: float(d["value"]) for d in desc}
        keys = list(DESCRIPTOR_LABEL)
        for i, k in enumerate(keys):
            x = dvals[k]
            axr.plot([0, x], [i, i], color=NEUTRAL["light"], lw=0.8, zorder=2)
            axr.scatter([x], [i], s=26, marker="D",
                        facecolor=YELLOW["mid"] if x < 0 else BLUE["dark"],
                        edgecolor=NEUTRAL["near_black"], linewidth=0.5, zorder=4)
            axr.annotate(f"{x:.3g}", xy=(x, i - 0.28), ha="center", va="bottom",
                         fontsize=6.3, color=NEUTRAL["near_black"])
            plotted.append({"figure": "FIG_C", "panel": "b", "trend_model_id": "RollingSmoothingSpline",
                            "quantity": k, "value": x})
        axr.set_xscale("symlog", linthresh=1e-9)
        axr.set_xticks([-1e6, 0, 1e-6, 1e6])
        axr.xaxis.set_minor_locator(plt.NullLocator())
        axr.set_xlim(-3e7, 3e7)
        axr.axvline(0, color=NEUTRAL["mid"], lw=0.7, zorder=1)
        axr.set_yticks(range(len(keys)))
        axr.set_yticklabels([DESCRIPTOR_LABEL[k] for k in keys], fontsize=7)
        axr.set_ylim(len(keys) - 0.5, -0.8)
        axr.set_xlabel("Value (symlog)")
        axr.spines["left"].set_visible(False)
        axr.tick_params(axis="y", length=0)
        axr.tick_params(axis="x", labelsize=6.5)
        axr.set_title("One persisted record\n(A05, healthy_1, smoothing spline)",
                      fontsize=7.2, pad=5)

        # The scalar-only nature of the evidence and the meaning of the
        # derivatives/curvature are stated in the caption, not the plot area.
        fig.subplots_adjust(left=0.225, right=0.985, top=0.86, bottom=0.16, wspace=0.62)

    write_figure_data("FIG_C", plotted, record)
    record.checks = {
        "trend_records": len(vals),
        "records_per_model": 64,
        "descriptors_plotted": len(keys),
        "trajectory_available": False,
        "reduced_figure": True,
    }
    return fig, record


def build_fig_d(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_D"]
    rows = read_csv(RISK)
    assert len(rows) == 768, f"expected 768 risk rows, got {len(rows)}"
    comps = list(RISK_LABEL)

    record = FigureRecord("FIG_D", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [RISK])
    record.display_transforms = [
        "empirical CDF computed for display (order statistics of the plotted column; no smoothing)",
        "count of distinct persisted values per component",
    ]

    data = {c: np.sort(np.array([float(r[c]) for r in rows])) for c in comps}
    plotted: list[dict] = []

    with style_for(spec["export_profile"]):
        fig, (axl, axr) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1.0]})

        # Panel (a): ECDFs. A vertical rise at 1.0 is exactly the saturation we
        # must not conceal, and an ECDF makes it unmissable.
        styles = {
            "classifier_uncertainty": (BLUE["dark_highlight"], "-", 1.5),
            "healthy_deviation": (BLUE["mid_highlight"], "--", 1.2),
            "combined_risk": (BLUE["dark"], "-", 1.2),
            "trend_uncertainty": (NEUTRAL["mid"], "-.", 1.2),
            "ood_contribution": (NEUTRAL["dark"], ":", 1.6),
        }
        # Five overlapping step curves cannot carry stable direct labels, so a
        # legend is used here (a documented exception to the direct-label
        # preference). The saturated fraction is folded into each legend entry,
        # which states the key result without an extra annotation.
        from matplotlib.lines import Line2D

        handles = []
        for c in comps:
            v = data[c]
            y = np.arange(1, len(v) + 1) / len(v)
            color, ls, lw = styles[c]
            axl.step(v, y, where="post", color=color, ls=ls, lw=lw, zorder=3)
            sat = float(np.mean(v == 1.0))
            handles.append(
                Line2D([0], [0], color=color, ls=ls, lw=lw, label=RISK_LABEL[c])
            )
            for x in v:
                plotted.append({"figure": "FIG_D", "panel": "a", "component": c,
                                "quantity": "component_value", "value": float(x)})
            record.checks.setdefault("saturated_fraction_at_1", {})[c] = round(sat, 4)

        axl.axvline(1.0, color=YELLOW["dark"], lw=1.0, ls="-", zorder=2, alpha=0.9)
        axl.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=False,
                   handlelength=2.4, labelspacing=0.35, borderpad=0.1,
                   bbox_to_anchor=(-0.01, 1.03))
        axl.set_xlim(0, 1.06)
        axl.set_ylim(0, 1.02)
        axl.set_xlabel("Component value (dimensionless, bounded [0, 1])")
        axl.set_ylabel("Empirical CDF over 768 records")

        # Panel (b): resolution. Distinct persisted values per component quantifies
        # how little information a saturated component can carry.
        nuniq = {c: len({r[c] for r in rows}) for c in comps}
        keys = sorted(comps, key=lambda c: nuniq[c])
        for i, c in enumerate(keys):
            n = nuniq[c]
            axr.plot([0, n], [i, i], color=NEUTRAL["light"], lw=0.9, zorder=2)
            axr.scatter([n], [i], s=30,
                        facecolor=YELLOW["mid"] if n <= 20 else BLUE["dark"],
                        edgecolor=NEUTRAL["near_black"], linewidth=0.5, zorder=4)
            axr.annotate(f"{n}", xy=(n, i), xytext=(5, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=6.6, color=NEUTRAL["near_black"])
            plotted.append({"figure": "FIG_D", "panel": "b", "component": c,
                            "quantity": "distinct_values", "value": n})
        axr.set_xscale("log")
        axr.set_xlim(3, 1500)
        axr.set_yticks(range(len(keys)))
        axr.set_yticklabels([RISK_LABEL[c] for c in keys])
        axr.set_ylim(len(keys) - 0.5, -0.5)
        axr.set_xlabel("Distinct values (log scale)")
        axr.spines["left"].set_visible(False)
        axr.tick_params(axis="y", length=0)
        axr.set_title("Resolution of each component", fontsize=7.2, pad=5)

        # Record composition, the no-retuning statement, and the meaning of the
        # combined risk are in the caption, not the plot area.
        fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.16, wspace=0.85)

    write_figure_data("FIG_D", plotted, record)
    record.checks["rows"] = len(rows)
    record.checks["distinct_values"] = nuniq
    return fig, record
