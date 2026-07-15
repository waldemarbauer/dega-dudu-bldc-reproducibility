"""Figures E and F: global permutation importance and explanation stability.

Only explanation analyses that ArticleV1 actually executed are shown: global
permutation importance, and top-k stability across assignment views. No SHAP,
PDP, ALE, ICE or surrogate method is introduced.
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

GLOBAL_MEANS = "Output/ArticleV1/XAI/Minimal/GlobalPermutationImportance/distributions.csv"
GLOBAL_REPEATS = "Output/ArticleV1/XAI/Minimal/GlobalPermutationImportance/repeat_distributions.csv"
JACCARD = "Output/ArticleV1/XAI/Minimal/ExplanationStability/jaccard_matrix.csv"
TOPK_FREQ = "Output/ArticleV1/XAI/Minimal/ExplanationStability/top10_frequency.csv"
FEATURE_DICT = "Output/ArticleV1/Tables/article_window_feature_dictionary.csv"


def _pretty(feature: str) -> str:
    return feature.replace("_", " ").replace(" hz", " (Hz)")


def build_fig_e(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_E"]
    means = read_csv(GLOBAL_MEANS)
    repeats = read_csv(GLOBAL_REPEATS)

    # Guard the §6.2 constraint: only the 28 executed features may appear, and no
    # harmonic amplitude feature (NI001-NI006) was implemented in ArticleV1.
    fdict = read_csv(FEATURE_DICT)
    included = {r["feature_name"] for r in fdict if r["included_in_feature_vector"] == "True"}
    assert len(included) == 28, f"expected 28 included features, got {len(included)}"
    assert len(means) == 28, f"expected 28 importance rows, got {len(means)}"
    assert {m["feature"] for m in means} <= included, "importance references a non-included feature"
    assert not any("harmonic" in f for f in {m["feature"] for m in means}), "harmonic feature leaked in"
    assert {m["classifier_id"] for m in means} == {"random_forest"}
    assert {m["assignment_id"] for m in means} == {"A00"}
    assert {m["scoring_metric"] for m in means} == {"balanced_accuracy"}
    assert len(repeats) == 28 * 30, f"expected 840 repeat rows, got {len(repeats)}"

    record = FigureRecord("FIG_E", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [GLOBAL_MEANS, GLOBAL_REPEATS, FEATURE_DICT])
    record.display_transforms = ["ordering of features by persisted mean_importance (display only)"]

    by_feature: dict[str, list[float]] = {}
    for r in repeats:
        by_feature.setdefault(r["feature"], []).append(float(r["importance_drop"]))

    ordered = sorted(means, key=lambda m: float(m["mean_importance"]))
    plotted: list[dict] = []

    with style_for(spec["export_profile"]):
        fig, ax = plt.subplots()
        for i, m in enumerate(ordered):
            f = m["feature"]
            mu = float(m["mean_importance"])
            reps = np.array(by_feature[f])
            # Individual permutation repeats as context; the persisted mean is primary.
            ax.scatter(reps, np.full(len(reps), i), s=5, facecolor=BLUE["mid"],
                       edgecolor="none", alpha=0.35, zorder=2)
            ax.plot([0, mu], [i, i], color=NEUTRAL["light"], lw=0.6, zorder=1)
            ax.scatter([mu], [i], s=20, marker="o", facecolor=BLUE["dark"],
                       edgecolor=NEUTRAL["near_black"], linewidth=0.4, zorder=4)
            plotted.append({"figure": "FIG_E", "feature": f, "mean_importance": mu,
                            "std_importance": float(m["std_importance"]),
                            "n_repeats": int(m["n_repeats"])})

        ax.axvline(0, color=NEUTRAL["mid"], lw=0.7, zorder=1)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels([_pretty(m["feature"]) for m in ordered], fontsize=6.0)
        ax.set_ylim(-0.8, len(ordered) - 0.2)
        ax.set_xlabel("Balanced-accuracy drop (permuted)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # The marker legend (mean vs repeats), the model/assignment, and the
        # case-specific caveat are stated in the caption, not the plot area.
        fig.subplots_adjust(left=0.44, right=0.97, top=0.985, bottom=0.08)

    write_figure_data("FIG_E", plotted, record)
    record.checks = {
        "features": len(ordered),
        "repeats_per_feature": 30,
        "harmonic_features_shown": 0,
        "model": "random_forest",
        "assignment": "A00",
        "scoring_metric": "balanced_accuracy",
    }
    return fig, record


def build_fig_f(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_F"]
    pairs = read_csv(JACCARD)
    freq = read_csv(TOPK_FREQ)
    assert len(pairs) == 256, f"expected 16x16=256 pairs, got {len(pairs)}"

    views = [f"A{i:02d}" for i in range(16)]
    idx = {v: i for i, v in enumerate(views)}
    M = np.full((16, 16), np.nan)
    for p in pairs:
        M[idx[p["assignment_a"]], idx[p["assignment_b"]]] = float(p["jaccard"])
    assert not np.isnan(M).any(), "incomplete Jaccard matrix"
    assert np.allclose(np.diag(M), 1.0), "diagonal must be self-similarity 1.0"
    assert np.allclose(M, M.T), "Jaccard matrix must be symmetric"

    # The 120 unique unordered off-diagonal pairs (upper triangle).
    tri = M[np.triu_indices(16, k=1)]
    assert len(tri) == 120

    record = FigureRecord("FIG_F", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [JACCARD, TOPK_FREQ])
    record.display_transforms = [
        "long-form pairwise table pivoted to a 16x16 matrix (reshape only)",
        "upper triangle extracted to avoid double-counting symmetric pairs",
    ]

    plotted = [{"figure": "FIG_F", "panel": "a", "assignment_a": p["assignment_a"],
                "assignment_b": p["assignment_b"], "jaccard": float(p["jaccard"])} for p in pairs]

    with style_for(spec["export_profile"]):
        fig, (axl, axr) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1.0]})

        cmap = mpl_blue_cmap()
        im = axl.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="upper")
        axl.set_xticks(range(16))
        axl.set_yticks(range(16))
        axl.set_xticklabels(views, fontsize=5.4, rotation=90)
        axl.set_yticklabels(views, fontsize=5.4)
        axl.set_xlabel("Assignment view")
        axl.set_ylabel("Assignment view")
        for s in axl.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
        cb = fig.colorbar(im, ax=axl, fraction=0.042, pad=0.04)
        cb.set_label("Top-10 Jaccard similarity", fontsize=6.5, labelpad=2)
        cb.ax.tick_params(labelsize=5.8, length=2)
        cb.outline.set_linewidth(0.5)

        # Panel (b): distribution of the 120 unique pairs. Dot histogram keeps
        # every pair visible and needs no density estimate.
        bins = np.linspace(0, 1, 21)
        which = np.digitize(tri, bins) - 1
        counts: dict[int, int] = {}
        for b, v in zip(which, tri):
            k = counts.get(b, 0)
            axr.scatter([v], [k + 0.5], s=9, facecolor=BLUE["mid"], edgecolor=BLUE["dark"],
                        linewidth=0.3, zorder=3)
            counts[b] = k + 1
        top = max(counts.values())
        med = float(np.median(tri))
        axr.axvline(med, color=NEUTRAL["near_black"], lw=1.2, zorder=4)
        axr.annotate(f"median {med:.2f}", xy=(med, top + 1.5), xytext=(3, 0),
                     textcoords="offset points", ha="left", va="bottom", fontsize=6.4,
                     color=NEUTRAL["near_black"])
        axr.axvline(1.0, color=YELLOW["dark"], lw=0.9, zorder=2)
        axr.set_xlim(0, 1.06)
        axr.set_ylim(0, top + 8)
        axr.set_xlabel("Top-10 Jaccard similarity")
        axr.set_ylabel("Assignment-view pairs")
        axr.set_title("All 120 unique pairs", fontsize=7.2, pad=5)
        for v in tri:
            plotted.append({"figure": "FIG_F", "panel": "b", "assignment_a": "",
                            "assignment_b": "", "jaccard": float(v)})

        # The definition of k, the similarity metric, the view dependence, and the
        # meaning of low/high similarity are stated in the caption.
        fig.subplots_adjust(left=0.095, right=0.98, top=0.92, bottom=0.17, wspace=0.62)

    write_figure_data("FIG_F", plotted, record)
    record.checks = {
        "pairs": len(pairs),
        "unique_offdiagonal_pairs": int(len(tri)),
        "k": 10,
        "similarity_metric": "jaccard",
        "median_offdiagonal": round(med, 4),
        "min_offdiagonal": round(float(tri.min()), 4),
        "max_offdiagonal": round(float(tri.max()), 4),
        "symmetric": True,
    }
    return fig, record


def mpl_blue_cmap():
    """Sequential colormap built only from the approved blue_ribbon_v1 ramp."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "blue_ribbon_v1",
        [NEUTRAL["near_white"], BLUE["light"], BLUE["mid"], BLUE["mid_highlight"],
         BLUE["dark"], BLUE["dark_highlight"]],
    )
