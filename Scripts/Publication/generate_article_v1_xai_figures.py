"""Render the ArticleV1 minimum XAI figures from persisted tables.

This module is deliberately a plotting-only boundary: it does not load models,
fit estimators, or calculate explanations.  The three input tables are the
provenance-bearing outputs of the frozen XAI execution stage.
"""
# Compact plotting statements are intentionally kept close to panel definitions.
# ruff: noqa: E501, E701, E702, B009

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustworthy_agent.provenance.hashing import sha256_file
from trustworthy_agent.publication.figure_style import FigureStyle, publication_style

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "Output/ArticleV1/Tables"
OUT = ROOT / "Output/ArticleV1/Figures/XAI"
STYLE = FigureStyle()


def _column(df: pd.DataFrame, names: list[str], default: str | None = None) -> str:
    for name in names:
        if name in df.columns:
            return name
    if default is not None:
        return default
    raise ValueError(f"None of columns {names!r} found; available={list(df.columns)!r}")


def _save(fig: plt.Figure, number: int, stem: str, data: pd.DataFrame, source: Path, title: str, caption: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / f"XAI_Figure_{number:02d}_{stem}"
    data.to_csv(base.with_name(base.name + "_data.csv"), index=False)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": STYLE.dpi})):
        fig.savefig(base.with_suffix("." + suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    metadata = {
        "figure_id": f"XAI_Figure_{number:02d}",
        "title": title,
        "caption": caption,
        "style_version": STYLE.version,
        "generator": "generate_article_v1_xai_figures.py",
        "source_artifact": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "figure_data": str(base.with_name(base.name + "_data.csv").relative_to(ROOT)),
    }
    base.with_name(base.name + "_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    base.with_name(base.name + "_caption.md").write_text(f"**{title}.** {caption}\n", encoding="utf-8")


def global_importance() -> None:
    src = TABLE / "minimal_xai_global_importance.csv"
    df = pd.read_csv(src)
    feature = _column(df, ["feature", "feature_name", "feature_id"])
    value = _column(df, ["importance_mean", "mean_importance", "importance", "score_decrease"])
    err = next((c for c in ("importance_std", "std_importance", "std") if c in df), None)
    d = df[[feature, value] + ([err] if err else [])].copy().sort_values(value).tail(10)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    y = np.arange(len(d))
    ax.barh(y, d[value], xerr=d[err] if err else None, color="#0072B2", alpha=0.85)
    ax.set_yticks(y, d[feature]); ax.set_xlabel("Decrease in held-out balanced accuracy"); ax.set_title("Global permutation importance (top 10)")
    ax.axvline(0, color="#666666", lw=0.8)
    _save(fig, 1, "global_permutation_importance", d.rename(columns={feature: "feature", value: "importance"}), src, "Global permutation importance", "Top ten features by held-out score decrease for the preregistered frozen Random Forest. Importance is descriptive and not causal.")


def local_contributions() -> None:
    src = TABLE / "minimal_xai_local_contributions.csv"
    df = pd.read_csv(src)
    feature = _column(df, ["feature", "feature_name", "feature_id"])
    value = _column(df, ["contribution", "logit_contribution", "additive_contribution"])
    d = df[[feature, value]].copy().sort_values(value)
    d = pd.concat([d.head(8), d.tail(8)]).drop_duplicates().sort_values(value)
    fig, ax = plt.subplots(figsize=(6.2, 4.2)); y = np.arange(len(d))
    ax.barh(y, d[value], color=np.where(d[value] >= 0, "#009E73", "#D55E00")); ax.set_yticks(y, d[feature]); ax.axvline(0, color="#1A1A1A", lw=0.8)
    ax.set_xlabel("Exact additive logistic-regression logit contribution"); ax.set_title("Local logistic-regression explanation")
    _save(fig, 2, "local_lr_contributions", d.rename(columns={feature: "feature", value: "contribution"}), src, "Local logistic-regression contributions", "Positive and negative contributions for the preregistered held-out window. These are exact additive model-logit terms, not generic SHAP values.")


def stability() -> None:
    src = TABLE / "minimal_xai_stability.csv"
    df = pd.read_csv(src)
    feature = _column(df, ["feature", "feature_name", "feature_id"])
    freq = _column(df, ["top10_frequency", "frequency", "occurrence_frequency", "count"])
    top = df[[feature, freq]].drop_duplicates().sort_values(freq).tail(10)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.1, 1.3]})
    axs[0].barh(np.arange(len(top)), top[freq], color="#0072B2"); axs[0].set_yticks(np.arange(len(top)), top[feature]); axs[0].set_xlabel("Assignments in top-10"); axs[0].set_title("Feature occurrence")
    matrix_src = OUT.parent.parent / "XAI/Minimal/ExplanationStability/jaccard_matrix.csv"
    matrix = pd.read_csv(matrix_src) if matrix_src.exists() else df
    left = _column(matrix, ["assignment_a", "assignment_id", "assignment"])
    right = _column(matrix, ["assignment_b", "assignment_id_b"], default=left)
    mat_col = _column(matrix, ["jaccard", "jaccard_similarity", "pairwise_jaccard"], default="")
    if mat_col and {left, right}.issubset(matrix.columns):
        ids = sorted(set(matrix[left].astype(str)) | set(matrix[right].astype(str)))
        m = np.eye(len(ids)); pos = {x: i for i, x in enumerate(ids)}
        for row in matrix.itertuples(index=False):
            values = row._asdict()
            m[pos[str(values[left])], pos[str(values[right])]] = values[mat_col]
        axs[1].imshow(m, vmin=0, vmax=1, cmap="Blues"); axs[1].set_title("Top-10 Jaccard similarity"); axs[1].set_xlabel("Assignment"); axs[1].set_ylabel("Assignment")
    else:
        axs[1].axis("off"); axs[1].text(0.5, 0.5, "Jaccard matrix not present\nin stability table", ha="center", va="center")
    fig.tight_layout(); _save(fig, 3, "explanation_stability", top, src, "Explanation stability", "Descriptive stability across the 16 assignment views. The assignments reuse eight acquisitions and are not independent replicates; no inferential p-values are reported.")


def main() -> None:
    with publication_style(STYLE):
        global_importance(); local_contributions(); stability()


if __name__ == "__main__":
    main()
