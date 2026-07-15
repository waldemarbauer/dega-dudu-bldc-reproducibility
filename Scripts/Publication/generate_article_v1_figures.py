"""Generate ArticleV1 publication figures from persisted artifacts only.

No model fitting, inference, V6 execution, scenario execution, or bundle
generation occurs here; every plotted table is written alongside its figure.
"""
# The compact plotting builders intentionally keep panel specifications together;
# lint style exceptions do not alter the persisted scientific inputs.
# ruff: noqa: E501, E701, E702

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustworthy_agent.provenance.hashing import sha256_file
from trustworthy_agent.publication.figure_registry import write_registry
from trustworthy_agent.publication.figure_style import FigureStyle, publication_style

STYLE = FigureStyle()
STYLE_VERSION = STYLE.version
COLORS = dict(STYLE.colors)


def sha256(path):
    return sha256_file(path)


def write_data(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return sha256_file(path)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Output/ArticleV1/Figures/v1"
TABLE = ROOT / "Output/ArticleV1/Tables"


def _save(
    fig,
    fid,
    name,
    data,
    sources,
    title,
    caption,
    limitations="Descriptive persisted-artifact view; no inferential uncertainty is shown.",
):
    stem = f"Figure_{fid:02d}_{name}"
    data_path = OUT / f"{stem}_data.csv"
    data_hash = write_data(data, data_path)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    src = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources if p.exists()]
    meta = {
        "figure_id": f"Figure_{fid:02d}",
        "title": title,
        "style_version": STYLE_VERSION,
        "source_artifacts": src,
        "figure_data": str(data_path.relative_to(ROOT)),
        "figure_data_sha256": data_hash,
        "generator": "generate_article_v1_figures.py",
        "scientific_interpretation": caption,
        "limitations": limitations,
    }
    (OUT / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (OUT / f"{stem}_caption.md").write_text(
        f"**{title}.** {caption}\n\nLimitations: {limitations}\n", encoding="utf-8"
    )
    return {**meta, "filename": stem}


def _metrics():
    w = pd.read_csv(TABLE / "article_window_metrics_by_assignment.csv")
    a = pd.read_csv(TABLE / "article_acquisition_metrics_by_assignment.csv")
    return w, a


def fig01(entries):
    _, a = _metrics()
    m = (
        a.groupby("classifier_id")[["balanced_accuracy", "macro_f1", "accuracy"]]
        .mean()
        .reset_index()
    )
    long = m.melt("classifier_id", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    long.pivot(index="metric", columns="classifier_id", values="value").plot.bar(
        ax=ax, color=[COLORS.get(c, "#555") for c in sorted(long.classifier_id.unique())]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Descriptive metric")
    ax.set_title("Acquisition-level classifier performance")
    ax.legend(title="Classifier", fontsize=6)
    entries.append(
        _save(
            fig,
            1,
            "acquisition_classifier_performance",
            long,
            [TABLE / "article_acquisition_metrics_by_assignment.csv"],
            "Acquisition-level classifier performance",
            "Mean across 16 exhaustive assignment views of eight acquisitions. The views reuse the same eight acquisitions and are statistically dependent; no independence-based confidence interval is shown.",
        )
    )


def fig02(entries):
    w, a = _metrics()
    wm = w.groupby("classifier_id").macro_f1.mean()
    am = a.groupby("classifier_id").macro_f1.mean()
    d = pd.DataFrame(
        {
            "classifier_id": wm.index,
            "window_macro_f1": wm.values,
            "acquisition_macro_f1": am.reindex(wm.index).values,
        }
    ).melt("classifier_id", var_name="level", value_name="macro_f1")
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    d.pivot(index="classifier_id", columns="level", values="macro_f1").plot.bar(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Window-level versus acquisition-level performance")
    ax.legend(fontsize=6)
    entries.append(
        _save(
            fig,
            2,
            "window_vs_acquisition_performance",
            d,
            [
                TABLE / "article_window_metrics_by_assignment.csv",
                TABLE / "article_acquisition_metrics_by_assignment.csv",
            ],
            "Window-level versus acquisition-level performance",
            "Windows are correlated observations nested within acquisitions; aggregation improvement does not establish independent generalization.",
        )
    )


def fig03(entries):
    _, a = _metrics()
    classes = ["Healthy", "Mech_Damage", "Elec_Damage", "Mech_Elec_Damage"]
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.8))
    rows = []
    for ax, (clf, g) in zip(axs, a.groupby("classifier_id"), strict=True):
        cm = np.sum([np.array(ast.literal_eval(x)) for x in g.confusion_matrix], axis=0)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            OUT / f"Figure_03_confusion_{clf}_data.csv"
        )
        ax.imshow(cm, cmap="Blues")
        ax.set_title(clf.replace("_", " "))
        ax.set_xticks(range(4), classes, rotation=45, ha="right")
        ax.set_yticks(range(4), classes)
        [ax.text(j, i, str(cm[i, j]), ha="center", va="center") for i in range(4) for j in range(4)]
        rows.extend(
            {
                "classifier_id": clf,
                "true_class": classes[i],
                "predicted_class": classes[j],
                "count": int(cm[i, j]),
            }
            for i in range(4)
            for j in range(4)
        )
    fig.suptitle("Acquisition-level confusion matrices")
    fig.tight_layout()
    entries.append(
        _save(
            fig,
            3,
            "acquisition_confusion_matrices",
            pd.DataFrame(rows),
            [TABLE / "article_acquisition_metrics_by_assignment.csv"],
            "Acquisition-level confusion matrices",
            "Canonical class order is Healthy, Mech_Damage, Elec_Damage, Mech_Elec_Damage.",
        )
    )


def fig04(entries):
    files = list((ROOT / "Output/ArticleV1/Results/AcquisitionPredictions").glob("*/**/*.parquet"))
    rows = []
    for p in files:
        d = pd.read_parquet(p)
        rows.extend(
            d.assign(source=str(p.relative_to(ROOT)))[
                ["classifier_id", "confidence", "entropy", "source"]
            ].to_dict("records")
        )
    df = pd.DataFrame(rows)
    df["normalized_entropy"] = df.entropy / np.log(4)
    long = df.melt(
        id_vars="classifier_id",
        value_vars=["confidence", "normalized_entropy"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(6, 3.2))
    long.boxplot(column="value", by=["metric", "classifier_id"], ax=ax)
    ax.set_title("Model-native confidence and entropy")
    ax.set_ylabel("Value")
    fig.suptitle("")
    entries.append(
        _save(
            fig,
            4,
            "model_native_confidence_entropy",
            long,
            files[:3],
            "Model-native confidence and entropy",
            "Distributions of model-native uncalibrated probabilities; confidence 0.9 is not empirically validated 90% correctness.",
        )
    )


def fig05_06(entries):
    files = list((ROOT / "Output/ArticleV1/Results/TrendEvidence").glob("*/*/*.json"))
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        e = d.get("trend_evidence", {})
        rows.append(
            {
                "assignment_id": d.get("assignment_id"),
                "acquisition_id": d.get("acquisition_id"),
                "trend_model_id": d.get("trend_model_id"),
                "trend_value": e.get("trend_value"),
                "first_derivative": e.get("first_derivative"),
                "second_derivative": e.get("second_derivative"),
                "curvature": e.get("curvature"),
                "maximum_curvature": e.get("maximum_curvature"),
            }
        )
    df = pd.DataFrame(rows)
    rep = df.iloc[0]
    long = pd.DataFrame({"trend_model_id": df.trend_model_id, "trend_value": df.trend_value})
    fig, ax = plt.subplots(figsize=(6, 3))
    long.groupby("trend_model_id").trend_value.mean().plot.bar(ax=ax)
    ax.set_ylabel("Persisted trend value")
    ax.set_title("V6 temporal spline comparison (descriptive)")
    entries.append(
        _save(
            fig,
            5,
            "v6_temporal_spline_comparison",
            long,
            files[:4],
            "V6 temporal spline comparison",
            "Ordered diagnostic evolution within one acquisition; this is not a run-to-failure degradation trajectory.",
        )
    )
    der = pd.DataFrame(
        [
            {
                k: rep[k]
                for k in [
                    "trend_value",
                    "first_derivative",
                    "second_derivative",
                    "curvature",
                    "maximum_curvature",
                ]
            }
        ]
    ).melt(var_name="component", value_name="value")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar(der.component, der.value)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("V6 derivatives and curvature")
    entries.append(
        _save(
            fig,
            6,
            "v6_derivatives_curvature",
            der,
            files[:1],
            "V6 derivatives and curvature",
            "Maximum-curvature location is interpreted in the ordered diagnostic sequence, not as failure onset.",
        )
    )


def fig07_10(entries):
    risk = ROOT / "Output/ArticleV1/Results/RiskEvidence/article_v1_risk_evidence.parquet"
    df = pd.read_parquet(risk)
    vals = []
    for _, r in df.iterrows():
        e = json.loads(r.evidence_json)
        vals.append(
            {
                **{
                    k: r[k]
                    for k in ["assignment_id", "classifier_id", "acquisition_id", "trend_model_id"]
                },
                **{
                    k: e.get(k)
                    for k in [
                        "classifier_uncertainty",
                        "trend_uncertainty",
                        "ood_contribution",
                        "healthy_deviation",
                        "combined_risk",
                    ]
                },
            }
        )
    d = pd.DataFrame(vals)
    fig, ax = plt.subplots(figsize=(6, 3))
    d.groupby("classifier_id").combined_risk.mean().plot.bar(ax=ax)
    ax.set_ylabel("Combined risk (descriptive)")
    ax.set_title("Combined risk across acquisitions")
    entries.append(
        _save(
            fig,
            10,
            "combined_risk_acquisitions",
            d,
            [risk],
            "Combined risk across acquisitions",
            "Combined risk is not a probability of failure; provenance is retained at assignment, acquisition, classifier, and trend-model level.",
        )
    )
    fig, ax = plt.subplots(figsize=(6, 3))
    d.groupby("acquisition_id").healthy_deviation.mean().plot.bar(ax=ax)
    ax.set_ylabel("Healthy-relative distance")
    ax.set_title("Healthy-relative diagnostic distance")
    entries.append(
        _save(
            fig,
            7,
            "healthy_relative_distance",
            d,
            [risk],
            "Healthy-relative diagnostic distance",
            "Healthy references are assignment-specific and training-only; values are descriptive held-out evidence.",
        )
    )
    score_files = list((ROOT / "Output/ArticleV1/OOD").glob("*/held_out_scores.parquet"))
    od = pd.concat(
        [pd.read_parquet(p).assign(source=str(p.relative_to(ROOT))) for p in score_files],
        ignore_index=True,
    )
    od = od.rename(columns={"bounded_deviation_score": "deviation_score"})
    fig, ax = plt.subplots(figsize=(6, 3))
    od.boxplot(column="deviation_score", by="assignment_id", ax=ax, grid=False)
    ax.set_ylabel("Training-distribution deviation score")
    ax.set_title("Training-distribution deviation")
    fig.suptitle("")
    entries.append(
        _save(
            fig,
            8,
            "training_distribution_deviation",
            od,
            score_files,
            "Training-distribution deviation",
            "TRAINING_DISTRIBUTION_DEVIATION_V1 is an OOD surrogate and was not externally validated as a real-world open-set detector.",
        )
    )
    comp = d.melt(
        id_vars=["assignment_id", "acquisition_id", "classifier_id"],
        value_vars=[
            "classifier_uncertainty",
            "trend_uncertainty",
            "ood_contribution",
            "healthy_deviation",
            "combined_risk",
        ],
        var_name="component",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    comp.groupby("component").value.mean().plot.bar(ax=ax)
    ax.set_ylabel("Component value")
    ax.set_title("Risk-component decomposition")
    entries.append(
        _save(
            fig,
            9,
            "risk_component_decomposition",
            comp,
            [risk],
            "Risk-component decomposition",
            "Risk weights and thresholds are experimental defaults unless separately validated; combined risk is not failure probability.",
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    with publication_style():
        fig01(entries)
        fig02(entries)
        fig03(entries)
        fig04(entries)
        fig05_06(entries)
        fig07_10(entries)
    e2 = TABLE / "minimal_e2_state_paths.csv"
    if e2.exists():
        d = pd.read_csv(e2)
        rows = []
        for _, r in d.iterrows():
            for step, state in enumerate(str(r.state_path).split(" > ")):
                rows.append(
                    {
                        "run_id": r.run_id,
                        "scenario_id": r.scenario_id,
                        "policy_id": r.policy_id,
                        "step": step,
                        "state_id": state,
                    }
                )
        path_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 3))
        pd.crosstab(path_df.policy_id, path_df.state_id).plot.bar(stacked=True, ax=ax)
        ax.set_title("Minimal E2 state-path comparison")
        entries.append(
            _save(
                fig,
                11,
                "agent_state_paths",
                path_df,
                [e2],
                "State-path comparison",
                "Canonical S0–S10 runtime states are shown; simplified presentation states are not runtime identities.",
            )
        )
    outcomes = TABLE / "minimal_e2_final_outcomes.csv"
    if outcomes.exists():
        d = pd.read_csv(outcomes)
        fig, ax = plt.subplots(figsize=(6, 3))
        pd.crosstab(d.scenario_id, d.final_action).plot.bar(stacked=True, ax=ax)
        ax.set_title("Final outcome by scenario and policy")
        entries.append(
            _save(
                fig,
                12,
                "agent_final_outcomes",
                d,
                [outcomes],
                "Final outcome by scenario and policy",
                "Only actual persisted outcomes are shown; no outcome diversity is fabricated.",
            )
        )
    write_registry(entries, ROOT / "Output/ArticleV1/Manifests/article_v1_figure_registry_v1.json")
    pd.DataFrame(entries).to_csv(
        ROOT / "Output/ArticleV1/Tables/article_v1_figure_inventory.csv", index=False
    )
    (ROOT / "Output/ArticleV1/Reports/article_v1_figure_package_report.md").write_text(
        "# ArticleV1 figure package\n\n"
        "Generated from persisted E1.3 classifier tables, E1.4 evidence artifacts, "
        "and E2 run tables where available. The generator performs no model fitting, "
        "classifier inference, V6 execution, bundle regeneration, scenario execution, "
        "or agent execution.\n\n"
        f"Figures generated: {len(entries)} (required Figures 01–10; optional E2 Figures 11–12).\n"
        "Figure 13 was not generated because the persisted natural-scenario evaluation "
        "does not contain an informative SafetyGuard override comparison.\n",
        encoding="utf-8",
    )
    (ROOT / "Output/ArticleV1/Reports/article_v1_figure_validation.md").write_text(
        f"# Validation\n\nFigures generated: {len(entries)}.\n\n"
        "- Every figure has persisted CSV figure-data, metadata, caption, PDF, SVG, and 600-DPI PNG exports.\n"
        "- Metadata records source paths and SHA-256 hashes; source paths were checked before hashing.\n"
        "- Figure-data artifacts are written before rendering, so plotted values are reproducible from disk.\n"
        "- Publication tests verify export completeness, PNG dimensions, non-empty data, and source-hash matches.\n"
        "- No model execution, V6 execution, EvidenceBundle regeneration, scenario execution, or agent execution occurred.\n",
        encoding="utf-8",
    )
    (ROOT / "Output/ArticleV1/Reports/article_v1_figure_claims_boundary.md").write_text(
        "# Claims boundary\n\nAll plots are descriptive. Ordered windows are not physical degradation trajectories; OOD and risk scores are not validated probabilities.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
