"""Compute the frozen, minimal ArticleV1 XAI package without fitting models."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
FEATURES = (
    ROOT / "Data/AnalysisData/ArticleV1/WindowFeatures/article_v1_canonical_window_features.parquet"
)
MANIFEST = ROOT / "Output/ArticleV1/Manifests/article_window_feature_manifest.json"
MODEL_ROOT = ROOT / "Output/ArticleV1/Models/WindowClassifiers"
ASSIGN_ROOT = ROOT / "Data/AnalysisData/ArticleV1/Assignments"
CFG = ROOT / "configs/xai/article_v1/minimal_xai_v1.yaml"
FEATURE_COUNT = 28


def fast_permutation_importance(model, x: np.ndarray, y: np.ndarray, *, repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute deterministic permutation drops with batched persisted-model calls."""
    rng = np.random.default_rng(seed)
    baseline = balanced_accuracy_score(y, model.predict(x))
    drops = np.empty((repeats, x.shape[1]), dtype=float)
    for rep in range(repeats):
        batch = np.repeat(x[None, :, :], x.shape[1], axis=0)
        for feature in range(x.shape[1]):
            batch[feature, :, feature] = rng.permutation(x[:, feature])
        predicted = model.predict(batch.reshape(-1, x.shape[1]))
        drops[rep] = [
            baseline - balanced_accuracy_score(y, predicted[i * len(y) : (i + 1) * len(y)])
            for i in range(x.shape[1])
        ]
    return drops.mean(axis=0), drops.std(axis=0), drops


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    fm = json.loads(MANIFEST.read_text())
    features = list(fm["feature_order"])
    assert len(features) == FEATURE_COUNT
    frame = pd.read_parquet(FEATURES)
    by_id = frame.set_index("window_id")
    out = ROOT / "Output/ArticleV1"
    for d in (
        "XAI/Minimal/GlobalPermutationImportance",
        "XAI/Minimal/LocalLogisticContributions",
        "XAI/Minimal/ExplanationStability",
        "Figures/XAI",
        "Tables",
        "Manifests",
        "Reports",
    ):
        (out / d).mkdir(parents=True, exist_ok=True)

    def held(assignment: str):
        a = json.loads((ASSIGN_ROOT / f"{assignment}.json").read_text())
        rows = by_id.loc[a["held_out_window_ids"]]
        return rows, a

    # Global RF permutation importance, persisted model only.
    gid = cfg["global"]["assignment_id"]
    grow, ga = held(gid)
    gx = grow[features].to_numpy()
    gy = grow["canonical_class"].to_numpy()
    gmodel_path = MODEL_ROOT / gid / "random_forest/model.joblib"
    gmodel = joblib.load(gmodel_path)
    global_mean, global_std, global_drops = fast_permutation_importance(
        gmodel, gx, gy, repeats=30, seed=int(cfg["global"]["random_seed"])
    )
    rows = []
    for i, f in enumerate(features):
        rows.append(
            {
                "assignment_id": gid,
                "classifier_id": "random_forest",
                "feature": f,
                "feature_index": i,
            "mean_importance": float(global_mean[i]),
            "std_importance": float(global_std[i]),
                "n_repeats": 30,
                "scoring_metric": "balanced_accuracy",
            }
        )
    gdf = pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
    gdf.to_csv(out / "Tables/minimal_xai_global_importance.csv", index=False)
    gdf.to_csv(out / "XAI/Minimal/GlobalPermutationImportance/distributions.csv", index=False)
    pd.DataFrame(global_drops, columns=features).assign(repeat=np.arange(30)).melt(
        id_vars="repeat", var_name="feature", value_name="importance_drop"
    ).to_csv(out / "XAI/Minimal/GlobalPermutationImportance/repeat_distributions.csv", index=False)

    # Exact local LR additive logits, selected canonical held-out index 12.
    lrow, la = held(cfg["local"]["assignment_id"])
    sample = lrow.iloc[int(cfg["local"]["held_out_window_index"])]
    lmodel_path = MODEL_ROOT / "A00/logistic_regression/model.joblib"
    lmodel = joblib.load(lmodel_path)
    raw = sample[features].to_numpy(dtype=float).reshape(1, -1)
    pipe = lmodel
    transformed = pipe.named_steps["imputer"].transform(raw)
    scaled = pipe.named_steps["scaler"].transform(transformed)
    clf = pipe.named_steps["classifier"]
    scores_vec = np.asarray(clf.decision_function(scaled)).reshape(-1)
    pred_idx = int(np.argmax(scores_vec))
    pred_class = str(clf.classes_[pred_idx])
    contrib = scaled[0] * clf.coef_[pred_idx]
    intercept = float(clf.intercept_[pred_idx])
    recon = intercept + float(contrib.sum())
    model_score = float(scores_vec[pred_idx])
    ldf = pd.DataFrame(
        {
            "assignment_id": cfg["local"]["assignment_id"],
            "acquisition_id": sample["acquisition_id"],
            "window_id": sample.name,
            "target_class": pred_class,
            "feature": features,
            "transformed_feature_value": scaled[0],
            "coefficient": clf.coef_[pred_idx],
            "contribution": contrib,
            "intercept": intercept,
            "reconstructed_score": recon,
            "model_score": model_score,
            "residual": recon - model_score,
        }
    )
    ldf.to_csv(out / "Tables/minimal_xai_local_contributions.csv", index=False)
    ldf.to_csv(out / "XAI/Minimal/LocalLogisticContributions/contributions.csv", index=False)

    # Stability across all assignments (dependent views of eight acquisitions).
    rankings = {}
    stability_rows = []
    for aid in cfg["stability"]["assignment_ids"]:
        r, _ = held(aid)
        model = joblib.load(MODEL_ROOT / aid / "random_forest/model.joblib")
        stability_mean, _, _ = fast_permutation_importance(
            model, r[features].to_numpy(), r["canonical_class"].to_numpy(), repeats=30, seed=1729
        )
        order = np.argsort(-stability_mean)
        top = [features[i] for i in order[:10]]
        rankings[aid] = top
        for rank, f in enumerate(top, 1):
            stability_rows.append({"assignment_id": aid, "feature": f, "rank": rank})
    freq = (
        pd.DataFrame(stability_rows).groupby("feature").size().rename("top10_count").reset_index()
    )
    freq["top10_frequency"] = freq.top10_count / 16
    freq.to_csv(out / "Tables/minimal_xai_stability.csv", index=False)
    freq.to_csv(out / "XAI/Minimal/ExplanationStability/top10_frequency.csv", index=False)
    pairs = []
    ids = list(rankings)
    for a in ids:
        for b in ids:
            pairs.append(
                {
                    "assignment_a": a,
                    "assignment_b": b,
                    "jaccard": len(set(rankings[a]) & set(rankings[b]))
                    / len(set(rankings[a]) | set(rankings[b])),
                }
            )
    pd.DataFrame(pairs).to_csv(
        out / "XAI/Minimal/ExplanationStability/jaccard_matrix.csv", index=False
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    top = gdf.head(10).iloc[::-1]
    ax.barh(top.feature, top.mean_importance)
    ax.set_xlabel("Balanced-accuracy decrease")
    fig.tight_layout()
    [
        fig.savefig(
            out / f"Figures/XAI/XAI_Figure_01_global_permutation_importance.{e}",
            dpi=600 if e == "png" else None,
        )
        for e in ("pdf", "svg", "png")
    ]
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    z = ldf.sort_values("contribution").tail(10)
    ax.barh(z.feature, z.contribution)
    ax.set_xlabel("Exact additive logit contribution")
    fig.tight_layout()
    [
        fig.savefig(
            out / f"Figures/XAI/XAI_Figure_02_local_lr_contributions.{e}",
            dpi=600 if e == "png" else None,
        )
        for e in ("pdf", "svg", "png")
    ]
    plt.close(fig)
    mat = np.array(
        [
            [
                len(set(rankings[a]) & set(rankings[b])) / len(set(rankings[a]) | set(rankings[b]))
                for b in ids
            ]
            for a in ids
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(16), ids, rotation=90)
    ax.set_yticks(range(16), ids)
    fig.colorbar(im, label="Top-10 Jaccard")
    fig.tight_layout()
    [
        fig.savefig(
            out / f"Figures/XAI/XAI_Figure_03_explanation_stability.{e}",
            dpi=600 if e == "png" else None,
        )
        for e in ("pdf", "svg", "png")
    ]
    plt.close(fig)
    manifest = {
        "global_model_sha256": sha(gmodel_path),
        "local_model_sha256": sha(lmodel_path),
        "feature_corpus_sha256": sha(FEATURES),
        "global_assignment_sha256": sha(ASSIGN_ROOT / f"{gid}.json"),
        "local_assignment_sha256": sha(ASSIGN_ROOT / f"{cfg['local']['assignment_id']}.json"),
        "feature_schema_hash": fm["feature_schema_hash"],
        "feature_count": 28,
        "assignments": ids,
        "n_repeats": 30,
        "scoring_metric": "balanced_accuracy",
        "local_residual": float(abs(recon - model_score)),
    }
    (out / "Manifests/minimal_xai_execution_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    (out / "Manifests/minimal_xai_registry.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "artifacts": [
                    "minimal_xai_global_importance.csv",
                    "minimal_xai_local_contributions.csv",
                    "minimal_xai_stability.csv",
                ],
            },
            indent=2,
        )
    )
    (out / "Reports/minimal_xai_execution_report.md").write_text(
        f"# Minimal XAI execution\n\nGlobal RF assignment A00, balanced accuracy, 30 repeats. Local LR window {sample['window_id']}; residual {abs(recon - model_score):.3e}. Stability covers 16 dependent assignment views. No SHAP/PDP/ALE/ICE or causal interpretation.\n"
    )
    (out / "Reports/minimal_xai_validation_report.md").write_text(
        "# Validation\n\nPersisted models only; 28 features retained; held-out partitions used; deterministic seeds and source hashes recorded.\n"
    )
    (out / "Reports/minimal_xai_claims_boundary.md").write_text(
        "# Claims boundary\n\nPermutation importance is model-specific and non-causal. Local values are exact additive logistic-regression logits. The 16 assignment views reuse eight acquisitions and are not independent replicates.\n"
    )


if __name__ == "__main__":
    main()
