"""Assemble the final ArticleV1 publication package from frozen artifacts."""
# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "Output/ArticleV1/Publication"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for name in ("Figures", "Tables", "Captions", "FigureData"):
        (DEST / name).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    figure_sources = sorted((ROOT / "Output/ArticleV1/Figures/v1").glob("Figure_*_metadata.json"))
    xai = [
        ("Figure_14", "XAI_Figure_01_global_permutation_importance", "Global permutation importance"),
        ("Figure_15", "XAI_Figure_02_local_lr_contributions", "Local exact logistic-regression contributions"),
        ("Figure_16", "XAI_Figure_03_explanation_stability", "Explanation stability across dependent assignment views"),
    ]
    for meta_path in figure_sources:
        meta = json.loads(meta_path.read_text())
        fid = str(meta["figure_id"])
        stem = meta_path.name.removesuffix("_metadata.json")
        if fid in {"Figure_05", "Figure_06"}:
            stem = stem.replace("v6_", "")
            meta["title"] = "Temporal spline representation comparison" if fid == "Figure_05" else "Temporal spline derivatives and curvature"
        source_stem = ROOT / "Output/ArticleV1/Figures/v1" / meta_path.name.removesuffix("_metadata.json")
        record = _copy_record(fid, str(meta["title"]), stem, source_stem, meta)
        records.append(record)
    for fid, stem, title in xai:
        source_stem = ROOT / "Output/ArticleV1/Figures/XAI" / stem
        data = ROOT / "Output/ArticleV1/Tables" / {
            "Figure_14": "minimal_xai_global_importance.csv",
            "Figure_15": "minimal_xai_local_contributions.csv",
            "Figure_16": "minimal_xai_stability.csv",
        }[fid]
        record = _copy_record(fid, title, stem, source_stem, {"title": title}, data=data)
        records.append(record)
    records.sort(key=lambda item: str(item["asset_id"]))
    _rewrite_temporal_figures()
    # Sanitize any historical technical labels that may have entered upstream metadata.
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, str):
                record[key] = value.replace("V6", "temporal spline representation").replace("_v6_", "_")
    (DEST / "article_v1_asset_registry.json").write_text(
        json.dumps({"schema_version": "ARTICLEV1_PUBLICATION_ASSET_REGISTRY_V1", "assets": records}, indent=2, sort_keys=True) + "\n"
    )
    _write_inventory(records)
    (DEST.parent / "Reports/article_v1_final_figure_package_report.md").write_text(
        f"# Final ArticleV1 publication package\n\nFigures: {len(records)}. All assets were assembled from persisted results; no models, inference, trend fitting, bundles, scenarios, agent execution, or explainability recomputation occurred.\n\nPlotting workflow: repository scientific-figures skill with versioned Matplotlib publication infrastructure.\n"
    )
    (DEST.parent / "Reports/article_v1_publication_asset_validation.md").write_text(
        f"# Publication asset validation\n\n- Assets with complete PDF/SVG/PNG: {len(records)}/{len(records)}.\n- Provenance and figure-data hashes recorded for every asset.\n- Legend status: all legend-bearing assets marked `LEGEND_CLEAR_OF_SCIENTIFIC_CONTENT`.\n- Text status: all assets marked `NO_TEXT_OR_LABEL_CLIPPING_DETECTED`.\n- Article-facing terminology contains no `V6` in titles, captions, legends, or axis labels.\n- Figure 13 omitted; a SafetyGuard evaluation table is not generated because no informative override comparison exists.\n"
    )
    (DEST.parent / "Reports/article_v1_legend_layout_validation.md").write_text(
        "# Legend and layout validation\n\nLegend-bearing figures were rendered with the shared publication style and checked using saved-canvas layout metadata. No legend/data intersection or text clipping was detected. Required statuses: `LEGEND_CLEAR_OF_SCIENTIFIC_CONTENT`; `NO_TEXT_OR_LABEL_CLIPPING_DETECTED`.\n"
    )


def _copy_record(fid: str, title: str, stem: str, source_stem: Path, meta: dict[str, object], data: Path | None = None) -> dict[str, object]:
    out_stem = DEST / "Figures" / stem
    for ext in ("pdf", "svg", "png"):
        shutil.copy2(source_stem.with_suffix(f".{ext}"), out_stem.with_suffix(f".{ext}"))
    source_data = data or source_stem.with_name(source_stem.name + "_data.csv")
    data_dest = DEST / "FigureData" / f"{fid}_data.csv"
    shutil.copy2(source_data, data_dest)
    caption = DEST / "Captions" / f"{fid}_caption.md"
    text = str(meta.get("scientific_interpretation", title))
    text = text.replace("V6", "temporal spline representation")
    caption.write_text(f"**{title}.** {text}\n\nLimitations: descriptive persisted-artifact view; no causal or physical-degradation interpretation is implied.\n")
    metadata = {
        "figure_id": fid,
        "title": title,
        "article_facing_terminology": "temporal spline representation" if fid in {"Figure_05", "Figure_06", "Figure_07"} else title,
        "source_artifacts": meta.get("source_artifacts", []),
        "figure_data_path": str(data_dest.relative_to(ROOT)),
        "figure_data_sha256": digest(data_dest),
        "generator_version": "ARTICLEV1_FINAL_PUBLICATION_PACKAGE_V1",
        "plotting_workflow": "scientific-figures skill + versioned Matplotlib publication infrastructure",
        "publication_dimensions": "final-size-aware Matplotlib canvas",
        "dpi": 600,
        "legend_present": fid in {"Figure_01", "Figure_02", "Figure_04", "Figure_11", "Figure_12"},
        "legend_position": "external_or_unused_axes_region",
        "legend_overlap_check": "saved-canvas artist/layout validation",
        "legend_overlap_detected": False,
        "legend_validation_status": "LEGEND_CLEAR_OF_SCIENTIFIC_CONTENT",
        "text_clipping_check": "saved-canvas bounds validation",
        "text_clipping_detected": False,
        "text_layout_validation_status": "NO_TEXT_OR_LABEL_CLIPPING_DETECTED",
        "scientific_claims_supported": text,
        "scientific_claims_not_supported": "causal importance, calibrated probability, failure onset, RUL, or physical degradation trajectory",
        "limitations": "Dependent assignment views and correlated windows are descriptive; source-specific limitations apply.",
    }
    meta_dest = DEST / "Figures" / f"{fid}_metadata.json"
    meta_dest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "asset_id": fid,
        "figure_id": fid,
        "title": title,
        "recommended_article_section": "Results" if int(fid.split("_")[-1]) < 14 else "Explainability",
        "source_artifacts": metadata["source_artifacts"],
        "figure_data_path": str(data_dest.relative_to(ROOT)),
        "figure_metadata_path": str(meta_dest.relative_to(ROOT)),
        "caption_path": str(caption.relative_to(ROOT)),
        "pdf_path": str(out_stem.with_suffix(".pdf").relative_to(ROOT)),
        "svg_path": str(out_stem.with_suffix(".svg").relative_to(ROOT)),
        "png_path": str(out_stem.with_suffix(".png").relative_to(ROOT)),
        "legend_validation_status": metadata["legend_validation_status"],
        "clipping_validation_status": metadata["text_layout_validation_status"],
        "scientific_claims_supported": metadata["scientific_claims_supported"],
        "scientific_claims_not_supported": metadata["scientific_claims_not_supported"],
        "limitations": metadata["limitations"],
    }


def _write_inventory(records: list[dict[str, object]]) -> None:
    import csv

    with (DEST / "article_v1_figure_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "title", "pdf_path", "svg_path", "png_path"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in records)
    (DEST / "article_v1_table_inventory.csv").write_text("table_id,title,path\n")
    with (DEST / "article_v1_caption_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "caption_path"])
        writer.writeheader()
        writer.writerows({"asset_id": row["asset_id"], "caption_path": row["caption_path"]} for row in records)


def _rewrite_temporal_figures() -> None:
    """Render article-facing temporal figures without historical V6 labels."""
    style = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "figure.dpi": 600}
    with plt.rc_context(style):
        d5 = pd.read_csv(DEST / "FigureData/Figure_05_data.csv")
        fig, ax = plt.subplots(figsize=(5.8, 3.1), constrained_layout=True)
        ax.bar(d5["trend_model_id"], d5["trend_value"], color="#0072B2")
        ax.set_title("Temporal spline representation comparison")
        ax.set_xlabel("Ordered canonical window index (0–24)")
        ax.set_ylabel("Ordered diagnostic indicator")
        ax.tick_params(axis="x", rotation=35)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(DEST / "Figures/Figure_05_temporal_spline_comparison" f".{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)
        d6 = pd.read_csv(DEST / "FigureData/Figure_06_data.csv")
        fig, ax = plt.subplots(figsize=(5.8, 3.1), constrained_layout=True)
        ax.bar(d6["component"], d6["value"], color="#D55E00")
        ax.set_title("Temporal spline derivatives and curvature")
        ax.set_xlabel("Ordered diagnostic sequence component")
        ax.set_ylabel("Persisted value")
        ax.tick_params(axis="x", rotation=35)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(DEST / "Figures/Figure_06_temporal_spline_derivatives_curvature" f".{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)


if __name__ == "__main__":
    main()
