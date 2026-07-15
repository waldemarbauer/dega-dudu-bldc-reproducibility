"""Shared infrastructure for the DEGA M3 computational figure pipeline.

This module provides the loader, style, and export layer. It deliberately does
not compute scientific quantities: every value reaching a figure is read from a
frozen ArticleV1 artifact. The only permitted operations here are selection,
reshaping, ordering, and exact display transforms, each of which is recorded in
the data-provenance matrix.

Dependencies are restricted to the project's declared runtime stack (numpy,
matplotlib, pyyaml) plus the standard library, so the figure build reproduces
under the same environment as the pipeline itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "figures_m3/config/figure-config.yaml"
STYLE_DIR = ROOT / "figures_m3/config/styles"
OUTPUT_ROOT = ROOT / "figures_m3/output"
MANIFEST_PATH = ROOT / "figures_m3/data_manifest/figure_manifest.yaml"

# Palette (scientific-figures skill, palette.yaml: blue_ribbon_v1 / neutral_v1 /
# yellow_ribbon_v1). No colour outside these versioned palettes is used.
BLUE = {
    "light": "#B3FFFF",
    "light_highlight": "#9AF6FF",
    "mid": "#67C3FF",
    "mid_highlight": "#3490CC",
    "dark": "#015D99",
    "dark_highlight": "#002A66",
}
NEUTRAL = {
    "white": "#FFFFFF",
    "near_white": "#F5F5F5",
    "light": "#D9D9D9",
    "mid": "#8C8C8C",
    "dark": "#4D4D4D",
    "near_black": "#1A1A1A",
    "black": "#000000",
}
YELLOW = {"mid": "#FDED2A", "mid_highlight": "#FFE41C", "dark": "#EECA02"}

# Redundant (non-colour) encodings, so every semantic distinction survives
# grayscale and colour-vision deficiency.
CLASSIFIER_COLOR = {
    "logistic_regression": BLUE["dark_highlight"],
    "random_forest": BLUE["mid_highlight"],
    "hist_gradient_boosting": NEUTRAL["mid"],
}
CLASSIFIER_MARKER = {
    "logistic_regression": "o",
    "random_forest": "s",
    "hist_gradient_boosting": "^",
}
CLASSIFIER_LINESTYLE = {
    "logistic_regression": "-",
    "random_forest": "--",
    "hist_gradient_boosting": ":",
}


def load_config() -> dict[str, Any]:
    """Return the M3 figure configuration."""
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a frozen CSV artifact as a list of string-valued rows.

    Values are intentionally left as strings; each figure converts only the
    columns it displays, so a silent type coercion cannot alter a plotted value.
    """
    full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(f"Frozen artifact missing: {path}")
    with full.open(newline="") as fh:
        return list(csv.DictReader(fh))


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file, for provenance recording."""
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def order_by(rows: list[dict], key: str, order: list[str]) -> list[dict]:
    """Sort rows into a canonical ordering; unknown keys sort last, stably."""
    rank = {v: i for i, v in enumerate(order)}
    return sorted(rows, key=lambda r: rank.get(r[key], len(order)))


@dataclass
class FigureRecord:
    """Provenance record emitted for one built figure."""

    figure_id: str
    title: str
    export_profile: str
    width_in: float
    height_in: float
    source_artifacts: list[dict[str, str]] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    figure_data: str | None = None
    figure_data_sha256: str | None = None
    scientific_transform: str = "none"
    display_transforms: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)


def style_for(export_profile: str):
    """Return the Matplotlib style context for a skill export profile."""
    asset = {
        "publication-single-column": "publication-single-column.mplstyle",
        "publication-double-column": "publication-double-column.mplstyle",
        "publication-full-page": "publication-full-page.mplstyle",
    }[export_profile]
    return mpl.style.context(STYLE_DIR / asset)


def export(fig, figure_id: str, spec: dict, record: FigureRecord) -> None:
    """Export one figure to vector PDF, 300-dpi PNG, and SVG at exact size.

    The figure size is asserted against the configured physical dimensions so a
    layout engine cannot silently change the manuscript-scale geometry.
    """
    w, h = spec["dimensions"]["width_in"], spec["dimensions"]["height_in"]
    fig.set_size_inches(w, h)
    got_w, got_h = fig.get_size_inches()
    if abs(got_w - w) > 1e-6 or abs(got_h - h) > 1e-6:
        raise AssertionError(f"{figure_id}: size {got_w}x{got_h} != configured {w}x{h}")

    stem = f"Figure_{figure_id.split('_')[-1]}"
    targets = {
        "pdf": OUTPUT_ROOT / "pdf" / f"{stem}.pdf",
        "png": OUTPUT_ROOT / "png" / f"{stem}.png",
        "svg": OUTPUT_ROOT / "svg" / f"{stem}.svg",
    }
    for fmt, path in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        # fonttype 42 keeps PDF/SVG text as embedded, selectable vector text.
        with mpl.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
            fig.savefig(path, format=fmt, dpi=300 if fmt == "png" else None)
        record.outputs[fmt] = str(path.relative_to(ROOT))

    record.width_in, record.height_in = w, h


def write_figure_data(figure_id: str, rows: list[dict], record: FigureRecord) -> None:
    """Persist the exact values plotted, so every mark is auditable."""
    path = OUTPUT_ROOT / "figure_data" / f"{figure_id}_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record.figure_data = str(path.relative_to(ROOT))
    record.figure_data_sha256 = sha256_file(path)


def attach_sources(record: FigureRecord, paths: list[str]) -> None:
    """Record the SHA-256 of every source artifact consumed by a figure."""
    for p in paths:
        if "*" in p:
            continue
        if (ROOT / p).exists():
            record.source_artifacts.append({"path": p, "sha256": sha256_file(p)})


def write_manifest(records: list[FigureRecord], config: dict) -> None:
    """Write the generated provenance manifest (agent-managed state)."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "milestone": "M3",
        "study": "ArticleV1",
        "generated_by": "figures_m3/scripts/make_all_figures.py",
        "evaluation_units": config["evaluation_units"],
        "figures": [
            {
                "figure_id": r.figure_id,
                "title": r.title,
                "export_profile": r.export_profile,
                "dimensions_in": {"width": r.width_in, "height": r.height_in},
                "scientific_transform": r.scientific_transform,
                "display_transforms": r.display_transforms,
                "source_artifacts": r.source_artifacts,
                "figure_data": r.figure_data,
                "figure_data_sha256": r.figure_data_sha256,
                "outputs": r.outputs,
                "checks": r.checks,
            }
            for r in records
        ],
    }
    with MANIFEST_PATH.open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
