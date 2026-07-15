"""Versioned, accessible matplotlib styling for paper-facing figures."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FigureStyle:
    """Publication style values shared by all ArticleV1 figure generators."""

    version: str = "ARTICLEV1_PUBLICATION_STYLE_V1"
    font_family: str = "DejaVu Sans"
    base_font_size: float = 8.0
    title_font_size: float = 9.0
    axis_label_size: float = 8.0
    tick_label_size: float = 7.0
    legend_font_size: float = 7.0
    line_width: float = 1.2
    marker_size: float = 4.0
    dpi: int = 600
    colors: Mapping[str, str] = field(
        default_factory=lambda: {
            "Healthy": "#0072B2",
            "Mech_Damage": "#D55E00",
            "Elec_Damage": "#009E73",
            "Mech_Elec_Damage": "#CC79A7",
            "reference": "#666666",
            "ink": "#1A1A1A",
        }
    )

    def rcparams(self) -> dict[str, Any]:
        """Return the matplotlib rcParams controlled by this contract."""
        return {
            "font.family": self.font_family,
            "font.size": self.base_font_size,
            "axes.titlesize": self.title_font_size,
            "axes.labelsize": self.axis_label_size,
            "xtick.labelsize": self.tick_label_size,
            "ytick.labelsize": self.tick_label_size,
            "legend.fontsize": self.legend_font_size,
            "axes.linewidth": self.line_width,
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }


def load_style(config: Mapping[str, Any] | None = None) -> FigureStyle:
    """Build a style from a validated mapping, retaining safe defaults."""
    if not config:
        return FigureStyle()
    allowed = {field for field in FigureStyle.__dataclass_fields__ if field != "colors"}
    values = {key: value for key, value in config.items() if key in allowed}
    if "colors" in config:
        values["colors"] = dict(config["colors"])
    return FigureStyle(**values)


@contextmanager
def publication_style(style: FigureStyle | None = None) -> Iterator[FigureStyle]:
    """Apply the common style while preserving the caller's matplotlib state."""
    selected = style or FigureStyle()
    with plt.rc_context(selected.rcparams()):  # type: ignore[arg-type]
        yield selected
