"""Build every DEGA M3 computational figure from frozen ArticleV1 artifacts.

Single entry point for the M3 figure pipeline:

    python figures_m3/scripts/make_all_figures.py

Writes vector PDF, 300-dpi PNG and SVG to figures_m3/output/, the exact plotted
values to figures_m3/output/figure_data/, and a provenance manifest to
figures_m3/data_manifest/figure_manifest.yaml.

This script never writes to Output/, so frozen scientific outputs and the
original figures are left untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

from _m3_common import export, load_config, write_manifest  # noqa: E402
from fig_agent import build_fig_g, build_fig_h, build_fig_i  # noqa: E402
from fig_classification import build_fig_a, build_fig_b  # noqa: E402
from fig_evidence import build_fig_c, build_fig_d  # noqa: E402
from fig_xai import build_fig_e, build_fig_f  # noqa: E402

BUILDERS = {
    "FIG_A": build_fig_a,
    "FIG_B": build_fig_b,
    "FIG_C": build_fig_c,
    "FIG_D": build_fig_d,
    "FIG_E": build_fig_e,
    "FIG_F": build_fig_f,
    "FIG_G": build_fig_g,
    "FIG_H": build_fig_h,
    "FIG_I": build_fig_i,
}


def main() -> int:
    cfg = load_config()
    records = []
    failures = []

    for fid, builder in BUILDERS.items():
        spec = cfg["figures"][fid]
        try:
            fig, record = builder(cfg)
            export(fig, fid, spec, record)
            records.append(record)
            import matplotlib.pyplot as plt

            plt.close(fig)
            print(f"[ok]   {fid}  {record.width_in}x{record.height_in} in  -> {record.outputs['pdf']}")
        except Exception as exc:  # noqa: BLE001
            failures.append((fid, exc))
            print(f"[FAIL] {fid}: {type(exc).__name__}: {exc}")

    if records:
        write_manifest(records, cfg)
        print(f"\nmanifest: figures_m3/data_manifest/figure_manifest.yaml ({len(records)} figures)")

    if failures:
        print(f"\n{len(failures)} figure(s) failed:")
        for fid, exc in failures:
            print(f"  - {fid}: {exc}")
        return 1

    print(f"\nAll {len(records)} M3 figures built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
