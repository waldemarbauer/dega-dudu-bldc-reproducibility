"""Technical validation of the exported M3 figures.

Checks, per figure:
  * PDF page size matches the configured physical dimensions;
  * PDF carries vector drawing operators and embedded (Type-42) fonts;
  * PNG pixel dimensions correspond to the configured size at 300 dpi;
  * all three formats (pdf, png, svg) exist;
  * a grayscale conversion of the PNG retains usable tonal separation.

Run:  python figures_m3/scripts/validate_m3_figures.py
"""

from __future__ import annotations

import re
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _m3_common import ROOT, load_config  # noqa: E402

TOL_IN = 0.02
PT_PER_IN = 72.0


def pdf_geometry(path: Path) -> tuple[float, float, bool, set[str]]:
    raw = path.read_bytes()
    box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
    x0, y0, x1, y1 = (float(g) for g in box.groups())
    w_in, h_in = (x1 - x0) / PT_PER_IN, (y1 - y0) / PT_PER_IN

    # Decompress content streams and look for vector path/text operators.
    vector = False
    for m in re.finditer(rb"stream\r?\n", raw):
        s = m.end()
        e = raw.find(b"endstream", s)
        try:
            data = zlib.decompress(raw[s:e])
        except Exception:
            continue
        if re.search(rb"\b(re|m|c|l)\b", data) and re.search(rb"\bBT\b", data):
            vector = True
            break

    fonts = {f.decode() for f in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", raw)}
    embedded = bool(re.search(rb"/FontFile[23]?\b", raw))
    return w_in, h_in, (vector and embedded), fonts


def png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    w, h = struct.unpack(">II", raw[16:24])
    return w, h


def grayscale_spread(path: Path) -> float:
    """Luminance spread of the rendered PNG, as a coarse grayscale-legibility cue."""
    try:
        import matplotlib.image as mpimg
        import numpy as np
    except ImportError:
        return float("nan")
    a = mpimg.imread(path)[..., :3]
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    ink = lum[lum < 0.97]  # ignore the white page
    return float(ink.max() - ink.min()) if ink.size else 0.0


def main() -> int:
    cfg = load_config()
    out = ROOT / "figures_m3/output"
    failures: list[str] = []
    print(f"{'fig':6s} {'PDF in':>13s} {'exp in':>13s} {'vec+font':>9s} {'PNG px':>12s} "
          f"{'dpi':>5s} {'svg':>4s} {'gray':>5s}")
    print("-" * 78)

    for fid, spec in cfg["figures"].items():
        stem = f"Figure_{fid.split('_')[-1]}"
        pdf, png, svg = out / "pdf" / f"{stem}.pdf", out / "png" / f"{stem}.png", out / "svg" / f"{stem}.svg"
        ew, eh = spec["dimensions"]["width_in"], spec["dimensions"]["height_in"]

        for p in (pdf, png, svg):
            if not p.exists():
                failures.append(f"{fid}: missing {p.name}")

        w, h, vecfont, _fonts = pdf_geometry(pdf)
        pw, ph = png_size(png)
        dpi = round(pw / ew)
        gray = grayscale_spread(png)

        if abs(w - ew) > TOL_IN or abs(h - eh) > TOL_IN:
            failures.append(f"{fid}: PDF {w:.2f}x{h:.2f} != configured {ew}x{eh}")
        if not vecfont:
            failures.append(f"{fid}: PDF not vector-with-embedded-fonts")
        if dpi != 300:
            failures.append(f"{fid}: PNG dpi {dpi} != 300")
        if gray < 0.5:
            failures.append(f"{fid}: low grayscale spread {gray:.2f}")

        print(f"{fid[-1]:6s} {w:6.2f}x{h:<6.2f} {ew:6.2f}x{eh:<6.2f} {str(vecfont):>9s} "
              f"{pw:5d}x{ph:<6d} {dpi:5d} {'yes' if svg.exists() else 'NO':>4s} {gray:5.2f}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 1
    print("All figures pass geometry, vector/font, DPI, format and grayscale checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
