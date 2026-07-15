#!/usr/bin/env python3
"""Inspect saved PDF and PNG figure artifacts.

The inspector verifies measurable properties only. It does not claim to detect
all visual defects, semantic accessibility problems, or scientific errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

from PIL import Image
from pypdf import PdfReader

from _common import Issue, default_assets_dir, dump_json, dump_yaml, load_yaml, summarize_issues


POINTS_PER_INCH = 72.0


def _close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def inspect_png(path: Path) -> dict[str, Any]:
    """Inspect PNG dimensions, resolution metadata, colour mode, and alpha."""

    with Image.open(path) as image:
        width_px, height_px = image.size
        dpi_value = image.info.get("dpi")
        if isinstance(dpi_value, tuple):
            dpi_x, dpi_y = float(dpi_value[0]), float(dpi_value[1])
        else:
            dpi_x = dpi_y = None
        result: dict[str, Any] = {
            "format": "png",
            "path": str(path),
            "width_px": width_px,
            "height_px": height_px,
            "mode": image.mode,
            "has_alpha": image.mode in {"LA", "RGBA", "PA"} or "transparency" in image.info,
            "dpi_x": dpi_x,
            "dpi_y": dpi_y,
        }
        if dpi_x and dpi_y:
            result["width_in"] = width_px / dpi_x
            result["height_in"] = height_px / dpi_y
        return result


def _font_descriptor(font: Any) -> Any | None:
    try:
        descriptor = font.get("/FontDescriptor")
        if descriptor is not None:
            return descriptor.get_object()
        descendants = font.get("/DescendantFonts")
        if descendants:
            descendant = descendants[0].get_object()
            descriptor = descendant.get("/FontDescriptor")
            if descriptor is not None:
                return descriptor.get_object()
    except Exception:
        return None
    return None


def inspect_pdf(path: Path) -> dict[str, Any]:
    """Inspect PDF page dimensions, fonts, and embedded raster images."""

    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []
    fonts: dict[str, dict[str, Any]] = {}
    image_count = 0

    for page_index, page in enumerate(reader.pages):
        box = page.mediabox
        width_pt = float(box.width)
        height_pt = float(box.height)
        pages.append(
            {
                "page": page_index + 1,
                "width_pt": width_pt,
                "height_pt": height_pt,
                "width_in": width_pt / POINTS_PER_INCH,
                "height_in": height_pt / POINTS_PER_INCH,
            }
        )
        resources = page.get("/Resources")
        if resources is None:
            continue
        resources = resources.get_object()
        font_resources = resources.get("/Font", {})
        try:
            font_resources = font_resources.get_object()
        except AttributeError:
            pass
        for resource_name, font_reference in dict(font_resources).items():
            font = font_reference.get_object()
            base_font = str(font.get("/BaseFont", "unknown"))
            descriptor = _font_descriptor(font)
            embedded = False
            if descriptor is not None:
                embedded = any(
                    key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
                )
            fonts[f"{resource_name}:{base_font}"] = {
                "resource": str(resource_name),
                "base_font": base_font,
                "subtype": str(font.get("/Subtype", "unknown")),
                "embedded": embedded,
            }

        xobjects = resources.get("/XObject", {})
        try:
            xobjects = xobjects.get_object()
        except AttributeError:
            pass
        for reference in dict(xobjects).values():
            obj = reference.get_object()
            if obj.get("/Subtype") == "/Image":
                image_count += 1

    return {
        "format": "pdf",
        "path": str(path),
        "page_count": len(reader.pages),
        "pages": pages,
        "fonts": list(fonts.values()),
        "all_fonts_embedded": all(item["embedded"] for item in fonts.values()) if fonts else None,
        "embedded_raster_image_count": image_count,
    }


def inspect_artifact(path: Path) -> dict[str, Any]:
    """Inspect a supported artifact by file extension."""

    suffix = path.suffix.lower()
    if suffix == ".png":
        return inspect_png(path)
    if suffix == ".pdf":
        return inspect_pdf(path)
    raise ValueError(f"Unsupported artifact format: {suffix}")


def validate_against_profile(
    inspection: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    dimension_tolerance_in: float = 0.02,
    dpi_tolerance: float = 1.5,
) -> list[Issue]:
    """Compare measurable artifact properties with an export profile."""

    issues: list[Issue] = []
    dimensions = profile.get("dimensions", {})
    expected_width = dimensions.get("width_in") if isinstance(dimensions, Mapping) else None
    expected_height = dimensions.get("height_in") if isinstance(dimensions, Mapping) else None
    artifact_format = inspection.get("format")

    if artifact_format == "png":
        dpi = profile.get("png", {}).get("dpi")
        actual_dpi_x = inspection.get("dpi_x")
        actual_dpi_y = inspection.get("dpi_y")
        if dpi is not None:
            if actual_dpi_x is None or actual_dpi_y is None:
                issues.append(
                    Issue(
                        "blocking",
                        "png-dpi-missing",
                        str(inspection.get("path")),
                        "PNG does not contain usable DPI metadata.",
                    )
                )
            elif not (_close(actual_dpi_x, dpi, dpi_tolerance) and _close(actual_dpi_y, dpi, dpi_tolerance)):
                issues.append(
                    Issue(
                        "blocking",
                        "png-dpi-mismatch",
                        str(inspection.get("path")),
                        f"Expected {dpi} DPI, measured {actual_dpi_x:.2f} × {actual_dpi_y:.2f} DPI.",
                    )
                )
        if expected_width and expected_height and dpi:
            expected_px = (round(expected_width * dpi), round(expected_height * dpi))
            actual_px = (inspection.get("width_px"), inspection.get("height_px"))
            if actual_px != expected_px:
                issues.append(
                    Issue(
                        "blocking",
                        "png-pixel-size-mismatch",
                        str(inspection.get("path")),
                        f"Expected {expected_px[0]} × {expected_px[1]} px, measured "
                        f"{actual_px[0]} × {actual_px[1]} px.",
                    )
                )

    if artifact_format == "pdf":
        pages = inspection.get("pages", [])
        if len(pages) != 1:
            issues.append(
                Issue(
                    "blocking",
                    "pdf-page-count",
                    str(inspection.get("path")),
                    f"A figure PDF should contain one page; measured {len(pages)}.",
                )
            )
        elif expected_width and expected_height:
            page = pages[0]
            if not (
                _close(float(page["width_in"]), float(expected_width), dimension_tolerance_in)
                and _close(float(page["height_in"]), float(expected_height), dimension_tolerance_in)
            ):
                issues.append(
                    Issue(
                        "blocking",
                        "pdf-page-size-mismatch",
                        str(inspection.get("path")),
                        f"Expected {expected_width} × {expected_height} in, measured "
                        f"{page['width_in']:.3f} × {page['height_in']:.3f} in.",
                    )
                )
        if inspection.get("all_fonts_embedded") is False:
            issues.append(
                Issue(
                    "blocking",
                    "pdf-font-not-embedded",
                    str(inspection.get("path")),
                    "At least one PDF font resource is not embedded.",
                )
            )
        if inspection.get("embedded_raster_image_count", 0) > 0:
            issues.append(
                Issue(
                    "warning",
                    "pdf-raster-content",
                    str(inspection.get("path")),
                    f"PDF contains {inspection['embedded_raster_image_count']} raster image object(s).",
                    "Verify that rasterization is intentional and sufficient at final size.",
                )
            )
    return issues


def inspect_with_profile(
    artifact: Path,
    profile_name: str | None,
    *,
    assets_dir: Path | None = None,
) -> dict[str, Any]:
    """Inspect an artifact and optionally validate it against an export profile."""

    inspection = inspect_artifact(artifact)
    issues: list[Issue] = []
    profile: dict[str, Any] | None = None
    if profile_name:
        assets = assets_dir or default_assets_dir()
        profiles = load_yaml(assets / "export-profiles.yaml").get("profiles", {})
        if profile_name not in profiles:
            raise KeyError(f"Unknown export profile: {profile_name}")
        profile = profiles[profile_name]
        issues = validate_against_profile(inspection, profile)
    return {
        "artifact": inspection,
        "export_profile": profile_name,
        "valid": not any(item.severity == "blocking" for item in issues),
        "summary": summarize_issues(issues),
        "issues": [item.to_dict() for item in issues],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--export-profile", default=None)
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = inspect_with_profile(
            args.artifact,
            args.export_profile,
            assets_dir=args.assets_dir,
        )
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f"Artifact inspection failed: {exc}", file=sys.stderr)
        return 2

    if args.output:
        if args.format == "json":
            dump_json(report, args.output)
        else:
            dump_yaml(report, args.output)
    else:
        import yaml

        print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    return 1 if report["summary"]["blocking"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
