#!/usr/bin/env python3
"""Validate a project scientific-figure configuration.

The tool performs JSON Schema validation and project-specific semantic checks
that are intentionally outside the schema: palette and export-profile
resolution, safe project-relative paths, profile compatibility, approval gates,
and lifecycle consistency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from _common import (
    Issue,
    default_assets_dir,
    dump_json,
    dump_yaml,
    is_safe_relative_path,
    load_json,
    load_yaml,
    print_issues,
    project_path,
    summarize_issues,
)
from figure_contract import (
    ELIGIBLE_CAPTION_STATUSES,
    IMPLEMENTED_STATUSES,
    RENDERED_STATUSES,
)


def _json_path(parts: Iterable[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _walk_paths(value: Any, location: str = "$") -> Iterable[tuple[str, str]]:
    """Yield likely path fields from a configuration tree."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{location}.{key}"
            if isinstance(item, str) and key in {
                "artifact",
                "file",
                "pdf",
                "png",
                "caption_markdown",
                "caption_latex",
                "validation",
                "baseline_artifact",
                "draft_stem",
                "final_stem",
                "destination",
            }:
                yield child, item
            yield from _walk_paths(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_paths(item, f"{location}[{index}]")


def _nonlinear_or_restricted_axes(specification: Mapping[str, Any]) -> bool:
    axes = specification.get("layout", {}).get("axes", {})
    if not isinstance(axes, Mapping):
        return False
    for axis in axes.values():
        if not isinstance(axis, Mapping):
            continue
        if axis.get("scale", "linear") != "linear":
            return True
        displayed = axis.get("displayed_range")
        if displayed not in {None, "full"}:
            return True
        if "limits" in axis:
            return True
    return False


def _has_interpretive_annotations(specification: Mapping[str, Any]) -> bool:
    annotations = specification.get("visual", {}).get("annotations", [])
    return any(
        isinstance(item, Mapping) and item.get("type") == "interpretive"
        for item in annotations
    )


def _artifact_paths_for_status(figure: Mapping[str, Any]) -> list[str]:
    artifacts = figure.get("implementation", {}).get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return []
    return [value for value in artifacts.values() if isinstance(value, str)]


def validate_configuration(
    config_path: Path,
    *,
    assets_dir: Path | None = None,
    project_root: Path | None = None,
    check_project_state: bool = False,
) -> tuple[dict[str, Any], list[Issue]]:
    """Validate a figure configuration and return the loaded document.

    Parameters
    ----------
    config_path:
        Path to ``config/figures.yaml``.
    assets_dir:
        Directory containing the schema, palette, export profiles, and styles.
        Defaults to the assets adjacent to these scripts.
    project_root:
        Root used for project-state checks. Defaults to the parent of the
        configuration directory when the configuration is under ``config/``.
    check_project_state:
        Check referenced files in addition to static configuration rules.
    """

    assets = assets_dir or default_assets_dir()
    config = load_yaml(config_path)
    schema = load_json(assets / "figures.schema.json")
    validator = Draft202012Validator(schema)
    issues: list[Issue] = []

    for error in sorted(validator.iter_errors(config), key=lambda item: list(item.path)):
        issues.append(
            Issue(
                severity="blocking",
                code="schema-validation",
                location=_json_path(error.absolute_path),
                message=error.message,
            )
        )

    try:
        palettes = load_yaml(assets / "palette.yaml").get("palettes", {})
        export_profiles = load_yaml(assets / "export-profiles.yaml").get("profiles", {})
    except (FileNotFoundError, ValueError, TypeError) as exc:
        issues.append(
            Issue(
                "blocking",
                "asset-loading",
                str(assets),
                str(exc),
            )
        )
        return config, issues

    for location, value in _walk_paths(config):
        if not is_safe_relative_path(value):
            issues.append(
                Issue(
                    "blocking",
                    "unsafe-project-path",
                    location,
                    f"Path must be portable and project-relative: {value}",
                )
            )

    defaults = config.get("defaults", {})
    for key in ("palette", "neutral_palette"):
        palette_name = defaults.get(key)
        if palette_name and palette_name not in palettes:
            issues.append(
                Issue(
                    "blocking",
                    "unknown-palette",
                    f"$.defaults.{key}",
                    f"Palette '{palette_name}' is not defined in palette.yaml.",
                )
            )

    figures = config.get("figures", {})
    if not isinstance(figures, Mapping):
        figures = {}

    for figure_id, figure in figures.items():
        if not isinstance(figure, Mapping):
            continue
        base = f"$.figures.{figure_id}"
        specification = figure.get("specification", {})
        implementation = figure.get("implementation", {})
        if not isinstance(specification, Mapping) or not isinstance(implementation, Mapping):
            continue
        profile = specification.get("profile")
        status = implementation.get("status")
        approvals = specification.get("approvals", {})
        outputs = specification.get("outputs", {})
        visual = specification.get("visual", {})
        accessibility = specification.get("accessibility", {})
        captions = specification.get("captions", {})

        palette_name = visual.get("palette") if isinstance(visual, Mapping) else None
        if palette_name and palette_name not in palettes:
            issues.append(
                Issue(
                    "blocking",
                    "unknown-palette",
                    f"{base}.specification.visual.palette",
                    f"Palette '{palette_name}' is not defined in palette.yaml.",
                )
            )

        export_profile_name = outputs.get("export_profile") if isinstance(outputs, Mapping) else None
        export_profile = export_profiles.get(export_profile_name) if export_profile_name else None
        if export_profile_name and export_profile is None:
            issues.append(
                Issue(
                    "blocking",
                    "unknown-export-profile",
                    f"{base}.specification.outputs.export_profile",
                    f"Export profile '{export_profile_name}' is not defined.",
                )
            )
        elif isinstance(export_profile, Mapping) and export_profile.get("profile") != profile:
            issues.append(
                Issue(
                    "blocking",
                    "export-profile-mismatch",
                    f"{base}.specification.outputs.export_profile",
                    f"Export profile '{export_profile_name}' targets "
                    f"'{export_profile.get('profile')}', not '{profile}'.",
                )
            )

        layout = specification.get("layout", {})
        if isinstance(layout, Mapping) and layout.get("class") == "custom":
            if approvals.get("custom_dimensions") != "user-approved":
                issues.append(
                    Issue(
                        "blocking",
                        "custom-dimensions-not-approved",
                        f"{base}.specification.approvals.custom_dimensions",
                        "Custom figure dimensions require explicit user approval.",
                    )
                )

        if _nonlinear_or_restricted_axes(specification):
            if approvals.get("interpretation_sensitive_scales") != "user-approved":
                issues.append(
                    Issue(
                        "blocking",
                        "interpretation-sensitive-scale-not-approved",
                        f"{base}.specification.approvals.interpretation_sensitive_scales",
                        "Non-linear scales, limits, or restricted ranges require user approval.",
                    )
                )

        if _has_interpretive_annotations(specification):
            if approvals.get("interpretive_annotations") != "user-approved":
                issues.append(
                    Issue(
                        "blocking",
                        "interpretive-annotation-not-approved",
                        f"{base}.specification.approvals.interpretive_annotations",
                        "Interpretive annotations require explicit user approval.",
                    )
                )

        exception = accessibility.get("exception", {}) if isinstance(accessibility, Mapping) else {}
        if isinstance(exception, Mapping) and exception.get("status") == "user-approved":
            if approvals.get("accessibility_exception") != "user-approved":
                issues.append(
                    Issue(
                        "blocking",
                        "accessibility-exception-inconsistent",
                        f"{base}.specification.approvals.accessibility_exception",
                        "The accessibility exception and approval fields are inconsistent.",
                    )
                )

        if status in IMPLEMENTED_STATUSES:
            source = implementation.get("source", {})
            if not isinstance(source, Mapping) or not source.get("file") or not source.get("builder"):
                issues.append(
                    Issue(
                        "blocking",
                        "implemented-source-incomplete",
                        f"{base}.implementation.source",
                        "Implemented figures require a source file and builder function.",
                    )
                )

        if status in RENDERED_STATUSES and not _artifact_paths_for_status(figure):
            issues.append(
                Issue(
                    "blocking",
                    "rendered-artifacts-missing",
                    f"{base}.implementation.artifacts",
                    "A rendered lifecycle status requires artifact paths.",
                )
            )

        if status == "final":
            content = specification.get("content", {})
            source = specification.get("analytical_source", {})
            intent = specification.get("intent", {})
            for section_name, section in (
                ("intent", intent),
                ("analytical_source", source),
                ("content", content),
            ):
                if not isinstance(section, Mapping) or section.get("status") != "resolved":
                    issues.append(
                        Issue(
                            "blocking",
                            "unresolved-final-specification",
                            f"{base}.specification.{section_name}",
                            f"Final figures require resolved {section_name.replace('_', ' ')}.",
                        )
                    )
            if approvals.get("final_render") != "user-approved":
                issues.append(
                    Issue(
                        "blocking",
                        "final-render-not-approved",
                        f"{base}.specification.approvals.final_render",
                        "Final status requires explicit user approval of the target-size render.",
                    )
                )
            if approvals.get("content_selection") != "user-approved":
                issues.append(
                    Issue(
                        "blocking",
                        "content-selection-not-approved",
                        f"{base}.specification.approvals.content_selection",
                        "Final status requires user-approved content selection.",
                    )
                )
            if profile == "publication":
                if not isinstance(captions, Mapping) or not captions.get("required"):
                    issues.append(
                        Issue(
                            "blocking",
                            "publication-caption-not-required",
                            f"{base}.specification.captions",
                            "Publication figures must require a caption.",
                        )
                    )
                elif captions.get("status") not in ELIGIBLE_CAPTION_STATUSES:
                    issues.append(
                        Issue(
                            "blocking",
                            "publication-caption-not-approved",
                            f"{base}.specification.captions.status",
                            "Final publication captions must be user-reviewed or final.",
                        )
                    )

        if profile == "publication" and isinstance(visual, Mapping):
            if visual.get("legend", {}).get("strategy") not in {None, "none"}:
                issues.append(
                    Issue(
                        "warning",
                        "publication-legend",
                        f"{base}.specification.visual.legend",
                        "A publication legend is allowed but should be justified against direct labels.",
                    )
                )
            if visual.get("title", {}).get("strategy") not in {None, "none"}:
                issues.append(
                    Issue(
                        "warning",
                        "publication-title",
                        f"{base}.specification.visual.title",
                        "Publication figures normally omit an internal full title.",
                    )
                )

    if check_project_state:
        root = project_root or config_path.parent.parent
        for location, value in _walk_paths(config):
            if not is_safe_relative_path(value):
                continue
            if location.endswith((".artifact", ".file", ".pdf", ".png", ".caption_markdown", ".caption_latex", ".validation", ".baseline_artifact")):
                path = project_path(root, value)
                if not path.exists():
                    issues.append(
                        Issue(
                            "blocking",
                            "referenced-path-missing",
                            location,
                            f"Referenced project path does not exist: {value}",
                        )
                    )

    return config, issues


def build_report(config_path: Path, issues: list[Issue]) -> dict[str, Any]:
    summary = summarize_issues(issues)
    return {
        "configuration": str(config_path),
        "valid": summary.get("blocking", 0) == 0,
        "summary": summary,
        "issues": [issue.to_dict() for issue in issues],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to config/figures.yaml")
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--check-project-state", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        _, issues = validate_configuration(
            args.config,
            assets_dir=args.assets_dir,
            project_root=args.project_root,
            check_project_state=args.check_project_state,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Configuration validation failed: {exc}", file=sys.stderr)
        return 2

    report = build_report(args.config, issues)
    if args.output:
        if args.format == "json":
            dump_json(report, args.output)
        else:
            dump_yaml(report, args.output)
    else:
        print_issues(issues)
        print(json.dumps(report["summary"], indent=2))

    return 1 if report["summary"]["blocking"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
