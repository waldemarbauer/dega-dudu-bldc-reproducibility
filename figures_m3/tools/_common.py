"""Shared utilities for the scientific-figures command-line tools."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import yaml


@dataclass(frozen=True)
class Issue:
    """A machine-readable validation or build issue.

    Parameters
    ----------
    severity:
        One of ``blocking``, ``warning``, or ``info``.
    code:
        Stable issue identifier suitable for tests and downstream tooling.
    location:
        Human-readable configuration or artifact location.
    message:
        Explanation of the problem.
    hint:
        Optional corrective guidance.
    """

    severity: str
    code: str
    location: str
    message: str
    hint: str | None = None

    def to_dict(self) -> dict[str, str]:
        result = {
            "severity": self.severity,
            "code": self.code,
            "location": self.location,
            "message": self.message,
        }
        if self.hint:
            result["hint"] = self.hint
        return result


def skill_root() -> Path:
    """Return the root directory of the ``scientific-figures`` skill."""

    return Path(__file__).resolve().parent.parent


def default_assets_dir() -> Path:
    """Return the default assets directory adjacent to the scripts."""

    return skill_root() / "assets"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML document and require a mapping at the document root."""

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"YAML file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping at document root: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON document and require a mapping at the document root."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object at document root: {path}")
    return data


def dump_yaml(data: Mapping[str, Any], path: Path) -> None:
    """Write YAML atomically using stable key order from the input mapping."""

    text = yaml.safe_dump(
        dict(data),
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    atomic_write_text(path, text)


def dump_json(data: Mapping[str, Any], path: Path) -> None:
    """Write formatted JSON atomically."""

    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to ``path`` without leaving a partially written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input.

    Lists and scalar values are replaced rather than concatenated. This mirrors
    the configuration precedence rule: a figure-specific value replaces a
    profile or project default at the same location.
    """

    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = value
    return result


def is_safe_relative_path(value: str) -> bool:
    """Return whether ``value`` is a portable project-relative path."""

    candidate = Path(value)
    if candidate.is_absolute():
        return False
    if len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}:
        return False
    return ".." not in candidate.parts


def project_path(project_root: Path, relative_path: str) -> Path:
    """Resolve a validated project-relative path below ``project_root``."""

    if not is_safe_relative_path(relative_path):
        raise ValueError(f"Unsafe or non-portable project path: {relative_path}")
    return project_root / Path(relative_path)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_files(root: Path) -> list[Path]:
    """Return all regular files below ``root`` in deterministic order."""

    return sorted(path for path in root.rglob("*") if path.is_file())


def write_checksums(root: Path, destination: Path) -> None:
    """Write SHA-256 checksums for all package files except the checksum file."""

    lines: list[str] = []
    destination_resolved = destination.resolve()
    for path in sorted_files(root):
        if path.resolve() == destination_resolved:
            continue
        lines.append(f"{file_sha256(path)}  {path.relative_to(root).as_posix()}")
    atomic_write_text(destination, "\n".join(lines) + "\n")


def copy_file(source: Path, destination: Path) -> None:
    """Copy a file while creating its destination directory."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def summarize_issues(issues: Iterable[Issue]) -> dict[str, int]:
    """Count issues by severity."""

    summary = {"blocking": 0, "warning": 0, "info": 0}
    for issue in issues:
        summary[issue.severity] = summary.get(issue.severity, 0) + 1
    return summary


def print_issues(issues: Iterable[Issue]) -> None:
    """Print issues in a compact human-readable format."""

    for issue in issues:
        print(f"[{issue.severity.upper()}] {issue.code} at {issue.location}")
        print(f"  {issue.message}")
        if issue.hint:
            print(f"  Hint: {issue.hint}")
