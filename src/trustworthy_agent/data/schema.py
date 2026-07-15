"""Dataset schema discovery for DUDU-BLDC CSV files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trustworthy_agent.data.manifest import (
    CANONICAL_CLASSES,
    CANONICAL_CSV_FILES,
    CANONICAL_FILE_CLASS_MAP,
)


@dataclass(frozen=True)
class ColumnSchema:
    """Discovered metadata for one CSV column."""

    name: str
    dtype: str
    missing_count: int
    infinity_count: int
    observed_non_missing: int

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable column metadata."""

        return {
            "name": self.name,
            "dtype": self.dtype,
            "missing_count": self.missing_count,
            "infinity_count": self.infinity_count,
            "observed_non_missing": self.observed_non_missing,
        }


@dataclass(frozen=True)
class FileSchema:
    """Discovered metadata for one CSV file."""

    path: str
    rows: int
    columns: tuple[ColumnSchema, ...]
    duplicated_columns: tuple[str, ...]
    duplicate_rows: int
    class_labels: tuple[str, ...]
    class_distribution: dict[str, int]
    source_class: str | None
    candidate_feature_groups: dict[str, tuple[str, ...]]
    available_identifiers: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable file metadata."""

        return {
            "path": self.path,
            "rows": self.rows,
            "columns": [column.to_json_dict() for column in self.columns],
            "column_count": len(self.columns),
            "duplicated_columns": list(self.duplicated_columns),
            "duplicate_rows": self.duplicate_rows,
            "class_labels": list(self.class_labels),
            "class_distribution": self.class_distribution,
            "source_class": self.source_class,
            "candidate_feature_groups": {
                name: list(columns) for name, columns in self.candidate_feature_groups.items()
            },
            "available_identifiers": list(self.available_identifiers),
        }


@dataclass(frozen=True)
class DatasetSchema:
    """Discovered DUDU-BLDC dataset schema metadata.

    Purpose:
        Store schema facts discovered from actual CSV file headers and values.
    Parameters:
        root: Project-relative or caller-supplied schema root.
        files: Discovered CSV file schemas.
        expected_files_present: Canonical analytical files found by basename.
        expected_files_missing: Canonical analytical files absent from the scan.
        unexpected_csv_files: CSV files outside the canonical five analytical names.
    Return value:
        Immutable schema metadata.
    Raised exceptions:
        None.
    Scientific assumptions:
        Empty schema means undiscovered, not valid.
    Side effects:
        None.
    Reproducibility implications:
        Schema discovery can be persisted for audit.
    """

    root: str
    files: tuple[FileSchema, ...]
    expected_files_present: tuple[str, ...]
    expected_files_missing: tuple[str, ...]
    unexpected_csv_files: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dataset schema metadata."""

        return {
            "root": self.root,
            "files": [file_schema.to_json_dict() for file_schema in self.files],
            "expected_files_present": list(self.expected_files_present),
            "expected_files_missing": list(self.expected_files_missing),
            "unexpected_csv_files": list(self.unexpected_csv_files),
        }


def discover_schema(root: Path) -> DatasetSchema:
    """Discover DUDU-BLDC CSV schema facts from actual files.

    Purpose:
        Inspect headers and values without assuming a fixed feature count.
    Parameters:
        root: Directory containing extracted or raw CSV files.
    Return value:
        DatasetSchema with all discovered columns represented.
    Raised exceptions:
        FileNotFoundError if the root does not exist.
        ValueError for unreadable/empty CSV structure.
    Scientific assumptions:
        Source-file class mapping is used only for the four canonical analytical
        split files documented by the protocol.
    Side effects:
        None.
    Reproducibility implications:
        Schema evidence is derived from local file contents and can be persisted.
    """

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    csv_paths = sorted(path for path in root.rglob("*.csv") if path.is_file())
    file_schemas = tuple(_discover_file_schema(path, root) for path in csv_paths)
    basenames = {path.name for path in csv_paths}
    expected_present = tuple(name for name in CANONICAL_CSV_FILES if name in basenames)
    expected_missing = tuple(name for name in CANONICAL_CSV_FILES if name not in basenames)
    unexpected = tuple(sorted(name for name in basenames if name not in set(CANONICAL_CSV_FILES)))
    return DatasetSchema(
        root=str(root),
        files=file_schemas,
        expected_files_present=expected_present,
        expected_files_missing=expected_missing,
        unexpected_csv_files=unexpected,
    )


def write_schema_reports(schema: DatasetSchema, output_root: Path) -> None:
    """Write schema, missingness, duplicates, and class-distribution reports.

    Purpose:
        Generate deterministic audit artifacts under `Output/`.
    Parameters:
        schema: Discovered dataset schema.
        output_root: Repository `Output` directory.
    Return value:
        None.
    Raised exceptions:
        OSError for report write failures.
    Scientific assumptions:
        Reports describe observed source structure only.
    Side effects:
        Writes JSON, Markdown, and CSV report files.
    Reproducibility implications:
        Recreates schema-audit outputs from local source files.
    """

    manifest_root = output_root / "Manifests"
    results_root = output_root / "Results"
    manifest_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "dataset_schema.json").write_text(
        json.dumps(schema.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (manifest_root / "dataset_schema.md").write_text(_schema_markdown(schema), encoding="utf-8")
    _write_class_distribution(schema, results_root / "dataset_class_distribution.csv")
    _write_missingness(schema, results_root / "dataset_missingness.csv")
    _write_duplicates(schema, results_root / "dataset_duplicate_report.csv")


def _discover_file_schema(path: Path, root: Path) -> FileSchema:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file has no header: {path}") from exc
        if not header:
            raise ValueError(f"CSV file has an empty header: {path}")
        duplicate_columns = _duplicates(header)
        column_values: dict[str, list[str]] = {column: [] for column in header}
        row_hashes: dict[tuple[str, ...], int] = {}
        row_count = 0
        for row in reader:
            row_count += 1
            normalized_row = tuple(row)
            row_hashes[normalized_row] = row_hashes.get(normalized_row, 0) + 1
            for index, column in enumerate(header):
                column_values[column].append(row[index] if index < len(row) else "")
    columns = tuple(
        ColumnSchema(
            name=column,
            dtype=_infer_dtype(values),
            missing_count=sum(1 for value in values if _is_missing(value)),
            infinity_count=sum(1 for value in values if _is_infinity(value)),
            observed_non_missing=sum(1 for value in values if not _is_missing(value)),
        )
        for column, values in column_values.items()
    )
    class_distribution = _class_distribution(path, column_values, row_count)
    return FileSchema(
        path=str(path.relative_to(root)),
        rows=row_count,
        columns=columns,
        duplicated_columns=duplicate_columns,
        duplicate_rows=sum(count - 1 for count in row_hashes.values() if count > 1),
        class_labels=tuple(sorted(class_distribution)),
        class_distribution=class_distribution,
        source_class=CANONICAL_FILE_CLASS_MAP.get(path.name),
        candidate_feature_groups=_candidate_feature_groups(tuple(column_values)),
        available_identifiers=_available_identifiers(tuple(column_values)),
    )


def _duplicates(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _infer_dtype(values: list[str]) -> str:
    non_missing = [value.strip() for value in values if not _is_missing(value)]
    if not non_missing:
        return "empty"
    if all(_is_integer(value) for value in non_missing):
        return "integer"
    if all(_is_float(value) for value in non_missing):
        return "float"
    if all(value.lower() in {"true", "false"} for value in non_missing):
        return "boolean"
    return "string"


def _is_missing(value: str) -> bool:
    return value.strip() == "" or value.strip().lower() in {"na", "nan", "null", "none"}


def _is_infinity(value: str) -> bool:
    return value.strip().lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _class_distribution(
    path: Path, column_values: dict[str, list[str]], row_count: int
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column, values in column_values.items():
        if column.lower() in {"class", "label", "target", "diagnosis", "fault_type", "condition"}:
            for value in values:
                if not _is_missing(value):
                    label = value.strip()
                    counts[label] = counts.get(label, 0) + 1
    if counts:
        return counts
    source_class = CANONICAL_FILE_CLASS_MAP.get(path.name)
    if source_class:
        return {source_class: row_count}
    return {}


def _candidate_feature_groups(columns: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {
        "current_time_domain": [],
        "current_frequency_domain": [],
        "current_harmonics": [],
        "speed_time_domain": [],
        "speed_frequency_domain": [],
        "speed_harmonics": [],
        "unclassified_candidate_features": [],
    }
    for column in columns:
        lower = column.lower()
        if lower in {"class", "label", "target", "diagnosis", "fault_type", "condition"}:
            continue
        if any(token in lower for token in ("id", "index", "motor", "scenario", "case", "window")):
            continue
        if "current" in lower or lower.startswith("i_") or lower.startswith("i"):
            if any(token in lower for token in ("freq", "fft", "hz", "spectrum")):
                groups["current_frequency_domain"].append(column)
            elif "harm" in lower:
                groups["current_harmonics"].append(column)
            else:
                groups["current_time_domain"].append(column)
        elif "speed" in lower or "rpm" in lower:
            if any(token in lower for token in ("freq", "fft", "hz", "spectrum")):
                groups["speed_frequency_domain"].append(column)
            elif "harm" in lower:
                groups["speed_harmonics"].append(column)
            else:
                groups["speed_time_domain"].append(column)
        else:
            groups["unclassified_candidate_features"].append(column)
    return {key: tuple(value) for key, value in groups.items()}


def _available_identifiers(columns: tuple[str, ...]) -> tuple[str, ...]:
    identifier_tokens = ("id", "index", "motor", "scenario", "case", "window", "group", "file")
    return tuple(
        column for column in columns if any(token in column.lower() for token in identifier_tokens)
    )


def _schema_markdown(schema: DatasetSchema) -> str:
    lines = [
        "# DUDU-BLDC Schema Discovery",
        "",
        f"- Root: `{schema.root}`",
        f"- Files discovered: {len(schema.files)}",
        f"- Expected files present: {', '.join(schema.expected_files_present) or 'none'}",
        f"- Expected files missing: {', '.join(schema.expected_files_missing) or 'none'}",
        f"- Unexpected CSV files: {', '.join(schema.unexpected_csv_files) or 'none'}",
        "",
    ]
    for file_schema in schema.files:
        lines.extend(
            [
                f"## {file_schema.path}",
                "",
                f"- Rows: {file_schema.rows}",
                f"- Columns: {len(file_schema.columns)}",
                f"- Duplicate rows: {file_schema.duplicate_rows}",
                f"- Class labels: {', '.join(file_schema.class_labels) or 'none'}",
                "- Available identifiers: "
                f"{', '.join(file_schema.available_identifiers) or 'none'}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_class_distribution(schema: DatasetSchema, path: Path) -> None:
    counts = {label: 0 for label in CANONICAL_CLASSES}
    unexpected: dict[str, int] = {}
    for file_schema in schema.files:
        for label, row_count in file_schema.class_distribution.items():
            if label in counts:
                counts[label] += row_count
            else:
                unexpected[label] = unexpected.get(label, 0) + row_count
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_label", "row_count", "status"])
        for label, count in counts.items():
            writer.writerow([label, count, "canonical"])
        for label, count in sorted(unexpected.items()):
            writer.writerow([label, count, "unexpected"])


def _write_missingness(schema: DatasetSchema, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "column", "dtype", "missing_count", "infinity_count"])
        for file_schema in schema.files:
            for column in file_schema.columns:
                writer.writerow(
                    [
                        file_schema.path,
                        column.name,
                        column.dtype,
                        column.missing_count,
                        column.infinity_count,
                    ]
                )


def _write_duplicates(schema: DatasetSchema, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "duplicate_rows", "duplicated_columns"])
        for file_schema in schema.files:
            writer.writerow(
                [
                    file_schema.path,
                    file_schema.duplicate_rows,
                    ";".join(file_schema.duplicated_columns),
                ]
            )
