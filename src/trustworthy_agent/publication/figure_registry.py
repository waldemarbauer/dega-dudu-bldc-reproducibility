"""Registry and validation for reproducible publication figure packages."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trustworthy_agent.provenance.hashing import sha256_file


@dataclass
class FigureRecord:
    """Provenance record for one rendered publication figure."""

    figure_id: str
    title: str
    caption_path: str
    data_path: str
    metadata_path: str
    output_paths: list[str]
    source_artifacts: dict[str, str]
    configuration: dict[str, Any] = field(default_factory=dict)
    configuration_hash: str = ""
    generator_version: str = "ARTICLEV1_FIGURE_GENERATOR_V1"
    scientific_interpretation: str = ""
    limitations: str = ""


class FigureRegistry:
    """Collect, validate, and persist figure records and inventory rows."""

    def __init__(
        self, project_root: Path, *, version: str = "ARTICLEV1_FIGURE_REGISTRY_V1"
    ) -> None:
        self.project_root = project_root
        self.version = version
        self._records: dict[str, FigureRecord] = {}

    @property
    def records(self) -> tuple[FigureRecord, ...]:
        """Return records in deterministic figure-ID order."""
        return tuple(self._records[key] for key in sorted(self._records))

    def register(self, record: FigureRecord) -> None:
        """Register a figure, rejecting duplicate IDs."""
        if record.figure_id in self._records:
            raise ValueError(f"duplicate figure ID: {record.figure_id}")
        self._records[record.figure_id] = record

    def validate(self) -> list[str]:
        """Return human-readable validation errors for registered artifacts."""
        errors: list[str] = []
        for record in self.records:
            paths = [
                record.data_path,
                record.caption_path,
                record.metadata_path,
                *record.output_paths,
            ]
            for relative in paths:
                if not (self.project_root / relative).exists():
                    errors.append(f"{record.figure_id}: missing {relative}")
            for relative, digest in record.source_artifacts.items():
                path = self.project_root / relative
                if not path.exists():
                    errors.append(f"{record.figure_id}: missing source {relative}")
                elif digest and sha256_file(path) != digest:
                    errors.append(f"{record.figure_id}: source hash mismatch {relative}")
        return errors

    def write_manifest(self, path: Path) -> Path:
        """Write a deterministic JSON registry manifest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.version,
            "figures": [asdict(item) for item in self.records],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_inventory(self, path: Path) -> Path:
        """Write a compact CSV inventory suitable for manuscript tracking."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["figure_id", "title", "data_path", "output_count", "source_count"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in self.records:
                writer.writerow(
                    {
                        "figure_id": item.figure_id,
                        "title": item.title,
                        "data_path": item.data_path,
                        "output_count": len(item.output_paths),
                        "source_count": len(item.source_artifacts),
                    }
                )
        return path


def write_registry(entries: Iterable[Mapping[str, Any]], path: Path) -> Path:
    """Persist lightweight generator metadata dictionaries in registry format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figures = sorted(
        (dict(entry) for entry in entries), key=lambda item: str(item.get("figure_id", ""))
    )
    payload = {"schema_version": "ARTICLEV1_FIGURE_REGISTRY_V1", "figures": figures}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path
