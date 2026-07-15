"""Pinned DUDU-BLDC v1 acquisition adapter."""

from __future__ import annotations

import json
import shutil
import stat
import urllib.request
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from trustworthy_agent.data.manifest import DUDU_BLDC_V1, DatasetManifest
from trustworthy_agent.data.schema import discover_schema, write_schema_reports
from trustworthy_agent.data.validation import validate_schema
from trustworthy_agent.provenance.hashing import md5_file, sha256_file

ZENODO_RECORD_URL = "https://zenodo.org/api/records/{record_id}"


class DatasetDownloader:
    """Explicit adapter boundary for external dataset downloads.

    Purpose:
        Download only the pinned DUDU-BLDC v1 archive, verify MD5, compute
        SHA-256, extract without modifying the archive, and persist source
        provenance.
    Parameters:
        None.
    Return value:
        Adapter instance.
    Raised exceptions:
        DatasetIntegrityError and OSError subclasses for acquisition failures.
    Scientific assumptions:
        The dataset identity is fixed by `SPEC.md`.
    Side effects:
        Writes under `Data/InputData/External/DUDU-BLDC-v1/`,
        `Output/Manifests/`, and schema-audit reports under `Output/Results/`.
    Reproducibility implications:
        Checksum verification and provenance files make local acquisition
        replayable.
    """

    def fetch(self, destination: Path) -> DatasetManifest:
        """Fetch the pinned DUDU-BLDC v1 archive into `destination`.

        Purpose:
            Backward-compatible adapter method for the DUDU-BLDC fetch workflow.
        Parameters:
            destination: Project root containing `Data/` and `Output/`.
        Return value:
            Dataset manifest with checksum metadata.
        Raised exceptions:
            OSError, ValueError, or DatasetIntegrityError on acquisition failure.
        Scientific assumptions:
            Only DUDU-BLDC v1 is supported.
        Side effects:
            May download and extract files under `destination`.
        Reproducibility implications:
            Verifies MD5 before extraction and records SHA-256.
        """

        return fetch_dudu_bldc(project_root=destination)


def fetch_dudu_bldc(project_root: Path) -> DatasetManifest:
    """Download, verify, extract, and manifest DUDU-BLDC v1.

    Purpose:
        Implement the canonical CLI data acquisition workflow.
    Parameters:
        project_root: Repository root containing `Data/` and `Output/`.
    Return value:
        Dataset manifest with local paths and checksums.
    Raised exceptions:
        ValueError for Zenodo metadata mismatch, OSError for file failures, and
        DatasetIntegrityError for checksum mismatch.
    Scientific assumptions:
        None beyond the pinned dataset identity.
    Side effects:
        Writes source archive, extracted files, schema-audit reports, and
        manifest/provenance JSON.
    Reproducibility implications:
        The raw archive is preserved and marked read-only after verification.
    """

    source_root = project_root / "Data/InputData/External/DUDU-BLDC-v1"
    manifest_root = project_root / "Output/Manifests"
    source_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    archive_path = source_root / DUDU_BLDC_V1.archive_name

    record = _read_zenodo_record(DUDU_BLDC_V1.record_id)
    file_metadata = _select_archive_file(record)
    download_url = _download_url(file_metadata)

    if not archive_path.exists():
        temporary_path = archive_path.with_suffix(".zip.part")
        _download_file(download_url, temporary_path)
        temporary_path.replace(archive_path)

    actual_md5 = md5_file(archive_path)
    if actual_md5 != DUDU_BLDC_V1.archive_md5:
        raise ValueError(
            f"MD5 mismatch for {archive_path}: "
            f"expected {DUDU_BLDC_V1.archive_md5}, got {actual_md5}."
        )
    archive_sha256 = sha256_file(archive_path)
    _mark_read_only(archive_path)

    extracted_root = source_root / "extracted"
    if not extracted_root.exists():
        _safe_extract_zip(archive_path, extracted_root)
        _mark_tree_read_only(extracted_root)

    schema = discover_schema(extracted_root)
    write_schema_reports(schema, project_root / "Output")
    validation_report = validate_schema(schema)

    manifest = replace(
        DUDU_BLDC_V1,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_path.stat().st_size,
        archive_path=str(archive_path.relative_to(project_root)),
        source_url=download_url,
        downloaded_at_utc=datetime.now(UTC).isoformat(),
        extracted_root=str(extracted_root.relative_to(project_root)),
        files=tuple(_file_records(source_root, project_root)),
    )
    _write_json(manifest_root / "dataset_manifest.json", manifest.to_json_dict())
    _write_json(
        manifest_root / "dataset_checksums.json",
        {
            "dataset_id": manifest.dataset_id,
            "record_id": manifest.record_id,
            "archive": manifest.archive_name,
            "md5": manifest.archive_md5,
            "sha256": manifest.archive_sha256,
            "files": list(manifest.files),
        },
    )
    _write_json(
        manifest_root / "source_provenance.json",
        {
            "dataset_id": manifest.dataset_id,
            "version": manifest.version,
            "doi": manifest.doi,
            "record_id": manifest.record_id,
            "license": manifest.license,
            "source_url": manifest.source_url,
            "archive_path": manifest.archive_path,
            "extracted_root": manifest.extracted_root,
            "downloaded_at_utc": manifest.downloaded_at_utc,
            "schema_validation": validation_report.to_json_dict(),
            "producer": "trustworthy_agent.data.downloader.fetch_dudu_bldc",
        },
    )
    return manifest


def _read_zenodo_record(record_id: str) -> dict[str, Any]:
    with urllib.request.urlopen(
        ZENODO_RECORD_URL.format(record_id=record_id), timeout=60
    ) as response:
        payload = response.read().decode("utf-8")
    record = cast(dict[str, Any], json.loads(payload))
    if str(record.get("id")) != record_id:
        raise ValueError(f"Resolved unexpected Zenodo record ID: {record.get('id')!r}")
    doi = record.get("doi")
    if doi != DUDU_BLDC_V1.doi:
        raise ValueError(f"Resolved unexpected DOI: {doi!r}")
    return record


def _select_archive_file(record: dict[str, Any]) -> dict[str, Any]:
    for file_metadata in record.get("files", []):
        if file_metadata.get("key") == DUDU_BLDC_V1.archive_name:
            checksum = str(file_metadata.get("checksum", ""))
            if checksum and checksum != f"md5:{DUDU_BLDC_V1.archive_md5}":
                raise ValueError(f"Zenodo checksum does not match SPEC.md: {checksum}")
            return dict(file_metadata)
    raise ValueError(f"Zenodo record does not contain {DUDU_BLDC_V1.archive_name}.")


def _download_url(file_metadata: dict[str, Any]) -> str:
    links = file_metadata.get("links", {})
    if not isinstance(links, dict) or "self" not in links:
        raise ValueError("Zenodo file metadata does not include a download URL.")
    return str(links["self"])


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = destination / member.filename
            resolved_target = target.resolve()
            if not resolved_target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(destination)


def _mark_read_only(path: Path) -> None:
    path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def _mark_tree_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            _mark_read_only(path)


def _file_records(root: Path, project_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append(
            {
                "path": str(path.relative_to(project_root)),
                "size_bytes": path.stat().st_size,
                "md5": md5_file(path),
                "sha256": sha256_file(path),
            }
        )
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
