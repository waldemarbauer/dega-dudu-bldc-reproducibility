"""Dataset manifest structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    """Pinned dataset manifest metadata.

    Purpose:
        Represent dataset identity and checksums without fetching data.
    Parameters:
        dataset_id, version, record_id, DOI, archive name, MD5, optional SHA-256.
    Return value:
        Immutable manifest metadata.
    Raised exceptions:
        None.
    Scientific assumptions:
        Dataset properties are metadata from the specification, not discovered
        schema facts.
    Side effects:
        None.
    Reproducibility implications:
        Provides stable fields for later source verification.
    """

    dataset_id: str
    version: str
    record_id: str
    doi: str
    archive_name: str
    archive_md5: str
    archive_sha256: str | None = None
    archive_size_bytes: int | None = None
    archive_path: str | None = None
    source_url: str | None = None
    license: str = "CC-BY-4.0"
    downloaded_at_utc: str | None = None
    extracted_root: str | None = None
    files: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest mapping.

        Purpose:
            Persist dataset identity, checksums, and local provenance without
            relying on dataclass internals.
        Parameters:
            None.
        Return value:
            JSON-serializable dictionary.
        Raised exceptions:
            None.
        Scientific assumptions:
            Manifest fields identify source data only; they are not schema
            validation evidence.
        Side effects:
            None.
        Reproducibility implications:
            Stable keys support future fingerprinting and audit.
        """

        return asdict(self)


DUDU_BLDC_V1 = DatasetManifest(
    dataset_id="DUDU-BLDC",
    version="v1",
    record_id="15522163",
    doi="10.5281/zenodo.15522163",
    archive_name="DUDU-BLDC.zip",
    archive_md5="b383b9ad1698a3aaf2fc05d4bf48dbe5",
)

CANONICAL_DATASET_ALIASES = frozenset({"dudu-bldc", "DUDU-BLDC"})
CANONICAL_CSV_FILES = (
    "healthy.csv",
    "healthy_zip.csv",
    "faulty.csv",
    "faulty_zip.csv",
    "motors.csv",
)
CANONICAL_CLASSES = (
    "Healthy",
    "Mech_Damage",
    "Elec_Damage",
    "Mech_Elec_Damage",
)
CANONICAL_FILE_CLASS_MAP = {
    "healthy.csv": "Healthy",
    "healthy_zip.csv": "Mech_Damage",
    "faulty.csv": "Elec_Damage",
    "faulty_zip.csv": "Mech_Elec_Damage",
}


def require_canonical_dataset(dataset: str, record_id: str | None = None) -> None:
    """Reject non-pinned dataset identifiers.

    Purpose:
        Enforce DUDU-BLDC v1 pinning before any download or validation work.
    Parameters:
        dataset: User-supplied dataset identifier.
        record_id: Optional Zenodo record ID.
    Return value:
        None.
    Raised exceptions:
        ValueError if the dataset or record ID differs from the specification.
    Scientific assumptions:
        The canonical identity comes from `SPEC.md`.
    Side effects:
        None.
    Reproducibility implications:
        Prevents silent dataset-version switching.
    """

    if dataset not in CANONICAL_DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset {dataset!r}; only DUDU-BLDC v1 is supported.")
    if record_id is not None and str(record_id) != DUDU_BLDC_V1.record_id:
        raise ValueError(
            f"Unsupported Zenodo record {record_id!r}; expected {DUDU_BLDC_V1.record_id}."
        )
