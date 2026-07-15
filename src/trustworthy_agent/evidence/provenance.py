"""Shared provenance contract for persisted diagnostic evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceValidationError(ValueError):
    """Report a deterministic evidence-contract failure with a reason code.

    Parameters
    ----------
    code : str
        Stable machine-readable failure code.
    message : str
        Human-readable detail that must not be parsed by downstream logic.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class EvidenceProvenance:
    """Identify the immutable inputs and software that created evidence.

    Parameters
    ----------
    schema_version : str
        Version of this provenance envelope. The current version is ``1.0.0``.
    creation_timestamp : str
        ISO-8601 timestamp recorded by the producing persisted artifact.
    creator_component : str
        Versioned component identity that created the evidence.
    input_hashes : tuple of str
        Ordered lowercase SHA-256 digests of all direct inputs.
    configuration_hash : str
        Lowercase SHA-256 digest of the resolved producing configuration.
    assignment_id : str
        Stable train/validation/test assignment identity.
    partition : str
        Partition consumed by the producer.
    model_hash : str
        Lowercase SHA-256 digest of the persisted classifier artifact.
    representation_hash : str
        Lowercase SHA-256 digest of the persisted representation contract.
    healthy_reference_hash : str or None
        Training-only Healthy reference digest, or ``None`` only when no
        Healthy-reference operation is scientifically applicable.

    Raises
    ------
    EvidenceValidationError
        If provenance is missing, unknown, malformed, or unsupported.

    Notes
    -----
    The timestamp is consumed from persisted provenance; this class never
    calls the system clock, which keeps replay deterministic.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    schema_version: str
    creation_timestamp: str
    creator_component: str
    input_hashes: tuple[str, ...]
    configuration_hash: str
    assignment_id: str
    partition: str
    model_hash: str
    representation_hash: str
    healthy_reference_hash: str | None

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise EvidenceValidationError(
                "EVIDENCE_SCHEMA_VERSION_MISMATCH",
                f"expected {self.SCHEMA_VERSION}, received {self.schema_version}",
            )
        if not self.creator_component or self.creator_component.lower() in {"unknown", "none"}:
            raise EvidenceValidationError(
                "EVIDENCE_UNKNOWN_PROVENANCE", "creator_component must be explicit"
            )
        try:
            datetime.fromisoformat(self.creation_timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EvidenceValidationError(
                "EVIDENCE_INVALID_TIMESTAMP", "creation_timestamp must be ISO-8601"
            ) from exc
        if not self.input_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_MISSING_INPUT_HASH", "at least one direct input hash is required"
            )
        for name, value in (
            ("configuration_hash", self.configuration_hash),
            ("model_hash", self.model_hash),
            ("representation_hash", self.representation_hash),
        ):
            _require_sha256(value, name)
        for value in self.input_hashes:
            _require_sha256(value, "input_hash")
        if self.healthy_reference_hash is not None:
            _require_sha256(self.healthy_reference_hash, "healthy_reference_hash")
        if not self.assignment_id:
            raise EvidenceValidationError(
                "EVIDENCE_MISSING_ASSIGNMENT", "assignment_id is mandatory"
            )
        if self.partition not in {"train", "validation", "test", "production"}:
            raise EvidenceValidationError(
                "EVIDENCE_UNKNOWN_PARTITION", f"unsupported partition {self.partition!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible provenance mapping.

        Returns
        -------
        provenance : dict of str to Any
            Closed provenance envelope suitable for persistence and hashing.
        """

        return {
            "schema_version": self.schema_version,
            "creation_timestamp": self.creation_timestamp,
            "creator_component": self.creator_component,
            "input_hashes": list(self.input_hashes),
            "configuration_hash": self.configuration_hash,
            "assignment_id": self.assignment_id,
            "partition": self.partition,
            "model_hash": self.model_hash,
            "representation_hash": self.representation_hash,
            "healthy_reference_hash": self.healthy_reference_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceProvenance:
        """Reconstruct and validate persisted provenance.

        Parameters
        ----------
        value : dict of str to Any
            Persisted provenance envelope.

        Returns
        -------
        provenance : EvidenceProvenance
            Immutable validated provenance.

        Raises
        ------
        EvidenceValidationError
            If a required field is absent or invalid.
        """

        required = {
            "schema_version",
            "creation_timestamp",
            "creator_component",
            "input_hashes",
            "configuration_hash",
            "assignment_id",
            "partition",
            "model_hash",
            "representation_hash",
            "healthy_reference_hash",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise EvidenceValidationError("EVIDENCE_MISSING_PROVENANCE_FIELD", ", ".join(missing))
        return cls(
            schema_version=str(value["schema_version"]),
            creation_timestamp=str(value["creation_timestamp"]),
            creator_component=str(value["creator_component"]),
            input_hashes=tuple(str(item) for item in value["input_hashes"]),
            configuration_hash=str(value["configuration_hash"]),
            assignment_id=str(value["assignment_id"]),
            partition=str(value["partition"]),
            model_hash=str(value["model_hash"]),
            representation_hash=str(value["representation_hash"]),
            healthy_reference_hash=(
                None
                if value["healthy_reference_hash"] is None
                else str(value["healthy_reference_hash"])
            ),
        )


def require_sha256(value: str, field_name: str) -> None:
    """Validate a lowercase SHA-256 digest for another evidence contract.

    Parameters
    ----------
    value : str
        Candidate digest.
    field_name : str
        Field identity included in deterministic failure details.

    Raises
    ------
    EvidenceValidationError
        If ``value`` is not a lowercase 64-character hexadecimal digest.
    """

    _require_sha256(value, field_name)


def _require_sha256(value: str, field_name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise EvidenceValidationError(
            "EVIDENCE_INVALID_HASH", f"{field_name} must be a lowercase SHA-256 digest"
        )
