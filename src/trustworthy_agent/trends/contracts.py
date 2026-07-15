"""Public contracts for case-local ordered-window trend models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

from trustworthy_agent.evidence import TrendEvidence


class TrendMode(StrEnum):
    """Select complete-sequence or observed-history-only operation."""

    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"


class WindowPartition(StrEnum):
    """Declare the frozen assignment partition of an ordered window."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class OrderedWindow:
    """Describe one canonical diagnostic window in an acquisition sequence.

    Parameters
    ----------
    acquisition_id : str
        Stable identity of the single source acquisition.
    window_id : str
        Stable identity unique within the acquisition.
    window_order : int
        Explicit zero-based or positive ordinal used to order windows.
    assignment_id : str
        Identity of the active frozen train/validation/test assignment.
    partition : str
        Partition label under the active assignment.
    features : tuple of float
        Canonical finite feature vector with stable column order.
    timestamp : str or None, optional
        ISO-8601 acquisition-local timestamp. Required when
        ``ordinal_position`` is absent.
    ordinal_position : float or None, optional
        Acquisition-local numeric position. Required when ``timestamp`` is
        absent. It is not interpreted as degradation time.

    Notes
    -----
    Structural validation is performed by a ``TrendModel`` so rejected input
    can carry structured failure evidence, provenance, and reason codes.
    """

    acquisition_id: str
    window_id: str
    window_order: int
    assignment_id: str
    partition: str
    features: tuple[float, ...]
    timestamp: str | None = None
    ordinal_position: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted representation of a canonical window.

        Returns
        -------
        window : dict of str to Any
            JSON-compatible field mapping for valid finite windows.
        """

        return {
            "acquisition_id": self.acquisition_id,
            "window_id": self.window_id,
            "window_order": self.window_order,
            "assignment_id": self.assignment_id,
            "partition": self.partition,
            "features": list(self.features),
            "timestamp": self.timestamp,
            "ordinal_position": self.ordinal_position,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OrderedWindow:
        """Reconstruct one window from deterministic model persistence.

        Parameters
        ----------
        value : dict of str to Any
            Persisted window field mapping.

        Returns
        -------
        window : OrderedWindow
            Immutable canonical window.

        Raises
        ------
        KeyError, TypeError, ValueError
            If mandatory fields are absent or have incompatible values.
        """

        return cls(
            acquisition_id=str(value["acquisition_id"]),
            window_id=str(value["window_id"]),
            window_order=int(value["window_order"]),
            assignment_id=str(value["assignment_id"]),
            partition=str(value["partition"]),
            features=tuple(float(item) for item in value["features"]),
            timestamp=None if value.get("timestamp") is None else str(value["timestamp"]),
            ordinal_position=(
                None if value.get("ordinal_position") is None else float(value["ordinal_position"])
            ),
        )

    def stable_hash(self) -> str:
        """Compute a deterministic identity even for rejected non-finite input.

        Returns
        -------
        digest : str
            Lowercase SHA-256 over canonical window identity and values.

        Notes
        -----
        Non-finite values are represented by explicit tokens only for hashing;
        they remain invalid model inputs and cannot enter a successful fit.
        """

        payload = {
            **self.to_dict(),
            "features": [_hashable_float(item) for item in self.features],
            "ordinal_position": _hashable_float(self.ordinal_position),
        }
        return canonical_hash(payload)


@dataclass(frozen=True)
class HealthyReference:
    """Carry a training-only Healthy feature reference with split provenance.

    Parameters
    ----------
    reference_id : str
        Stable reference artifact identity.
    assignment_id : str
        Frozen assignment from which reference rows originated.
    source_partitions : tuple of str
        All partitions contributing to the reference. Only exactly
        ``("train",)`` is admissible.
    feature_mean : tuple of float
        Training Healthy mean for each canonical feature.
    feature_scale : tuple of float
        Positive training Healthy scale for each canonical feature.
    source_window_hashes : tuple of str
        Identities of training windows used to construct the reference.
    reference_version : str, optional
        Version of the reference artifact contract.

    Scientific Assumptions
    ----------------------
    The reference is a split-local statistical baseline, not a universal
    physical definition of health.
    """

    reference_id: str
    assignment_id: str
    source_partitions: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    source_window_hashes: tuple[str, ...]
    reference_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic reference fields for persistence and hashing.

        Returns
        -------
        reference : dict of str to Any
            JSON-compatible Healthy reference mapping.
        """

        return {
            "reference_id": self.reference_id,
            "assignment_id": self.assignment_id,
            "source_partitions": list(self.source_partitions),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "source_window_hashes": list(self.source_window_hashes),
            "reference_version": self.reference_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HealthyReference:
        """Reconstruct a Healthy reference from persisted JSON fields.

        Parameters
        ----------
        value : dict of str to Any
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        reference : HealthyReference
            Immutable reference object; admissibility is checked during fit.
        """

        return cls(
            reference_id=str(value["reference_id"]),
            assignment_id=str(value["assignment_id"]),
            source_partitions=tuple(str(item) for item in value["source_partitions"]),
            feature_mean=tuple(float(item) for item in value["feature_mean"]),
            feature_scale=tuple(float(item) for item in value["feature_scale"]),
            source_window_hashes=tuple(str(item) for item in value["source_window_hashes"]),
            reference_version=str(value.get("reference_version", "1.0.0")),
        )

    def stable_hash(self) -> str:
        """Compute the canonical SHA-256 identity of the reference.

        Returns
        -------
        digest : str
            Lowercase SHA-256 over all reference fields.
        """

        return canonical_hash(self.to_dict())


class TrendModelFailure(RuntimeError):
    """Reject a trend operation while preserving structured failure evidence.

    Parameters
    ----------
    evidence : TrendEvidence
        Failed evidence containing reason, metadata, and provenance.

    Attributes
    ----------
    evidence : TrendEvidence
        Machine-readable failure result available to callers and audit layers.
    """

    def __init__(self, evidence: TrendEvidence) -> None:
        if evidence.reason is None:
            raise ValueError("TrendModelFailure requires failed evidence with a reason.")
        self.evidence = evidence
        super().__init__(evidence.reason)


@runtime_checkable
class TrendModel(Protocol):
    """Define the common lifecycle of every V6 temporal trend model."""

    model_name: str
    model_version: str

    def fit(
        self,
        windows: Sequence[OrderedWindow],
        *,
        healthy_reference: HealthyReference | None = None,
    ) -> Self:
        """Validate and retain one acquisition's ordered input history."""

        ...

    def transform(
        self,
        windows: Sequence[OrderedWindow] | None = None,
        *,
        observed_through: int | None = None,
    ) -> TrendEvidence:
        """Compute evidence from complete or already-observed history."""

        ...

    def update(
        self,
        window: OrderedWindow,
        *,
        observed_through: int | None = None,
    ) -> TrendEvidence:
        """Append one observed online window and recompute current evidence."""

        ...

    def save(self, path: Path) -> Path:
        """Persist configuration, validated history, and reference as JSON."""

        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore a model from deterministic JSON persistence."""

        ...

    def schema(self) -> dict[str, Any]:
        """Return the input and TrendEvidence schema contract."""

        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return stable implementation, configuration, and fit metadata."""

        ...


def canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic canonical settings.

    Parameters
    ----------
    value : Any
        Value containing only finite JSON-compatible content.

    Returns
    -------
    digest : str
        Lowercase SHA-256 of sorted, compact UTF-8 JSON.

    Raises
    ------
    TypeError, ValueError
        If the value is not JSON serializable or contains NaN/infinity.

    Reproducibility Implications
    ----------------------------
    The digest excludes timestamps and machine paths unless explicitly present
    in the supplied value.
    """

    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _hashable_float(value: float | None) -> float | str | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric):
        return "NONFINITE_NAN"
    if numeric == math.inf:
        return "NONFINITE_POSITIVE_INFINITY"
    if numeric == -math.inf:
        return "NONFINITE_NEGATIVE_INFINITY"
    return numeric
