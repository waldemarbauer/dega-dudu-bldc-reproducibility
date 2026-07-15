"""Evidence contract for ordered-window temporal spline representations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar


class TrendFitStatus(StrEnum):
    """Classify successful and failed temporal trend computations."""

    FIT_OK = "FIT_OK"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    NONFINITE_INPUT = "NONFINITE_INPUT"
    UNORDERED_WINDOWS = "UNORDERED_WINDOWS"
    MIXED_ACQUISITIONS = "MIXED_ACQUISITIONS"
    UNKNOWN_HEALTHY_REFERENCE = "UNKNOWN_HEALTHY_REFERENCE"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"


@dataclass(frozen=True)
class TrendEvidence:
    """Represent case-local evolution over ordered windows from one acquisition.

    Parameters
    ----------
    trend_value : float or None
        Fitted diagnostic-index value at the last observed window.
    first_derivative : float or None
        First derivative with respect to normalized ordered-window position.
    second_derivative : float or None
        Second derivative with respect to normalized ordered-window position.
    curvature : float or None
        Absolute planar curvature at the last observed window.
    maximum_curvature : float or None
        Maximum absolute curvature over the observed sequence.
    healthy_distance : float or None
        Standardized Euclidean distance of the last observed feature vector
        from an admissible training-only Healthy reference. ``None`` means no
        reference was supplied; it never means zero distance.
    fit_rmse : float or None
        Root-mean-square residual on the observed diagnostic-index sequence.
    roughness : float or None
        Mean integrated squared second derivative over normalized position.
    effective_degrees_of_freedom : float or None
        Model-specific effective degrees of freedom for the fitted smoother.
    uncertainty : float or None
        Residual standard-error proxy; this is not predictive or physical
        failure-time uncertainty.
    window_count : int
        Number of input windows represented by this evidence.
    fit_status : TrendFitStatus
        Explicit success or failure classification.
    representation_id : str
        Stable representation identifier. V6 uses
        ``WINDOW_TEMPORAL_SPLINE``.
    representation_version : str
        Version of the representation contract.
    configuration_hash : str
        SHA-256 of the canonical resolved model configuration.
    input_window_hashes : tuple of str
        Ordered SHA-256 identities of all consumed windows.
    reason : str or None
        Machine-readable failure reason or ``None`` for a successful fit.
    metadata : dict of str to Any
        Model and validation metadata required to interpret the evidence.
    provenance : dict of str to Any
        Acquisition, assignment, partition, and implementation provenance.

    Raises
    ------
    ValueError
        If counts, hashes, successful numeric outputs, or failure reason
        semantics violate the evidence contract.

    Notes
    -----
    Derivatives are with respect to normalized ordering, not elapsed time.
    This evidence does not establish degradation, failure onset, or remaining
    useful life.
    """

    REPRESENTATION_ID: ClassVar[str] = "WINDOW_TEMPORAL_SPLINE"
    REPRESENTATION_VERSION: ClassVar[str] = "1.0.0"

    trend_value: float | None
    first_derivative: float | None
    second_derivative: float | None
    curvature: float | None
    maximum_curvature: float | None
    healthy_distance: float | None
    fit_rmse: float | None
    roughness: float | None
    effective_degrees_of_freedom: float | None
    uncertainty: float | None
    window_count: int
    fit_status: TrendFitStatus
    representation_id: str
    representation_version: str
    configuration_hash: str
    input_window_hashes: tuple[str, ...]
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate evidence without assigning defaults to unavailable values.

        Raises
        ------
        ValueError
            If the evidence is internally inconsistent or non-finite.
        """

        if self.window_count < 0:
            raise ValueError("window_count must be non-negative.")
        if len(self.input_window_hashes) != self.window_count:
            raise ValueError("input_window_hashes must match window_count.")
        if self.representation_id != self.REPRESENTATION_ID:
            raise ValueError(f"representation_id must be {self.REPRESENTATION_ID}.")
        if self.representation_version != self.REPRESENTATION_VERSION:
            raise ValueError(f"representation_version must be {self.REPRESENTATION_VERSION}.")
        if len(self.configuration_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.configuration_hash
        ):
            raise ValueError("configuration_hash must be a lowercase SHA-256 digest.")
        if any(
            len(window_hash) != 64
            or any(character not in "0123456789abcdef" for character in window_hash)
            for window_hash in self.input_window_hashes
        ):
            raise ValueError("Every input window hash must be a lowercase SHA-256 digest.")
        numeric_values = (
            self.trend_value,
            self.first_derivative,
            self.second_derivative,
            self.curvature,
            self.maximum_curvature,
            self.healthy_distance,
            self.fit_rmse,
            self.roughness,
            self.effective_degrees_of_freedom,
            self.uncertainty,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric_values):
            raise ValueError("TrendEvidence numeric fields must be finite or None.")
        required_success_values = (
            self.trend_value,
            self.first_derivative,
            self.second_derivative,
            self.curvature,
            self.maximum_curvature,
            self.fit_rmse,
            self.roughness,
            self.effective_degrees_of_freedom,
            self.uncertainty,
        )
        if self.fit_status is TrendFitStatus.FIT_OK:
            if self.reason is not None:
                raise ValueError("Successful evidence must not contain a failure reason.")
            if any(value is None for value in required_success_values):
                raise ValueError("Successful evidence requires all non-reference numeric fields.")
        elif not self.reason:
            raise ValueError("Failed evidence requires a machine-readable reason.")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible evidence mapping.

        Returns
        -------
        evidence : dict of str to Any
            Mapping containing all nineteen persisted evidence fields.

        Side Effects
        ------------
        None.
        """

        return {
            "trend_value": self.trend_value,
            "first_derivative": self.first_derivative,
            "second_derivative": self.second_derivative,
            "curvature": self.curvature,
            "maximum_curvature": self.maximum_curvature,
            "healthy_distance": self.healthy_distance,
            "fit_rmse": self.fit_rmse,
            "roughness": self.roughness,
            "effective_degrees_of_freedom": self.effective_degrees_of_freedom,
            "uncertainty": self.uncertainty,
            "window_count": self.window_count,
            "fit_status": self.fit_status.value,
            "representation_id": self.representation_id,
            "representation_version": self.representation_version,
            "configuration_hash": self.configuration_hash,
            "input_window_hashes": list(self.input_window_hashes),
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrendEvidence:
        """Reconstruct validated evidence from a persisted mapping.

        Parameters
        ----------
        value : dict of str to Any
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        evidence : TrendEvidence
            Validated immutable evidence object.

        Raises
        ------
        KeyError
            If a mandatory field is absent.
        TypeError, ValueError
            If a field cannot satisfy the evidence contract.
        """

        return cls(
            trend_value=_optional_float(value["trend_value"]),
            first_derivative=_optional_float(value["first_derivative"]),
            second_derivative=_optional_float(value["second_derivative"]),
            curvature=_optional_float(value["curvature"]),
            maximum_curvature=_optional_float(value["maximum_curvature"]),
            healthy_distance=_optional_float(value["healthy_distance"]),
            fit_rmse=_optional_float(value["fit_rmse"]),
            roughness=_optional_float(value["roughness"]),
            effective_degrees_of_freedom=_optional_float(value["effective_degrees_of_freedom"]),
            uncertainty=_optional_float(value["uncertainty"]),
            window_count=int(value["window_count"]),
            fit_status=TrendFitStatus(str(value["fit_status"])),
            representation_id=str(value["representation_id"]),
            representation_version=str(value["representation_version"]),
            configuration_hash=str(value["configuration_hash"]),
            input_window_hashes=tuple(str(item) for item in value["input_window_hashes"]),
            reason=None if value["reason"] is None else str(value["reason"]),
            metadata=dict(value["metadata"]),
            provenance=dict(value["provenance"]),
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Describe the persisted TrendEvidence JSON contract.

        Returns
        -------
        schema : dict of str to Any
            JSON-Schema-compatible description with all fields required.

        Reproducibility Implications
        ----------------------------
        A versioned closed schema prevents silent evidence-field drift.
        """

        nullable_number = {"type": ["number", "null"]}
        properties: dict[str, Any] = {
            name: dict(nullable_number)
            for name in (
                "trend_value",
                "first_derivative",
                "second_derivative",
                "curvature",
                "maximum_curvature",
                "healthy_distance",
                "fit_rmse",
                "roughness",
                "effective_degrees_of_freedom",
                "uncertainty",
            )
        }
        properties.update(
            {
                "window_count": {"type": "integer", "minimum": 0},
                "fit_status": {"enum": [status.value for status in TrendFitStatus]},
                "representation_id": {"const": cls.REPRESENTATION_ID},
                "representation_version": {"const": cls.REPRESENTATION_VERSION},
                "configuration_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "input_window_hashes": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "reason": {"type": ["string", "null"]},
                "metadata": {"type": "object"},
                "provenance": {"type": "object"},
            }
        )
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "TrendEvidence",
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        }

    def as_context_fact(self) -> dict[str, Any]:
        """Package evidence for attachment through ``AgentContext.derived_facts``.

        Returns
        -------
        fact : dict of str to Any
            A one-key mapping suitable for immutable context construction or
            ``StateResult`` facts without changing ``AgentContext``.

        Side Effects
        ------------
        None.
        """

        return {"trend_evidence": self.to_dict()}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
