"""Immutable contracts for persisted classifier, aggregate, and risk evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar

from trustworthy_agent.evidence.provenance import (
    EvidenceProvenance,
    EvidenceValidationError,
    require_sha256,
)
from trustworthy_agent.evidence.trend import TrendEvidence

CANONICAL_CLASSES = ("Healthy", "Mech_Damage", "Elec_Damage", "Mech_Elec_Damage")


class AggregationStrategy(StrEnum):
    """Enumerate label-free probability aggregation strategies."""

    MEAN_PROBABILITY = "mean_probability"
    MEDIAN_PROBABILITY = "median_probability"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass(frozen=True)
class WindowPrediction:
    """Represent one prediction read from a persisted classifier output.

    Parameters
    ----------
    predicted_class : str
        Predicted canonical DUDU-BLDC class.
    probabilities : tuple of float
        Probability vector in ``class_order`` order; values sum to one.
    class_order : tuple of str
        Canonical class-coordinate order.
    ood_score : float
        Persisted out-of-distribution score in ``[0, 1]``.
    prediction_source_hash : str
        SHA-256 of the persisted prediction artifact.

    Notes
    -----
    Ground-truth labels are deliberately absent. This contract cannot carry
    labels into diagnostic evidence aggregation.
    """

    predicted_class: str
    probabilities: tuple[float, ...]
    class_order: tuple[str, ...]
    ood_score: float
    prediction_source_hash: str

    def __post_init__(self) -> None:
        _validate_probability_vector(self.probabilities, self.class_order)
        if self.predicted_class != _argmax_label(self.probabilities, self.class_order):
            raise EvidenceValidationError(
                "EVIDENCE_PREDICTION_MISMATCH",
                "predicted_class must equal deterministic probability argmax",
            )
        _require_unit_interval(self.ood_score, "ood_score")
        require_sha256(self.prediction_source_hash, "prediction_source_hash")


@dataclass(frozen=True)
class WindowEvidence:
    """Record persisted model evidence for one acquisition window.

    Parameters
    ----------
    prediction : str
        Predicted canonical class.
    probabilities : tuple of float
        Probability vector in ``class_order`` order.
    class_order : tuple of str
        Canonical class-coordinate order.
    confidence : float
        Maximum class probability in ``[0, 1]``.
    entropy : float
        Shannon entropy normalized by ``log(class_count)`` to ``[0, 1]``.
    ood_score : float
        Persisted OOD contribution in ``[0, 1]``.
    representation_id, representation_version : str
        Persisted representation identity and version.
    classifier_id, classifier_version : str
        Persisted classifier identity and version.
    model_hash, feature_schema_hash : str
        SHA-256 identities of the model and expected feature schema.
    assignment_id, partition, window_id, acquisition_id : str
        Split and case-local identifiers required for contamination checks.
    ordinal_index : int or None
        Ordered window index when no source timestamp exists.
    timestamp : str or None
        Source timestamp when temporal semantics are available.
    prediction_provenance : tuple of tuple of str
        Closed key/value provenance copied from the prediction export.
    provenance : EvidenceProvenance
        Shared immutable provenance envelope.

    Raises
    ------
    EvidenceValidationError
        If prediction, probability, identity, or provenance fields conflict.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    prediction: str
    probabilities: tuple[float, ...]
    class_order: tuple[str, ...]
    confidence: float
    entropy: float
    ood_score: float
    representation_id: str
    representation_version: str
    classifier_id: str
    classifier_version: str
    model_hash: str
    feature_schema_hash: str
    assignment_id: str
    partition: str
    window_id: str
    acquisition_id: str
    ordinal_index: int | None
    timestamp: str | None
    prediction_provenance: tuple[tuple[str, str], ...]
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        _validate_probability_vector(self.probabilities, self.class_order)
        if self.prediction != _argmax_label(self.probabilities, self.class_order):
            raise EvidenceValidationError(
                "EVIDENCE_PREDICTION_MISMATCH", "prediction must equal probability argmax"
            )
        if not math.isclose(self.confidence, max(self.probabilities), abs_tol=1e-12):
            raise EvidenceValidationError(
                "EVIDENCE_CONFIDENCE_MISMATCH", "confidence must equal max probability"
            )
        if not math.isclose(self.entropy, normalized_entropy(self.probabilities), abs_tol=1e-12):
            raise EvidenceValidationError(
                "EVIDENCE_ENTROPY_MISMATCH", "entropy must match the probability vector"
            )
        _require_unit_interval(self.ood_score, "ood_score")
        require_sha256(self.model_hash, "model_hash")
        require_sha256(self.feature_schema_hash, "feature_schema_hash")
        for name, value in (
            ("representation_id", self.representation_id),
            ("representation_version", self.representation_version),
            ("classifier_id", self.classifier_id),
            ("classifier_version", self.classifier_version),
            ("assignment_id", self.assignment_id),
            ("partition", self.partition),
            ("window_id", self.window_id),
            ("acquisition_id", self.acquisition_id),
        ):
            if not value:
                raise EvidenceValidationError("EVIDENCE_MISSING_FIELD", name)
        if (self.ordinal_index is None) == (self.timestamp is None):
            raise EvidenceValidationError(
                "EVIDENCE_WINDOW_ORDER_AMBIGUOUS",
                "exactly one of ordinal_index or timestamp is required",
            )
        if self.ordinal_index is not None and self.ordinal_index < 0:
            raise EvidenceValidationError(
                "EVIDENCE_INVALID_ORDINAL", "ordinal_index must be non-negative"
            )
        if self.timestamp is not None:
            try:
                datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise EvidenceValidationError(
                    "EVIDENCE_INVALID_TIMESTAMP", "window timestamp must be ISO-8601"
                ) from exc
        if not self.prediction_provenance:
            raise EvidenceValidationError(
                "EVIDENCE_UNKNOWN_PROVENANCE", "prediction provenance is mandatory"
            )
        if self.provenance.model_hash != self.model_hash:
            raise EvidenceValidationError(
                "EVIDENCE_MODEL_HASH_MISMATCH", "window and provenance model hashes differ"
            )
        if self.provenance.assignment_id != self.assignment_id:
            raise EvidenceValidationError(
                "EVIDENCE_ASSIGNMENT_MISMATCH", "window and provenance assignments differ"
            )
        if self.provenance.partition != self.partition:
            raise EvidenceValidationError(
                "EVIDENCE_PARTITION_MISMATCH", "window and provenance partitions differ"
            )

    @property
    def evidence_hash(self) -> str:
        """Return the deterministic SHA-256 identity of this evidence."""

        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the closed persisted WindowEvidence mapping."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "prediction": self.prediction,
            "probabilities": dict(zip(self.class_order, self.probabilities, strict=True)),
            "confidence": self.confidence,
            "entropy": self.entropy,
            "ood_score": self.ood_score,
            "representation_id": self.representation_id,
            "representation_version": self.representation_version,
            "classifier_id": self.classifier_id,
            "classifier_version": self.classifier_version,
            "model_hash": self.model_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "assignment_id": self.assignment_id,
            "partition": self.partition,
            "window_id": self.window_id,
            "acquisition_id": self.acquisition_id,
            "ordinal_index": self.ordinal_index,
            "timestamp": self.timestamp,
            "prediction_provenance": dict(self.prediction_provenance),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WindowEvidence:
        """Reconstruct validated WindowEvidence from persisted JSON data."""

        _require_fields(
            value,
            {
                "schema_version",
                "prediction",
                "probabilities",
                "confidence",
                "entropy",
                "ood_score",
                "representation_id",
                "representation_version",
                "classifier_id",
                "classifier_version",
                "model_hash",
                "feature_schema_hash",
                "assignment_id",
                "partition",
                "window_id",
                "acquisition_id",
                "ordinal_index",
                "timestamp",
                "prediction_provenance",
                "provenance",
            },
        )
        _require_schema(value, cls.SCHEMA_VERSION)
        probabilities = _probabilities_from_mapping(value["probabilities"])
        return cls(
            prediction=str(value["prediction"]),
            probabilities=probabilities,
            class_order=CANONICAL_CLASSES,
            confidence=float(value["confidence"]),
            entropy=float(value["entropy"]),
            ood_score=float(value["ood_score"]),
            representation_id=str(value["representation_id"]),
            representation_version=str(value["representation_version"]),
            classifier_id=str(value["classifier_id"]),
            classifier_version=str(value["classifier_version"]),
            model_hash=str(value["model_hash"]),
            feature_schema_hash=str(value["feature_schema_hash"]),
            assignment_id=str(value["assignment_id"]),
            partition=str(value["partition"]),
            window_id=str(value["window_id"]),
            acquisition_id=str(value["acquisition_id"]),
            ordinal_index=(None if value["ordinal_index"] is None else int(value["ordinal_index"])),
            timestamp=None if value["timestamp"] is None else str(value["timestamp"]),
            prediction_provenance=tuple(
                sorted(
                    (str(key), str(item)) for key, item in value["prediction_provenance"].items()
                )
            ),
            provenance=EvidenceProvenance.from_dict(dict(value["provenance"])),
        )


@dataclass(frozen=True)
class AcquisitionEvidence:
    """Aggregate label-free window probabilities for one acquisition.

    Parameters
    ----------
    aggregation_method : AggregationStrategy
        Registered deterministic aggregation method.
    input_window_hashes : tuple of str
        Ordered hashes of every valid or invalid input record.
    window_count, invalid_window_count : int
        Total and explicitly invalid input counts.
    prediction, probabilities, class_order, confidence, entropy : object
        Aggregate classifier evidence using the same semantics as
        :class:`WindowEvidence`.
    acquisition_id, assignment_id, partition : str
        Identities shared by all contributing windows.
    provenance : EvidenceProvenance
        Aggregate-producing provenance. Its input hashes must match
        ``input_window_hashes``.

    Notes
    -----
    Aggregation never accepts or reads ground-truth labels.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    aggregation_method: AggregationStrategy
    input_window_hashes: tuple[str, ...]
    window_count: int
    invalid_window_count: int
    prediction: str
    probabilities: tuple[float, ...]
    class_order: tuple[str, ...]
    confidence: float
    entropy: float
    acquisition_id: str
    assignment_id: str
    partition: str
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        if self.window_count <= 0 or self.invalid_window_count < 0:
            raise EvidenceValidationError(
                "EVIDENCE_INVALID_WINDOW_COUNT", "counts must be positive/non-negative"
            )
        if self.invalid_window_count >= self.window_count:
            raise EvidenceValidationError(
                "EVIDENCE_NO_VALID_WINDOWS", "at least one valid window is required"
            )
        if len(self.input_window_hashes) != self.window_count:
            raise EvidenceValidationError(
                "EVIDENCE_INPUT_HASH_COUNT_MISMATCH", "input hashes must match window_count"
            )
        for item in self.input_window_hashes:
            require_sha256(item, "input_window_hash")
        _validate_probability_vector(self.probabilities, self.class_order)
        if self.prediction != _argmax_label(self.probabilities, self.class_order):
            raise EvidenceValidationError(
                "EVIDENCE_PREDICTION_MISMATCH", "aggregate prediction must equal argmax"
            )
        if not math.isclose(self.confidence, max(self.probabilities), abs_tol=1e-12):
            raise EvidenceValidationError(
                "EVIDENCE_CONFIDENCE_MISMATCH", "aggregate confidence must equal max probability"
            )
        if not math.isclose(self.entropy, normalized_entropy(self.probabilities), abs_tol=1e-12):
            raise EvidenceValidationError(
                "EVIDENCE_ENTROPY_MISMATCH", "aggregate entropy must match probabilities"
            )
        if not self.acquisition_id:
            raise EvidenceValidationError(
                "EVIDENCE_MISSING_ACQUISITION", "acquisition_id is mandatory"
            )
        if self.provenance.input_hashes != self.input_window_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_INPUT_HASH_MISMATCH", "aggregate provenance inputs differ"
            )
        if self.provenance.assignment_id != self.assignment_id:
            raise EvidenceValidationError(
                "EVIDENCE_ASSIGNMENT_MISMATCH", "aggregate assignment differs"
            )
        if self.provenance.partition != self.partition:
            raise EvidenceValidationError(
                "EVIDENCE_PARTITION_MISMATCH", "aggregate partition differs"
            )

    @property
    def evidence_hash(self) -> str:
        """Return the deterministic SHA-256 identity of this aggregate."""

        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the closed persisted AcquisitionEvidence mapping."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "aggregation_method": self.aggregation_method.value,
            "input_window_hashes": list(self.input_window_hashes),
            "window_count": self.window_count,
            "invalid_window_count": self.invalid_window_count,
            "prediction": self.prediction,
            "probabilities": dict(zip(self.class_order, self.probabilities, strict=True)),
            "confidence": self.confidence,
            "entropy": self.entropy,
            "acquisition_id": self.acquisition_id,
            "assignment_id": self.assignment_id,
            "partition": self.partition,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AcquisitionEvidence:
        """Reconstruct validated AcquisitionEvidence from persisted JSON data."""

        _require_fields(
            value,
            {
                "schema_version",
                "aggregation_method",
                "input_window_hashes",
                "window_count",
                "invalid_window_count",
                "prediction",
                "probabilities",
                "confidence",
                "entropy",
                "acquisition_id",
                "assignment_id",
                "partition",
                "provenance",
            },
        )
        _require_schema(value, cls.SCHEMA_VERSION)
        return cls(
            aggregation_method=AggregationStrategy(str(value["aggregation_method"])),
            input_window_hashes=tuple(str(item) for item in value["input_window_hashes"]),
            window_count=int(value["window_count"]),
            invalid_window_count=int(value["invalid_window_count"]),
            prediction=str(value["prediction"]),
            probabilities=_probabilities_from_mapping(value["probabilities"]),
            class_order=CANONICAL_CLASSES,
            confidence=float(value["confidence"]),
            entropy=float(value["entropy"]),
            acquisition_id=str(value["acquisition_id"]),
            assignment_id=str(value["assignment_id"]),
            partition=str(value["partition"]),
            provenance=EvidenceProvenance.from_dict(dict(value["provenance"])),
        )


@dataclass(frozen=True)
class PersistedTrendEvidence:
    """Bind immutable V6 TrendEvidence to complete persisted provenance.

    The wrapped evidence is loaded with :meth:`TrendEvidence.from_dict`; this
    adapter never fits, transforms, or recomputes the trend representation.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    evidence: TrendEvidence
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        if self.evidence.configuration_hash != self.provenance.configuration_hash:
            raise EvidenceValidationError(
                "EVIDENCE_TREND_CONFIG_MISMATCH", "trend and envelope configuration hashes differ"
            )
        if self.evidence.input_window_hashes != self.provenance.input_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_TREND_INPUT_MISMATCH", "trend and envelope input hashes differ"
            )

    @property
    def evidence_hash(self) -> str:
        """Return the deterministic SHA-256 identity of the trend envelope."""

        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted trend envelope without recomputation."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "trend_evidence": self.evidence.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PersistedTrendEvidence:
        """Load immutable V6 trend evidence from a persisted envelope."""

        _require_schema(value, cls.SCHEMA_VERSION)
        return cls(
            evidence=TrendEvidence.from_dict(dict(value["trend_evidence"])),
            provenance=EvidenceProvenance.from_dict(dict(value["provenance"])),
        )


@dataclass(frozen=True)
class RiskEvidence:
    """Summarize diagnostic uncertainty without selecting an FSM decision.

    All numeric components are normalized to ``[0, 1]``. ``combined_risk`` is
    a configured convex combination of the remaining uncertainty components;
    it is evidence supplied to SafetyGuard, not an action or transition.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    combined_risk: float
    classifier_uncertainty: float
    trend_uncertainty: float
    ood_contribution: float
    healthy_deviation: float
    classifier_trend_conflict: bool
    window_prediction_instability: bool
    acquisition_id: str
    assignment_id: str
    partition: str
    input_evidence_hashes: tuple[str, ...]
    safety_guard_input: tuple[tuple[str, str], ...]
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        for name, value in (
            ("combined_risk", self.combined_risk),
            ("classifier_uncertainty", self.classifier_uncertainty),
            ("trend_uncertainty", self.trend_uncertainty),
            ("ood_contribution", self.ood_contribution),
            ("healthy_deviation", self.healthy_deviation),
        ):
            _require_unit_interval(value, name)
        if not self.safety_guard_input:
            raise EvidenceValidationError(
                "EVIDENCE_MISSING_SAFETY_INPUT", "SafetyGuard input projection is mandatory"
            )
        if self.input_evidence_hashes != self.provenance.input_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_INPUT_HASH_MISMATCH", "risk provenance inputs differ"
            )
        if self.provenance.assignment_id != self.assignment_id:
            raise EvidenceValidationError("EVIDENCE_ASSIGNMENT_MISMATCH", "risk assignment differs")

    @property
    def evidence_hash(self) -> str:
        """Return the deterministic SHA-256 identity of this risk evidence."""

        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the closed persisted RiskEvidence mapping."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "combined_risk": self.combined_risk,
            "classifier_uncertainty": self.classifier_uncertainty,
            "trend_uncertainty": self.trend_uncertainty,
            "ood_contribution": self.ood_contribution,
            "healthy_deviation": self.healthy_deviation,
            "conflict_indicators": {
                "classifier_trend_conflict": self.classifier_trend_conflict,
                "window_prediction_instability": self.window_prediction_instability,
            },
            "acquisition_id": self.acquisition_id,
            "assignment_id": self.assignment_id,
            "partition": self.partition,
            "input_evidence_hashes": list(self.input_evidence_hashes),
            "safety_guard_input": dict(self.safety_guard_input),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RiskEvidence:
        """Reconstruct validated RiskEvidence from persisted JSON data."""

        _require_schema(value, cls.SCHEMA_VERSION)
        conflicts = value["conflict_indicators"]
        return cls(
            combined_risk=float(value["combined_risk"]),
            classifier_uncertainty=float(value["classifier_uncertainty"]),
            trend_uncertainty=float(value["trend_uncertainty"]),
            ood_contribution=float(value["ood_contribution"]),
            healthy_deviation=float(value["healthy_deviation"]),
            classifier_trend_conflict=bool(conflicts["classifier_trend_conflict"]),
            window_prediction_instability=bool(conflicts["window_prediction_instability"]),
            acquisition_id=str(value["acquisition_id"]),
            assignment_id=str(value["assignment_id"]),
            partition=str(value["partition"]),
            input_evidence_hashes=tuple(str(item) for item in value["input_evidence_hashes"]),
            safety_guard_input=tuple(
                sorted((str(key), str(item)) for key, item in value["safety_guard_input"].items())
            ),
            provenance=EvidenceProvenance.from_dict(dict(value["provenance"])),
        )


@dataclass(frozen=True)
class SafetyEvidence:
    """Project evidence fields to the existing SafetyGuard context interface.

    This object contains no decision, transition, terminal state, or action.
    SafetyGuard remains authoritative when it consumes the projected fields.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    risk_score: float
    confidence: float
    ood_score: float
    spline_classifier_conflict: bool
    evidence_hashes: tuple[str, ...]
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        for name, value in (
            ("risk_score", self.risk_score),
            ("confidence", self.confidence),
            ("ood_score", self.ood_score),
        ):
            _require_unit_interval(value, name)
        if not self.evidence_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_MISSING_SAFETY_INPUT", "safety evidence hashes are mandatory"
            )
        if self.evidence_hashes != self.provenance.input_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_INPUT_HASH_MISMATCH", "safety provenance inputs differ"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible SafetyGuard input projection."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "ood_score": self.ood_score,
            "spline_classifier_conflict": self.spline_classifier_conflict,
            "evidence_hashes": list(self.evidence_hashes),
            "provenance": self.provenance.to_dict(),
        }


def normalized_entropy(probabilities: tuple[float, ...]) -> float:
    """Compute Shannon entropy normalized to ``[0, 1]``.

    Parameters
    ----------
    probabilities : tuple of float
        Non-negative probability vector that sums to one.

    Returns
    -------
    entropy : float
        ``-sum(p log(p)) / log(K)`` for ``K`` classes. Zero-probability terms
        contribute zero.

    Raises
    ------
    EvidenceValidationError
        If fewer than two probabilities are supplied or the vector is invalid.
    """

    _validate_probability_vector(probabilities, tuple(str(i) for i in range(len(probabilities))))
    terms = (-(value * math.log(value)) if value > 0.0 else 0.0 for value in probabilities)
    return sum(terms) / math.log(len(probabilities))


def stable_hash(value: dict[str, Any]) -> str:
    """Return SHA-256 over canonical compact JSON with sorted keys."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_probability_vector(
    probabilities: tuple[float, ...], class_order: tuple[str, ...]
) -> None:
    if class_order != CANONICAL_CLASSES and not all(item.isdigit() for item in class_order):
        raise EvidenceValidationError(
            "EVIDENCE_CLASS_SCHEMA_MISMATCH", "canonical DUDU-BLDC class order is required"
        )
    if len(probabilities) != len(class_order) or len(probabilities) < 2:
        raise EvidenceValidationError(
            "EVIDENCE_PROBABILITY_SHAPE", "probability and class vectors must align"
        )
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in probabilities):
        raise EvidenceValidationError(
            "EVIDENCE_INVALID_PROBABILITY", "probabilities must be finite values in [0, 1]"
        )
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise EvidenceValidationError(
            "EVIDENCE_PROBABILITY_SUM", "probability vector must sum to one"
        )


def _argmax_label(probabilities: tuple[float, ...], class_order: tuple[str, ...]) -> str:
    index = max(range(len(probabilities)), key=lambda item: probabilities[item])
    return class_order[index]


def _require_unit_interval(value: float, field_name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise EvidenceValidationError(
            "EVIDENCE_VALUE_OUT_OF_RANGE", f"{field_name} must be finite and in [0, 1]"
        )


def _require_schema(value: dict[str, Any], expected: str) -> None:
    received = value.get("schema_version")
    if received != expected:
        raise EvidenceValidationError(
            "EVIDENCE_SCHEMA_VERSION_MISMATCH", f"expected {expected}, received {received}"
        )


def _require_fields(value: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise EvidenceValidationError("EVIDENCE_MISSING_FIELD", ", ".join(missing))


def _probabilities_from_mapping(value: Any) -> tuple[float, ...]:
    if not isinstance(value, dict) or set(value) != set(CANONICAL_CLASSES):
        raise EvidenceValidationError(
            "EVIDENCE_CLASS_SCHEMA_MISMATCH",
            "probability mapping keys must match canonical classes",
        )
    return tuple(float(value[label]) for label in CANONICAL_CLASSES)
