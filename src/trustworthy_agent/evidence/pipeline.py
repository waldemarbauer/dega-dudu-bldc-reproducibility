"""Deterministic adapter from persisted predictions to agent evidence context."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.context.evidence import EvidenceAgentContext
from trustworthy_agent.evidence.diagnostic import (
    CANONICAL_CLASSES,
    AcquisitionEvidence,
    AggregationStrategy,
    PersistedTrendEvidence,
    RiskEvidence,
    SafetyEvidence,
    WindowEvidence,
    WindowPrediction,
    normalized_entropy,
)
from trustworthy_agent.evidence.provenance import EvidenceProvenance, EvidenceValidationError
from trustworthy_agent.evidence.trend import TrendFitStatus


@dataclass(frozen=True)
class PipelineIdentity:
    """Supply fixed provenance identities for one deterministic pipeline run.

    Parameters
    ----------
    creation_timestamp : str
        ISO-8601 timestamp originating from the persisted input/export event.
    creator_component : str
        Versioned pipeline component identity.
    configuration_hash : str
        SHA-256 of the resolved evidence-pipeline configuration.
    assignment_id, partition : str
        Stable split assignment and partition.
    model_hash, representation_hash : str
        Persisted model and representation SHA-256 identities.
    healthy_reference_hash : str or None
        Training-only Healthy reference hash when used.
    """

    creation_timestamp: str
    creator_component: str
    configuration_hash: str
    assignment_id: str
    partition: str
    model_hash: str
    representation_hash: str
    healthy_reference_hash: str | None

    def provenance(self, input_hashes: tuple[str, ...]) -> EvidenceProvenance:
        """Build validated provenance for explicit direct input hashes.

        Parameters
        ----------
        input_hashes : tuple of str
            Ordered direct input identities.

        Returns
        -------
        provenance : EvidenceProvenance
            Immutable provenance with no clock or random-state dependency.
        """

        return EvidenceProvenance(
            schema_version=EvidenceProvenance.SCHEMA_VERSION,
            creation_timestamp=self.creation_timestamp,
            creator_component=self.creator_component,
            input_hashes=input_hashes,
            configuration_hash=self.configuration_hash,
            assignment_id=self.assignment_id,
            partition=self.partition,
            model_hash=self.model_hash,
            representation_hash=self.representation_hash,
            healthy_reference_hash=self.healthy_reference_hash,
        )


@dataclass(frozen=True)
class WindowIdentity:
    """Identify a persisted window and its producing model contracts."""

    window_id: str
    acquisition_id: str
    ordinal_index: int | None
    timestamp: str | None
    representation_id: str
    representation_version: str
    classifier_id: str
    classifier_version: str
    feature_schema_hash: str
    prediction_provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RiskConfiguration:
    """Configure normalized risk evidence without encoding a decision.

    Parameters
    ----------
    classifier_uncertainty_weight, trend_uncertainty_weight : float
        Convex-combination weights for entropy and trend uncertainty.
    ood_weight, healthy_deviation_weight, conflict_weight : float
        Convex-combination weights for OOD, Healthy deviation, and conflict.
    trend_uncertainty_scale, healthy_deviation_scale : float
        Positive saturation scales that map persisted evidence to ``[0, 1]``.
    conflict_healthy_deviation_threshold : float
        Normalized Healthy-deviation threshold for comparison with classifier
        evidence. This produces only a conflict indicator.

    Raises
    ------
    EvidenceValidationError
        If weights are negative, do not sum to one, or scales are non-positive.
    """

    classifier_uncertainty_weight: float
    trend_uncertainty_weight: float
    ood_weight: float
    healthy_deviation_weight: float
    conflict_weight: float
    trend_uncertainty_scale: float
    healthy_deviation_scale: float
    conflict_healthy_deviation_threshold: float

    def __post_init__(self) -> None:
        weights = (
            self.classifier_uncertainty_weight,
            self.trend_uncertainty_weight,
            self.ood_weight,
            self.healthy_deviation_weight,
            self.conflict_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise EvidenceValidationError(
                "EVIDENCE_INVALID_RISK_CONFIG", "risk weights must be finite and non-negative"
            )
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
            raise EvidenceValidationError(
                "EVIDENCE_INVALID_RISK_CONFIG", "risk weights must sum to one"
            )
        if self.trend_uncertainty_scale <= 0.0 or self.healthy_deviation_scale <= 0.0:
            raise EvidenceValidationError(
                "EVIDENCE_INVALID_RISK_CONFIG", "risk normalization scales must be positive"
            )
        if not 0.0 <= self.conflict_healthy_deviation_threshold <= 1.0:
            raise EvidenceValidationError(
                "EVIDENCE_INVALID_RISK_CONFIG", "conflict threshold must be in [0, 1]"
            )


@dataclass(frozen=True)
class DiagnosticEvidenceBundle:
    """Persist the complete evidence chain consumed by an agent scenario."""

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    window_evidence: tuple[WindowEvidence, ...]
    acquisition_evidence: AcquisitionEvidence
    trend_evidence: PersistedTrendEvidence
    risk_evidence: RiskEvidence
    safety_evidence: SafetyEvidence
    explanation_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.window_evidence:
            raise EvidenceValidationError(
                "EVIDENCE_NO_VALID_WINDOWS", "bundle requires WindowEvidence"
            )
        window_hashes = tuple(item.evidence_hash for item in self.window_evidence)
        if window_hashes != self.acquisition_evidence.input_window_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_INPUT_HASH_MISMATCH", "bundle windows differ from acquisition inputs"
            )
        expected_risk_inputs = (
            self.acquisition_evidence.evidence_hash,
            self.trend_evidence.evidence_hash,
        )
        if expected_risk_inputs != self.risk_evidence.input_evidence_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_INPUT_HASH_MISMATCH", "bundle aggregate/trend differ from risk inputs"
            )
        expected_safety_inputs = (
            self.acquisition_evidence.evidence_hash,
            self.risk_evidence.evidence_hash,
        )
        if expected_safety_inputs != self.safety_evidence.evidence_hashes:
            raise EvidenceValidationError(
                "EVIDENCE_INPUT_HASH_MISMATCH", "bundle aggregate/risk differ from safety inputs"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the closed persisted diagnostic evidence bundle."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "window_evidence": [item.to_dict() for item in self.window_evidence],
            "acquisition_evidence": self.acquisition_evidence.to_dict(),
            "trend_evidence": self.trend_evidence.to_dict(),
            "risk_evidence": self.risk_evidence.to_dict(),
            "safety_evidence": self.safety_evidence.to_dict(),
            "explanation_references": list(self.explanation_references),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DiagnosticEvidenceBundle:
        """Load a validated evidence chain without model or trend execution."""

        if value.get("schema_version") != cls.SCHEMA_VERSION:
            raise EvidenceValidationError(
                "EVIDENCE_SCHEMA_VERSION_MISMATCH",
                f"expected {cls.SCHEMA_VERSION}, received {value.get('schema_version')}",
            )
        _reject_label_fields(value)
        window_evidence = tuple(WindowEvidence.from_dict(item) for item in value["window_evidence"])
        risk = RiskEvidence.from_dict(dict(value["risk_evidence"]))
        safety_raw = value["safety_evidence"]
        if safety_raw.get("schema_version") != SafetyEvidence.SCHEMA_VERSION:
            raise EvidenceValidationError(
                "EVIDENCE_SCHEMA_VERSION_MISMATCH",
                "SafetyEvidence schema version is unsupported",
            )
        safety = SafetyEvidence(
            risk_score=float(safety_raw["risk_score"]),
            confidence=float(safety_raw["confidence"]),
            ood_score=float(safety_raw["ood_score"]),
            spline_classifier_conflict=bool(safety_raw["spline_classifier_conflict"]),
            evidence_hashes=tuple(str(item) for item in safety_raw["evidence_hashes"]),
            provenance=EvidenceProvenance.from_dict(dict(safety_raw["provenance"])),
        )
        return cls(
            window_evidence=window_evidence,
            acquisition_evidence=AcquisitionEvidence.from_dict(dict(value["acquisition_evidence"])),
            trend_evidence=PersistedTrendEvidence.from_dict(dict(value["trend_evidence"])),
            risk_evidence=risk,
            safety_evidence=safety,
            explanation_references=tuple(str(item) for item in value["explanation_references"]),
        )

    @classmethod
    def load(cls, path: Path) -> DiagnosticEvidenceBundle:
        """Read and validate a persisted JSON bundle.

        Raises
        ------
        EvidenceValidationError
            If the artifact is absent, malformed, label-bearing, or invalid.
        """

        if not path.is_file():
            raise EvidenceValidationError(
                "EVIDENCE_ARTIFACT_MISSING", f"persisted bundle not found: {path}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceValidationError(
                "EVIDENCE_ARTIFACT_UNREADABLE", f"cannot read persisted bundle: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceValidationError(
                "EVIDENCE_SCHEMA_MISMATCH", "bundle root must be an object"
            )
        return cls.from_dict(value)


class DiagnosticEvidencePipeline:
    """Build evidence deterministically from persisted model outputs.

    This adapter never calls ``fit``, ``predict``, ``predict_proba``, or a V6
    trend method. The classifier and TrendEvidence outputs must already exist.
    """

    def __init__(self, identity: PipelineIdentity, risk: RiskConfiguration) -> None:
        self.identity = identity
        self.risk = risk
        # Validate the shared identity immediately, before evidence can enter context.
        identity.provenance(("0" * 64,))

    def make_window_evidence(
        self, prediction: WindowPrediction, identity: WindowIdentity
    ) -> WindowEvidence:
        """Convert one persisted prediction record into WindowEvidence."""

        provenance = self.identity.provenance((prediction.prediction_source_hash,))
        return WindowEvidence(
            prediction=prediction.predicted_class,
            probabilities=prediction.probabilities,
            class_order=prediction.class_order,
            confidence=max(prediction.probabilities),
            entropy=normalized_entropy(prediction.probabilities),
            ood_score=prediction.ood_score,
            representation_id=identity.representation_id,
            representation_version=identity.representation_version,
            classifier_id=identity.classifier_id,
            classifier_version=identity.classifier_version,
            model_hash=self.identity.model_hash,
            feature_schema_hash=identity.feature_schema_hash,
            assignment_id=self.identity.assignment_id,
            partition=self.identity.partition,
            window_id=identity.window_id,
            acquisition_id=identity.acquisition_id,
            ordinal_index=identity.ordinal_index,
            timestamp=identity.timestamp,
            prediction_provenance=identity.prediction_provenance,
            provenance=provenance,
        )

    def aggregate(
        self,
        windows: Sequence[WindowEvidence],
        strategy: AggregationStrategy,
        *,
        invalid_window_hashes: tuple[str, ...] = (),
    ) -> AcquisitionEvidence:
        """Aggregate window probabilities without access to labels.

        Raises
        ------
        EvidenceValidationError
            If no valid windows exist or identities, assignments, partitions,
            models, representations, or feature schemas are mixed.
        """

        if not windows:
            raise EvidenceValidationError(
                "EVIDENCE_NO_VALID_WINDOWS", "aggregation requires at least one window"
            )
        _validate_homogeneous_windows(windows)
        probabilities = _aggregate_probabilities(windows, strategy)
        input_hashes = (*tuple(item.evidence_hash for item in windows), *invalid_window_hashes)
        first = windows[0]
        provenance = self.identity.provenance(input_hashes)
        return AcquisitionEvidence(
            aggregation_method=strategy,
            input_window_hashes=input_hashes,
            window_count=len(input_hashes),
            invalid_window_count=len(invalid_window_hashes),
            prediction=_argmax(probabilities),
            probabilities=probabilities,
            class_order=CANONICAL_CLASSES,
            confidence=max(probabilities),
            entropy=normalized_entropy(probabilities),
            acquisition_id=first.acquisition_id,
            assignment_id=first.assignment_id,
            partition=first.partition,
            provenance=provenance,
        )

    def make_risk_evidence(
        self,
        acquisition: AcquisitionEvidence,
        trend: PersistedTrendEvidence,
        windows: Sequence[WindowEvidence],
    ) -> RiskEvidence:
        """Combine persisted uncertainty components without making a decision.

        Raises
        ------
        EvidenceValidationError
            If trend evidence failed, or identifiers/provenance do not align.
        """

        if trend.evidence.fit_status is not TrendFitStatus.FIT_OK:
            raise EvidenceValidationError(
                "EVIDENCE_TREND_NOT_USABLE",
                f"persisted TrendEvidence status is {trend.evidence.fit_status.value}",
            )
        if trend.provenance.assignment_id != acquisition.assignment_id:
            raise EvidenceValidationError(
                "EVIDENCE_ASSIGNMENT_MISMATCH", "trend and acquisition assignments differ"
            )
        if trend.provenance.partition != acquisition.partition:
            raise EvidenceValidationError(
                "EVIDENCE_PARTITION_MISMATCH", "trend and acquisition partitions differ"
            )
        if trend.provenance.model_hash != acquisition.provenance.model_hash:
            raise EvidenceValidationError(
                "EVIDENCE_MODEL_HASH_MISMATCH", "trend and acquisition model hashes differ"
            )
        if (
            trend.evidence.healthy_distance is not None
            and trend.provenance.healthy_reference_hash is None
        ):
            raise EvidenceValidationError(
                "EVIDENCE_UNKNOWN_HEALTHY_REFERENCE",
                "Healthy deviation requires a training-only reference hash",
            )
        trend_acquisition = str(trend.evidence.provenance.get("acquisition_id", ""))
        if trend_acquisition != acquisition.acquisition_id:
            raise EvidenceValidationError(
                "EVIDENCE_MIXED_ACQUISITIONS", "trend and aggregate acquisitions differ"
            )
        trend_uncertainty = _saturate(trend.evidence.uncertainty, self.risk.trend_uncertainty_scale)
        healthy_deviation = _saturate(
            trend.evidence.healthy_distance, self.risk.healthy_deviation_scale
        )
        ood_contribution = max(window.ood_score for window in windows)
        trend_anomalous = healthy_deviation >= self.risk.conflict_healthy_deviation_threshold
        classifier_anomalous = acquisition.prediction != "Healthy"
        conflict = trend_anomalous != classifier_anomalous
        instability = len({window.prediction for window in windows}) > 1
        conflict_component = 1.0 if conflict or instability else 0.0
        combined = (
            self.risk.classifier_uncertainty_weight * acquisition.entropy
            + self.risk.trend_uncertainty_weight * trend_uncertainty
            + self.risk.ood_weight * ood_contribution
            + self.risk.healthy_deviation_weight * healthy_deviation
            + self.risk.conflict_weight * conflict_component
        )
        inputs = (acquisition.evidence_hash, trend.evidence_hash)
        provenance = self.identity.provenance(inputs)
        safety_input = (
            ("confidence", str(acquisition.confidence)),
            ("ood_score", str(ood_contribution)),
            ("risk_score", str(combined)),
            ("spline_classifier_conflict", str(conflict).lower()),
        )
        return RiskEvidence(
            combined_risk=combined,
            classifier_uncertainty=acquisition.entropy,
            trend_uncertainty=trend_uncertainty,
            ood_contribution=ood_contribution,
            healthy_deviation=healthy_deviation,
            classifier_trend_conflict=conflict,
            window_prediction_instability=instability,
            acquisition_id=acquisition.acquisition_id,
            assignment_id=acquisition.assignment_id,
            partition=acquisition.partition,
            input_evidence_hashes=inputs,
            safety_guard_input=safety_input,
            provenance=provenance,
        )

    def make_safety_evidence(
        self, acquisition: AcquisitionEvidence, risk: RiskEvidence
    ) -> SafetyEvidence:
        """Project diagnostic evidence to fields already read by SafetyGuard."""

        input_hashes = (acquisition.evidence_hash, risk.evidence_hash)
        return SafetyEvidence(
            risk_score=risk.combined_risk,
            confidence=acquisition.confidence,
            ood_score=risk.ood_contribution,
            spline_classifier_conflict=risk.classifier_trend_conflict,
            evidence_hashes=input_hashes,
            provenance=self.identity.provenance(input_hashes),
        )

    def attach_to_context(
        self, context: AgentContext, bundle: DiagnosticEvidenceBundle
    ) -> EvidenceAgentContext:
        """Attach evidence to a new backward-compatible AgentContext.

        Existing scalar fields are populated as an explicit projection for
        unchanged strategies and SafetyGuard. No state, transition, action, or
        audit field is changed.
        """

        acquisition = bundle.acquisition_evidence
        risk = bundle.risk_evidence
        return EvidenceAgentContext.from_context(
            context,
            diagnosis=acquisition.prediction,
            class_probabilities=dict(
                zip(acquisition.class_order, acquisition.probabilities, strict=True)
            ),
            confidence=acquisition.confidence,
            risk_score=risk.combined_risk,
            ood_score=risk.ood_contribution,
            spline_classifier_conflict=risk.classifier_trend_conflict,
            window_evidence=bundle.window_evidence,
            acquisition_evidence=acquisition,
            trend_evidence=bundle.trend_evidence,
            risk_evidence=risk,
            safety_evidence=bundle.safety_evidence,
            explanation_references=bundle.explanation_references,
            representation_metadata={
                "representation_id": bundle.window_evidence[0].representation_id,
                "representation_version": bundle.window_evidence[0].representation_version,
                "representation_hash": self.identity.representation_hash,
            },
            model_metadata={
                "classifier_id": bundle.window_evidence[0].classifier_id,
                "classifier_version": bundle.window_evidence[0].classifier_version,
                "model_hash": self.identity.model_hash,
            },
            healthy_reference_metadata={
                "healthy_reference_hash": self.identity.healthy_reference_hash,
                "fit_scope": "training_only"
                if self.identity.healthy_reference_hash is not None
                else "not_applicable",
            },
        )


def _aggregate_probabilities(
    windows: Sequence[WindowEvidence], strategy: AggregationStrategy
) -> tuple[float, ...]:
    columns = tuple(zip(*(item.probabilities for item in windows), strict=True))
    if strategy is AggregationStrategy.MEAN_PROBABILITY:
        values = tuple(sum(column) / len(column) for column in columns)
    elif strategy is AggregationStrategy.MEDIAN_PROBABILITY:
        values = tuple(float(statistics.median(column)) for column in columns)
    elif strategy is AggregationStrategy.MAJORITY_VOTE:
        counts = [0.0] * len(CANONICAL_CLASSES)
        for window in windows:
            counts[CANONICAL_CLASSES.index(window.prediction)] += 1.0
        values = tuple(item / len(windows) for item in counts)
    elif strategy is AggregationStrategy.CONFIDENCE_WEIGHTED:
        denominator = sum(window.confidence for window in windows)
        if denominator <= 0.0:
            raise EvidenceValidationError(
                "EVIDENCE_INVALID_WEIGHT", "confidence-weight denominator must be positive"
            )
        values = tuple(
            sum(window.confidence * window.probabilities[index] for window in windows) / denominator
            for index in range(len(CANONICAL_CLASSES))
        )
    else:  # pragma: no cover - enum construction prevents this branch
        raise EvidenceValidationError(
            "EVIDENCE_UNKNOWN_AGGREGATION", f"unknown aggregation strategy {strategy}"
        )
    total = sum(values)
    if total <= 0.0:
        raise EvidenceValidationError(
            "EVIDENCE_INVALID_PROBABILITY", "aggregate probability mass must be positive"
        )
    return tuple(item / total for item in values)


def _validate_homogeneous_windows(windows: Sequence[WindowEvidence]) -> None:
    checks: tuple[tuple[str, str], ...] = (
        ("acquisition_id", "EVIDENCE_MIXED_ACQUISITIONS"),
        ("assignment_id", "EVIDENCE_MIXED_ASSIGNMENTS"),
        ("partition", "EVIDENCE_MIXED_PARTITIONS"),
        ("model_hash", "EVIDENCE_MIXED_MODELS"),
        ("representation_id", "EVIDENCE_MIXED_REPRESENTATIONS"),
        ("feature_schema_hash", "EVIDENCE_MIXED_FEATURE_SCHEMAS"),
    )
    for attribute, code in checks:
        if len({getattr(item, attribute) for item in windows}) != 1:
            raise EvidenceValidationError(code, f"window {attribute} values differ")
    if len({item.window_id for item in windows}) != len(windows):
        raise EvidenceValidationError(
            "EVIDENCE_DUPLICATE_WINDOW", "window_id values must be unique"
        )


def _argmax(probabilities: tuple[float, ...]) -> str:
    index = max(range(len(probabilities)), key=lambda item: probabilities[item])
    return CANONICAL_CLASSES[index]


def _saturate(value: float | None, scale: float) -> float:
    if value is None:
        raise EvidenceValidationError(
            "EVIDENCE_TREND_FIELD_MISSING", "required persisted trend component is unavailable"
        )
    return min(1.0, abs(value) / scale)


def _reject_label_fields(value: Mapping[str, Any], path: str = "bundle") -> None:
    forbidden = {"label", "true_label", "ground_truth", "target", "y_true"}
    for key, item in value.items():
        if str(key).lower() in forbidden:
            raise EvidenceValidationError("EVIDENCE_LABEL_LEAKAGE", f"forbidden field {path}.{key}")
        if isinstance(item, Mapping):
            _reject_label_fields(item, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                if isinstance(child, Mapping):
                    _reject_label_fields(child, f"{path}.{key}[{index}]")
