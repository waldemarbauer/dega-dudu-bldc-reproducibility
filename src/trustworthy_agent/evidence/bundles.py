"""Persist complete diagnostic evidence bundles without executing diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trustworthy_agent.evidence.diagnostic import (
    AggregationStrategy,
    PersistedTrendEvidence,
    WindowPrediction,
    stable_hash,
)
from trustworthy_agent.evidence.pipeline import (
    DiagnosticEvidenceBundle,
    DiagnosticEvidencePipeline,
    PipelineIdentity,
    RiskConfiguration,
    WindowIdentity,
)
from trustworthy_agent.evidence.provenance import (
    EvidenceProvenance,
    EvidenceValidationError,
    require_sha256,
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class BundleMetadata:
    """Identify one acquisition-level evidence bundle.

    Parameters
    ----------
    scenario_id : str
        Natural article scenario that will later consume the bundle.
    assignment_id, partition, acquisition_id : str
        Persisted assignment, split partition, and acquisition identities.
    representation_id, representation_version, representation_hash : str
        Classifier representation identity, version, and SHA-256 digest.
    classifier_id, classifier_version, classifier_hash : str
        Persisted classifier identity, version, and model SHA-256 digest.
    trend_model_id, trend_model_version, trend_evidence_hash : str
        Persisted V6 producer identity/version and evidence-envelope SHA-256.
    healthy_reference_hash : str
        SHA-256 of the training-only Healthy reference.

    Raises
    ------
    EvidenceValidationError
        If an identity is absent or a digest is malformed.
    """

    scenario_id: str
    assignment_id: str
    partition: str
    acquisition_id: str
    representation_id: str
    representation_version: str
    representation_hash: str
    classifier_id: str
    classifier_version: str
    classifier_hash: str
    trend_model_id: str
    trend_model_version: str
    trend_evidence_hash: str
    healthy_reference_hash: str

    def __post_init__(self) -> None:
        for name, value in (
            ("scenario_id", self.scenario_id),
            ("assignment_id", self.assignment_id),
            ("partition", self.partition),
            ("acquisition_id", self.acquisition_id),
            ("representation_id", self.representation_id),
            ("representation_version", self.representation_version),
            ("classifier_id", self.classifier_id),
            ("classifier_version", self.classifier_version),
            ("trend_model_id", self.trend_model_id),
            ("trend_model_version", self.trend_model_version),
        ):
            if not value or value.lower() == "unknown":
                raise EvidenceValidationError("EVIDENCE_UNKNOWN_PROVENANCE", name)
        for name, value in (
            ("representation_hash", self.representation_hash),
            ("classifier_hash", self.classifier_hash),
            ("trend_evidence_hash", self.trend_evidence_hash),
            ("healthy_reference_hash", self.healthy_reference_hash),
        ):
            require_sha256(value, name)

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic JSON-compatible metadata mapping."""

        return {
            "scenario_id": self.scenario_id,
            "assignment_id": self.assignment_id,
            "partition": self.partition,
            "acquisition_id": self.acquisition_id,
            "representation_id": self.representation_id,
            "representation_version": self.representation_version,
            "representation_hash": self.representation_hash,
            "classifier_id": self.classifier_id,
            "classifier_version": self.classifier_version,
            "classifier_hash": self.classifier_hash,
            "trend_model_id": self.trend_model_id,
            "trend_model_version": self.trend_model_version,
            "trend_evidence_hash": self.trend_evidence_hash,
            "healthy_reference_hash": self.healthy_reference_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BundleMetadata:
        """Reconstruct validated metadata from persisted JSON data."""

        required = {
            "scenario_id",
            "assignment_id",
            "partition",
            "acquisition_id",
            "representation_id",
            "representation_version",
            "representation_hash",
            "classifier_id",
            "classifier_version",
            "classifier_hash",
            "trend_model_id",
            "trend_model_version",
            "trend_evidence_hash",
            "healthy_reference_hash",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise EvidenceValidationError("EVIDENCE_MISSING_FIELD", ", ".join(missing))
        return cls(**{name: str(value[name]) for name in required})


@dataclass(frozen=True)
class PersistedEvidenceBundle:
    """Bind an immutable evidence chain to bundle metadata and configuration.

    Parameters
    ----------
    metadata : BundleMetadata
        Acquisition, model, representation, trend, and Healthy identities.
    evidence : DiagnosticEvidenceBundle
        Complete validated Window/Acquisition/Trend/Risk/Safety chain.
    configuration_json : str
        Canonical JSON object describing the bundle-generation configuration.
    provenance : EvidenceProvenance
        Bundle-level provenance whose configuration and input hashes must match.

    Raises
    ------
    EvidenceValidationError
        If any evidence identity, configuration hash, or input hash conflicts.

    Notes
    -----
    Bundle construction never calls classifier inference, TrendModel methods,
    FSM transitions, TransitionPolicy, SafetyGuard, or scenario execution.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    GENERATOR_VERSION: ClassVar[str] = "1.0.0"

    metadata: BundleMetadata
    evidence: DiagnosticEvidenceBundle
    configuration_json: str
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        configuration = _configuration(self.configuration_json)
        configuration_hash = stable_hash(configuration)
        if self.provenance.configuration_hash != configuration_hash:
            raise EvidenceValidationError(
                "EVIDENCE_CONFIGURATION_MISMATCH",
                "bundle configuration does not match provenance",
            )
        expected_inputs = _bundle_input_hashes(self.evidence)
        if self.provenance.input_hashes != expected_inputs:
            raise EvidenceValidationError(
                "EVIDENCE_INPUT_HASH_MISMATCH", "bundle input hashes differ from evidence"
            )
        if self.metadata.assignment_id != self.evidence.acquisition_evidence.assignment_id:
            raise EvidenceValidationError(
                "EVIDENCE_ASSIGNMENT_MISMATCH", "bundle and acquisition assignments differ"
            )
        if self.metadata.partition != self.evidence.acquisition_evidence.partition:
            raise EvidenceValidationError(
                "EVIDENCE_PARTITION_MISMATCH", "bundle and acquisition partitions differ"
            )
        if self.metadata.acquisition_id != self.evidence.acquisition_evidence.acquisition_id:
            raise EvidenceValidationError(
                "EVIDENCE_MIXED_ACQUISITIONS", "bundle and evidence acquisitions differ"
            )
        first = self.evidence.window_evidence[0]
        if self.metadata.classifier_hash != first.model_hash:
            raise EvidenceValidationError(
                "EVIDENCE_MODEL_HASH_MISMATCH", "bundle and window classifier hashes differ"
            )
        if self.metadata.representation_hash != first.provenance.representation_hash:
            raise EvidenceValidationError(
                "EVIDENCE_REPRESENTATION_HASH_MISMATCH",
                "bundle and window representation hashes differ",
            )
        if self.metadata.trend_evidence_hash != self.evidence.trend_evidence.evidence_hash:
            raise EvidenceValidationError(
                "EVIDENCE_TREND_HASH_MISMATCH", "bundle and TrendEvidence hashes differ"
            )
        healthy_hash = self.evidence.trend_evidence.provenance.healthy_reference_hash
        if healthy_hash != self.metadata.healthy_reference_hash:
            raise EvidenceValidationError(
                "EVIDENCE_HEALTHY_REFERENCE_MISMATCH",
                "bundle and TrendEvidence Healthy hashes differ",
            )
        if self.provenance.model_hash != self.metadata.classifier_hash:
            raise EvidenceValidationError(
                "EVIDENCE_MODEL_HASH_MISMATCH", "bundle provenance model hash differs"
            )
        if self.provenance.representation_hash != self.metadata.representation_hash:
            raise EvidenceValidationError(
                "EVIDENCE_REPRESENTATION_HASH_MISMATCH",
                "bundle provenance representation hash differs",
            )
        if self.provenance.healthy_reference_hash != self.metadata.healthy_reference_hash:
            raise EvidenceValidationError(
                "EVIDENCE_HEALTHY_REFERENCE_MISMATCH",
                "bundle provenance Healthy hash differs",
            )

    @property
    def configuration_hash(self) -> str:
        """Return the canonical bundle-generation configuration hash."""

        return stable_hash(_configuration(self.configuration_json))

    @property
    def bundle_hash(self) -> str:
        """Return SHA-256 over canonical bundle content excluding this hash."""

        return stable_hash(self._content_dict())

    @property
    def filename_stem(self) -> str:
        """Encode required scientific identities in a safe bundle filename."""

        components = (
            self.metadata.assignment_id,
            self.metadata.acquisition_id,
            self.metadata.classifier_id,
            self.metadata.representation_id,
            self.metadata.trend_model_id,
            f"HR-{self.metadata.healthy_reference_hash[:12]}",
        )
        return "__".join(_safe_component(item) for item in components) + ".bundle"

    def to_dict(self) -> dict[str, Any]:
        """Return a closed JSON mapping including its replay hash."""

        return {**self._content_dict(), "bundle_hash": self.bundle_hash}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PersistedEvidenceBundle:
        """Reconstruct and verify a persisted evidence bundle.

        Raises
        ------
        EvidenceValidationError
            If fields are missing, schemas differ, or the recorded hash fails.
        """

        required = {
            "schema_version",
            "generator_version",
            "metadata",
            "configuration",
            "configuration_hash",
            "window_evidence",
            "acquisition_evidence",
            "trend_evidence",
            "risk_evidence",
            "safety_evidence",
            "explanation_references",
            "provenance",
            "bundle_hash",
        }
        missing = sorted(required - value.keys())
        if missing:
            code = (
                "EVIDENCE_TREND_MISSING"
                if "trend_evidence" in missing
                else "EVIDENCE_MISSING_FIELD"
            )
            raise EvidenceValidationError(code, ", ".join(missing))
        if value["schema_version"] != cls.SCHEMA_VERSION:
            raise EvidenceValidationError(
                "EVIDENCE_SCHEMA_VERSION_MISMATCH",
                f"unsupported bundle schema {value['schema_version']}",
            )
        if value["generator_version"] != cls.GENERATOR_VERSION:
            raise EvidenceValidationError(
                "EVIDENCE_SCHEMA_VERSION_MISMATCH",
                f"unsupported generator version {value['generator_version']}",
            )
        configuration_json = _canonical_json(value["configuration"])
        bundle = cls(
            metadata=BundleMetadata.from_dict(dict(value["metadata"])),
            evidence=DiagnosticEvidenceBundle.from_dict(value),
            configuration_json=configuration_json,
            provenance=EvidenceProvenance.from_dict(dict(value["provenance"])),
        )
        if str(value["configuration_hash"]) != bundle.configuration_hash:
            raise EvidenceValidationError(
                "EVIDENCE_CONFIGURATION_MISMATCH", "recorded configuration hash differs"
            )
        if str(value["bundle_hash"]) != bundle.bundle_hash:
            raise EvidenceValidationError(
                "EVIDENCE_BUNDLE_HASH_MISMATCH", "recorded bundle hash differs"
            )
        return bundle

    @classmethod
    def load_json(cls, path: Path) -> PersistedEvidenceBundle:
        """Load and verify a JSON bundle from disk."""

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceValidationError(
                "EVIDENCE_ARTIFACT_UNREADABLE", f"cannot load JSON bundle {path}"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceValidationError(
                "EVIDENCE_SCHEMA_MISMATCH", "bundle JSON root must be an object"
            )
        return cls.from_dict(value)

    @classmethod
    def load_parquet(cls, path: Path) -> PersistedEvidenceBundle:
        """Load and verify a one-row Parquet bundle from disk."""

        try:
            table = pq.read_table(path)
        except (OSError, pa.ArrowException) as exc:
            raise EvidenceValidationError(
                "EVIDENCE_ARTIFACT_UNREADABLE", f"cannot load Parquet bundle {path}"
            ) from exc
        if table.num_rows != 1 or "bundle_json" not in table.column_names:
            raise EvidenceValidationError(
                "EVIDENCE_SCHEMA_MISMATCH", "Parquet bundle must contain one bundle_json row"
            )
        row = table.to_pylist()[0]
        bundle = cls.from_dict(json.loads(str(row["bundle_json"])))
        expected_scalars = {
            "bundle_hash": bundle.bundle_hash,
            "assignment_id": bundle.metadata.assignment_id,
            "partition": bundle.metadata.partition,
            "acquisition_id": bundle.metadata.acquisition_id,
        }
        if any(str(row[name]) != expected for name, expected in expected_scalars.items()):
            raise EvidenceValidationError(
                "EVIDENCE_PARQUET_METADATA_MISMATCH",
                "Parquet scalar metadata differs from embedded bundle JSON",
            )
        return bundle

    def _content_dict(self) -> dict[str, Any]:
        configuration = _configuration(self.configuration_json)
        return {
            **self.evidence.to_dict(),
            "schema_version": self.SCHEMA_VERSION,
            "generator_version": self.GENERATOR_VERSION,
            "metadata": self.metadata.to_dict(),
            "configuration": configuration,
            "configuration_hash": stable_hash(configuration),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class BundleArtifact:
    """Record the persisted files and hashes for one evidence bundle."""

    bundle_hash: str
    json_path: Path
    json_sha256: str
    parquet_path: Path
    parquet_sha256: str
    size_bytes: int
    metadata: BundleMetadata


@dataclass(frozen=True)
class PersistedAcquisitionInputs:
    """Collect already persisted inputs for one acquisition bundle.

    Parameters
    ----------
    scenario_id : str
        Natural article scenario identity; the scenario is not executed.
    predictions : tuple of WindowPrediction
        Persisted classifier outputs without ground-truth labels.
    window_identities : tuple of WindowIdentity
        Persisted acquisition/window/model identities aligned to predictions.
    trend_evidence : PersistedTrendEvidence
        Previously persisted immutable V6 output. It is never recomputed.
    pipeline_identity : PipelineIdentity
        Persisted assignment, partition, model, representation, and reference
        identities used by the evidence-only adapter.
    risk_configuration : RiskConfiguration
        Predeclared evidence-only risk normalization and weights.
    aggregation_strategy : AggregationStrategy
        Label-free acquisition aggregation strategy.
    trend_model_id, trend_model_version : str
        Identity of the producer recorded by persisted TrendEvidence metadata.
    bundle_configuration : dict of str to Any
        Generation configuration persisted in the final bundle.
    explanation_references : tuple of str
        Optional references to already persisted explanations.

    Raises
    ------
    EvidenceValidationError
        If classifier outputs are missing, window counts differ, or duplicate
        window identities are supplied.
    """

    scenario_id: str
    predictions: tuple[WindowPrediction, ...]
    window_identities: tuple[WindowIdentity, ...]
    trend_evidence: PersistedTrendEvidence
    pipeline_identity: PipelineIdentity
    risk_configuration: RiskConfiguration
    aggregation_strategy: AggregationStrategy
    trend_model_id: str
    trend_model_version: str
    bundle_configuration: dict[str, Any]
    explanation_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.predictions:
            raise EvidenceValidationError(
                "EVIDENCE_CLASSIFIER_OUTPUT_MISSING",
                "at least one persisted WindowPrediction is required",
            )
        if len(self.predictions) != len(self.window_identities):
            raise EvidenceValidationError(
                "EVIDENCE_CLASSIFIER_OUTPUT_MISSING",
                "predictions and window identities must have equal counts",
            )
        window_ids = tuple(item.window_id for item in self.window_identities)
        if len(window_ids) != len(set(window_ids)):
            raise EvidenceValidationError(
                "EVIDENCE_DUPLICATE_WINDOW", "window_id values must be unique"
            )


class EvidenceBundleGenerator:
    """Persist validated acquisition bundles as JSON and Parquet.

    Parameters
    ----------
    output_directory : Path
        Destination directory. Existing identical outputs are accepted;
        conflicting outputs fail closed.

    Side Effects
    ------------
    Creates the output directory and writes JSON/Parquet bundle files. It does
    not execute any diagnostic, agent, safety, transition, or scenario code.
    """

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory

    def generate(self, bundles: tuple[PersistedEvidenceBundle, ...]) -> tuple[BundleArtifact, ...]:
        """Persist unique acquisition bundles in deterministic name order.

        Parameters
        ----------
        bundles : tuple of PersistedEvidenceBundle
            Complete bundles constructed only from persisted evidence.

        Returns
        -------
        artifacts : tuple of BundleArtifact
            File paths, checksums, sizes, and scientific identities.

        Raises
        ------
        EvidenceValidationError
            If assignments/acquisitions are duplicated or output conflicts.
        """

        _reject_duplicate_bundle_identities(bundles)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return tuple(
            self._persist(bundle) for bundle in sorted(bundles, key=lambda x: x.filename_stem)
        )

    def build(self, inputs: PersistedAcquisitionInputs) -> PersistedEvidenceBundle:
        """Transform persisted inputs into a complete bundle without inference.

        Parameters
        ----------
        inputs : PersistedAcquisitionInputs
            Persisted predictions, identities, and immutable TrendEvidence.

        Returns
        -------
        bundle : PersistedEvidenceBundle
            Complete validated Window/Acquisition/Trend/Risk/Safety evidence.

        Raises
        ------
        EvidenceValidationError
            If inputs are mixed, incomplete, inconsistent, or unprovenanced.

        Side Effects
        ------------
        None. Persistence occurs only when :meth:`generate` is called.
        """

        pipeline = DiagnosticEvidencePipeline(inputs.pipeline_identity, inputs.risk_configuration)
        windows = tuple(
            pipeline.make_window_evidence(prediction, identity)
            for prediction, identity in zip(
                inputs.predictions, inputs.window_identities, strict=True
            )
        )
        acquisition = pipeline.aggregate(windows, inputs.aggregation_strategy)
        risk = pipeline.make_risk_evidence(acquisition, inputs.trend_evidence, windows)
        safety = pipeline.make_safety_evidence(acquisition, risk)
        evidence = DiagnosticEvidenceBundle(
            window_evidence=windows,
            acquisition_evidence=acquisition,
            trend_evidence=inputs.trend_evidence,
            risk_evidence=risk,
            safety_evidence=safety,
            explanation_references=inputs.explanation_references,
        )
        return build_persisted_bundle(
            scenario_id=inputs.scenario_id,
            evidence=evidence,
            trend_model_id=inputs.trend_model_id,
            trend_model_version=inputs.trend_model_version,
            configuration=inputs.bundle_configuration,
            creator_component="EvidenceBundleGenerator/1.0.0",
        )

    def _persist(self, bundle: PersistedEvidenceBundle) -> BundleArtifact:
        json_path = self.output_directory / f"{bundle.filename_stem}.json"
        parquet_path = self.output_directory / f"{bundle.filename_stem}.parquet"
        json_bytes = (_canonical_json(bundle.to_dict()) + "\n").encode("utf-8")
        table = pa.Table.from_pylist(
            [
                {
                    "schema_version": bundle.SCHEMA_VERSION,
                    "bundle_hash": bundle.bundle_hash,
                    "assignment_id": bundle.metadata.assignment_id,
                    "partition": bundle.metadata.partition,
                    "acquisition_id": bundle.metadata.acquisition_id,
                    "representation_id": bundle.metadata.representation_id,
                    "classifier_id": bundle.metadata.classifier_id,
                    "trend_model_id": bundle.metadata.trend_model_id,
                    "healthy_reference_hash": bundle.metadata.healthy_reference_hash,
                    "bundle_json": _canonical_json(bundle.to_dict()),
                }
            ]
        )
        _write_json_idempotently(json_path, json_bytes)
        _write_parquet_idempotently(parquet_path, table, bundle.bundle_hash)
        return BundleArtifact(
            bundle_hash=bundle.bundle_hash,
            json_path=json_path,
            json_sha256=_file_hash(json_path),
            parquet_path=parquet_path,
            parquet_sha256=_file_hash(parquet_path),
            size_bytes=json_path.stat().st_size + parquet_path.stat().st_size,
            metadata=bundle.metadata,
        )


def build_persisted_bundle(
    *,
    scenario_id: str,
    evidence: DiagnosticEvidenceBundle,
    trend_model_id: str,
    trend_model_version: str,
    configuration: dict[str, Any],
    creator_component: str,
) -> PersistedEvidenceBundle:
    """Construct a validated bundle from an already persisted evidence chain.

    Parameters
    ----------
    scenario_id : str
        Natural scenario identity; no scenario is executed.
    evidence : DiagnosticEvidenceBundle
        Previously produced immutable evidence chain.
    trend_model_id, trend_model_version : str
        Persisted V6 producer identity and version.
    configuration : dict of str to Any
        Bundle-generation configuration without diagnostic facts.
    creator_component : str
        Versioned identity of the evidence-only generator.

    Returns
    -------
    bundle : PersistedEvidenceBundle
        Validated replayable acquisition bundle.

    Raises
    ------
    EvidenceValidationError
        If the evidence lacks required hashes or contains mixed identities.
    """

    first = evidence.window_evidence[0]
    trend = evidence.trend_evidence
    healthy_hash = trend.provenance.healthy_reference_hash
    if healthy_hash is None:
        raise EvidenceValidationError(
            "EVIDENCE_HEALTHY_REFERENCE_MISSING", "bundle requires a HealthyReference hash"
        )
    metadata = BundleMetadata(
        scenario_id=scenario_id,
        assignment_id=evidence.acquisition_evidence.assignment_id,
        partition=evidence.acquisition_evidence.partition,
        acquisition_id=evidence.acquisition_evidence.acquisition_id,
        representation_id=first.representation_id,
        representation_version=first.representation_version,
        representation_hash=first.provenance.representation_hash,
        classifier_id=first.classifier_id,
        classifier_version=first.classifier_version,
        classifier_hash=first.model_hash,
        trend_model_id=trend_model_id,
        trend_model_version=trend_model_version,
        trend_evidence_hash=trend.evidence_hash,
        healthy_reference_hash=healthy_hash,
    )
    configuration_json = _canonical_json(configuration)
    provenance = EvidenceProvenance(
        schema_version=EvidenceProvenance.SCHEMA_VERSION,
        creation_timestamp=trend.provenance.creation_timestamp,
        creator_component=creator_component,
        input_hashes=_bundle_input_hashes(evidence),
        configuration_hash=stable_hash(configuration),
        assignment_id=metadata.assignment_id,
        partition=metadata.partition,
        model_hash=metadata.classifier_hash,
        representation_hash=metadata.representation_hash,
        healthy_reference_hash=metadata.healthy_reference_hash,
    )
    return PersistedEvidenceBundle(metadata, evidence, configuration_json, provenance)


def checksum_lines(artifacts: tuple[BundleArtifact, ...], root: Path) -> tuple[str, ...]:
    """Return sorted SHA-256 checksum lines for JSON and Parquet bundles."""

    lines: list[str] = []
    for artifact in artifacts:
        for digest, path in (
            (artifact.json_sha256, artifact.json_path),
            (artifact.parquet_sha256, artifact.parquet_path),
        ):
            try:
                relative = path.relative_to(root)
            except ValueError as exc:
                raise EvidenceValidationError(
                    "EVIDENCE_INVALID_ARTIFACT_PATH", f"bundle is outside repository root: {path}"
                ) from exc
            lines.append(f"{digest}  {relative.as_posix()}")
    return tuple(sorted(lines))


def _bundle_input_hashes(evidence: DiagnosticEvidenceBundle) -> tuple[str, ...]:
    return (
        *(window.evidence_hash for window in evidence.window_evidence),
        evidence.acquisition_evidence.evidence_hash,
        evidence.trend_evidence.evidence_hash,
        evidence.risk_evidence.evidence_hash,
        stable_hash(evidence.safety_evidence.to_dict()),
    )


def _configuration(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvidenceValidationError(
            "EVIDENCE_CONFIGURATION_MISMATCH", "configuration_json must be valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise EvidenceValidationError(
            "EVIDENCE_CONFIGURATION_MISMATCH", "configuration_json must contain an object"
        )
    return parsed


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _safe_component(value: str) -> str:
    cleaned = _SAFE_NAME.sub("-", value).strip("-.")
    if not cleaned:
        raise EvidenceValidationError(
            "EVIDENCE_INVALID_ARTIFACT_PATH", "bundle filename component is empty"
        )
    return cleaned


def _reject_duplicate_bundle_identities(bundles: tuple[PersistedEvidenceBundle, ...]) -> None:
    identities = [(item.metadata.assignment_id, item.metadata.acquisition_id) for item in bundles]
    if len(identities) != len(set(identities)):
        raise EvidenceValidationError(
            "EVIDENCE_DUPLICATE_ACQUISITION",
            "each assignment/acquisition may appear in only one bundle",
        )


def _write_json_idempotently(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise EvidenceValidationError(
                "EVIDENCE_ARTIFACT_CONFLICT", f"existing JSON bundle differs: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_parquet_idempotently(path: Path, table: pa.Table, bundle_hash: str) -> None:
    if path.exists():
        existing = PersistedEvidenceBundle.load_parquet(path)
        if existing.bundle_hash != bundle_hash:
            raise EvidenceValidationError(
                "EVIDENCE_ARTIFACT_CONFLICT", f"existing Parquet bundle differs: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", write_statistics=True)
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
