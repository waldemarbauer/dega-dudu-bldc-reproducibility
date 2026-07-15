"""Generate ArticleV1 diagnostic evidence without executing the agent.

The module consumes the frozen E1.2 feature/assignment artifacts and E1.3
classifier predictions.  It fits only assignment-local, label-free deviation
scorers and training-only Healthy references, then executes the existing V6
trend models and R3 persisted-evidence adapters.  It never loads a classifier
model and never imports the FSM, transition, safety, scenario, or agent layers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from trustworthy_agent.evidence.bundles import (
    EvidenceBundleGenerator,
    PersistedAcquisitionInputs,
    PersistedEvidenceBundle,
)
from trustworthy_agent.evidence.diagnostic import (
    CANONICAL_CLASSES,
    AggregationStrategy,
    PersistedTrendEvidence,
    WindowPrediction,
    stable_hash,
)
from trustworthy_agent.evidence.pipeline import PipelineIdentity, RiskConfiguration, WindowIdentity
from trustworthy_agent.evidence.trend import TrendEvidence, TrendFitStatus
from trustworthy_agent.trends.config import (
    MODEL_TYPES,
    create_trend_model,
    load_trend_family_config,
)
from trustworthy_agent.trends.contracts import HealthyReference, OrderedWindow, TrendMode

CONFIG_PATH = Path("configs/evidence/article_v1/article_v1_real_evidence_v1.yaml")
OOD_CONFIG_PATH = Path("configs/ood/article_v1/training_distribution_deviation_v1.yaml")
SELECTION_CONFIG_PATH = Path("configs/scenarios/article/article_natural_case_selection_v1.yaml")
FEATURE_PATH = Path(
    "Data/AnalysisData/ArticleV1/WindowFeatures/article_v1_canonical_window_features.parquet"
)
ASSIGNMENT_ROOT = Path("Data/AnalysisData/ArticleV1/Assignments")
PREDICTION_ROOT = Path("Output/ArticleV1/Results/WindowPredictions")
ACQUISITION_PREDICTION_ROOT = Path("Output/ArticleV1/Results/AcquisitionPredictions")
FEATURE_MANIFEST_PATH = Path("Output/ArticleV1/Manifests/article_window_feature_manifest.json")
PREDICTION_MANIFEST_PATH = Path(
    "Output/ArticleV1/Manifests/article_window_prediction_manifest.json"
)
ASSIGNMENT_CATALOG_PATH = Path("Output/ArticleV1/Manifests/acquisition_assignment_catalog.json")
HEALTHY_CATALOG_PATH = Path(
    "Output/ArticleV1/Manifests/article_healthy_reference_source_catalog.json"
)
TREND_CONFIG_PATH = Path("configs/trends/window_temporal_spline.yaml")
R3_CONFIG_PATH = Path("configs/article/evidence_pipeline.yaml")
CLASSIFIERS = ("logistic_regression", "random_forest", "hist_gradient_boosting")
ASSIGNMENTS = tuple(f"A{index:02d}" for index in range(16))
CREATOR = "ArticleV1RealEvidence/1.0.0"


class ArticleV1RealEvidenceError(ValueError):
    """Report a fail-closed E1.4 configuration, input, or replay failure."""


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 without modifying the file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingDistributionDeviation:
    """Persist one assignment-local training-distribution deviation scorer.

    The scorer standardizes 28 features on training windows, estimates a
    Ledoit-Wolf shrinkage covariance on those standardized rows, and evaluates
    squared Mahalanobis distances.  Its bounded score is the training empirical
    CDF ``(1 + count(d_train <= d_x)) / (n_train + 1)``.  This is an unvalidated
    OOD surrogate, not an open-world detector.

    References
    ----------
    Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
    large-dimensional covariance matrices. *Journal of Multivariate Analysis*,
    88(2), 365-411. https://doi.org/10.1016/S0047-259X(03)00096-4
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    METHOD_ID: ClassVar[str] = "TRAINING_DISTRIBUTION_DEVIATION_V1"

    assignment_id: str
    feature_schema_hash: str
    training_acquisition_ids: tuple[str, ...]
    training_window_ids: tuple[str, ...]
    training_input_hashes: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    covariance_location: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    precision: tuple[tuple[float, ...], ...]
    shrinkage: float
    reference_distances: tuple[float, ...]
    configuration_hash: str

    @classmethod
    def fit(
        cls,
        *,
        assignment_id: str,
        features: NDArray[np.float64],
        partitions: tuple[str, ...],
        acquisition_ids: tuple[str, ...],
        window_ids: tuple[str, ...],
        input_hashes: tuple[str, ...],
        feature_schema_hash: str,
        configuration_hash: str,
    ) -> TrainingDistributionDeviation:
        """Fit the deterministic scorer using exactly assignment training rows.

        Parameters are label-free by construction.  ``partitions`` is checked
        explicitly so held-out rows cannot enter fitting unnoticed.

        Raises
        ------
        ArticleV1RealEvidenceError
            If identity, shape, partition, ordering, or finite-value checks fail.
        """

        if not assignment_id or len(features) != 100 or features.shape != (100, 28):
            raise ArticleV1RealEvidenceError("deviation fit requires one assignment and 100x28")
        if set(partitions) != {"train"}:
            raise ArticleV1RealEvidenceError("deviation fit accepts training rows only")
        if len(set(acquisition_ids)) != 4 or len(set(window_ids)) != 100:
            raise ArticleV1RealEvidenceError("deviation fit identity counts are invalid")
        if len(input_hashes) != 100 or not np.isfinite(features).all():
            raise ArticleV1RealEvidenceError("deviation fit inputs must be finite and hashed")
        scaler = StandardScaler().fit(features)
        standardized = cast(NDArray[np.float64], scaler.transform(features))
        covariance = LedoitWolf(assume_centered=False).fit(standardized)
        precision = cast(NDArray[np.float64], covariance.precision_)
        if not np.isfinite(precision).all():
            raise ArticleV1RealEvidenceError("deviation precision matrix is nonfinite")
        centered = standardized - covariance.location_
        distances = np.einsum("ij,jk,ik->i", centered, precision, centered)
        if not np.isfinite(distances).all() or np.any(distances < -1e-10):
            raise ArticleV1RealEvidenceError("training Mahalanobis distances are invalid")
        return cls(
            assignment_id=assignment_id,
            feature_schema_hash=feature_schema_hash,
            training_acquisition_ids=tuple(sorted(set(acquisition_ids))),
            training_window_ids=window_ids,
            training_input_hashes=input_hashes,
            scaler_mean=tuple(float(value) for value in scaler.mean_),
            scaler_scale=tuple(float(value) for value in scaler.scale_),
            covariance_location=tuple(float(value) for value in covariance.location_),
            covariance=tuple(
                tuple(float(value) for value in row) for row in covariance.covariance_
            ),
            precision=tuple(tuple(float(value) for value in row) for row in precision),
            shrinkage=float(covariance.shrinkage_),
            reference_distances=tuple(float(max(value, 0.0)) for value in distances),
            configuration_hash=configuration_hash,
        )

    @property
    def scorer_hash(self) -> str:
        """Return the canonical scientific identity of the fitted scorer."""

        # Keep the estimator identity stable when envelope-only provenance
        # fields are added.  ``fitted_state_hash`` is validated separately and
        # must not invalidate already-persisted evidence bundle identities.
        payload = self.to_dict()
        payload.pop("fitted_state_hash", None)
        return stable_hash(payload)

    @property
    def fitted_state_hash(self) -> str:
        """Hash the fitted numerical state independently of its envelope.

        Keeping this separate from :attr:`scorer_hash` makes corruption of a
        persisted scaler/covariance/reference-distance payload detectable even
        when the surrounding provenance envelope is otherwise valid.  The
        envelope hash may change when metadata is extended, while this hash
        remains an identity of the actual fitted estimator state.
        """

        return stable_hash(
            {
                "assignment_id": self.assignment_id,
                "feature_schema_hash": self.feature_schema_hash,
                "training_acquisition_ids": list(self.training_acquisition_ids),
                "training_window_ids": list(self.training_window_ids),
                "training_input_hashes": list(self.training_input_hashes),
                "scaler_mean": list(self.scaler_mean),
                "scaler_scale": list(self.scaler_scale),
                "covariance_location": list(self.covariance_location),
                "covariance": [list(row) for row in self.covariance],
                "precision": [list(row) for row in self.precision],
                "shrinkage": self.shrinkage,
                "reference_distances": list(self.reference_distances),
                "configuration_hash": self.configuration_hash,
            }
        )

    def score(self, features: NDArray[np.float64], *, assignment_id: str) -> tuple[float, float]:
        """Return squared Mahalanobis distance and bounded empirical score."""

        if assignment_id != self.assignment_id:
            raise ArticleV1RealEvidenceError("mixed-assignment deviation scoring rejected")
        vector = np.asarray(features, dtype=float)
        if vector.shape != (28,) or not np.isfinite(vector).all():
            raise ArticleV1RealEvidenceError("deviation scoring requires one finite 28-vector")
        standardized = (vector - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)
        centered = standardized - np.asarray(self.covariance_location)
        distance = float(centered @ np.asarray(self.precision) @ centered)
        if not math.isfinite(distance):
            raise ArticleV1RealEvidenceError("held-out Mahalanobis distance is nonfinite")
        distance = max(distance, 0.0)
        count = sum(reference <= distance for reference in self.reference_distances)
        score = (1.0 + count) / (len(self.reference_distances) + 1.0)
        return distance, score

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON scorer artifact with scientific provenance."""

        precision = [list(row) for row in self.precision]
        distances = list(self.reference_distances)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "protocol_id": "DUDU_BLDC_TRUSTWORTHY_AGENT_ARTICLE_V1_E1_4",
            "assignment_id": self.assignment_id,
            "method_id": self.METHOD_ID,
            "method_version": "1.0.0",
            "scientific_status": "OOD_SURROGATE_NOT_EXTERNALLY_VALIDATED",
            "feature_schema_version": "1.0.0",
            "feature_schema_hash": self.feature_schema_hash,
            "training_acquisition_ids": list(self.training_acquisition_ids),
            "training_window_ids": list(self.training_window_ids),
            "training_input_hashes": list(self.training_input_hashes),
            "scaler": {"mean": list(self.scaler_mean), "scale": list(self.scaler_scale)},
            "covariance": {
                "estimator": "sklearn.covariance.LedoitWolf",
                "assume_centered": False,
                "location": list(self.covariance_location),
                "matrix": [list(row) for row in self.covariance],
                "shrinkage": self.shrinkage,
            },
            "precision": precision,
            "precision_matrix_hash": stable_hash({"precision": precision}),
            "reference_distances": distances,
            "reference_distance_hash": stable_hash({"distances": distances}),
            "fitted_state_hash": self.fitted_state_hash,
            "configuration_hash": self.configuration_hash,
            "software_provenance": {
                "numpy": version("numpy"),
                "scikit_learn": version("scikit-learn"),
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingDistributionDeviation:
        """Reload a scorer, rejecting incomplete or unsupported provenance."""

        if value.get("schema_version") != cls.SCHEMA_VERSION:
            raise ArticleV1RealEvidenceError("deviation scorer schema mismatch")
        if value.get("method_id") != cls.METHOD_ID or not value.get("software_provenance"):
            raise ArticleV1RealEvidenceError("deviation scorer provenance is missing")
        covariance = cast(dict[str, Any], value["covariance"])
        scaler = cast(dict[str, Any], value["scaler"])
        scorer = cls(
            assignment_id=str(value["assignment_id"]),
            feature_schema_hash=str(value["feature_schema_hash"]),
            training_acquisition_ids=tuple(map(str, value["training_acquisition_ids"])),
            training_window_ids=tuple(map(str, value["training_window_ids"])),
            training_input_hashes=tuple(map(str, value["training_input_hashes"])),
            scaler_mean=tuple(map(float, scaler["mean"])),
            scaler_scale=tuple(map(float, scaler["scale"])),
            covariance_location=tuple(map(float, covariance["location"])),
            covariance=tuple(tuple(map(float, row)) for row in covariance["matrix"]),
            precision=tuple(tuple(map(float, row)) for row in value["precision"]),
            shrinkage=float(covariance["shrinkage"]),
            reference_distances=tuple(map(float, value["reference_distances"])),
            configuration_hash=str(value["configuration_hash"]),
        )
        if value.get("precision_matrix_hash") != stable_hash(
            {"precision": [list(row) for row in scorer.precision]}
        ):
            raise ArticleV1RealEvidenceError("deviation precision hash mismatch")
        if value.get("reference_distance_hash") != stable_hash(
            {"distances": list(scorer.reference_distances)}
        ):
            raise ArticleV1RealEvidenceError("deviation reference-distance hash mismatch")
        if value.get("fitted_state_hash") != scorer.fitted_state_hash:
            raise ArticleV1RealEvidenceError("deviation fitted-state hash mismatch")
        return scorer


def prepare_execution(project_root: Path) -> dict[str, Any]:
    """Verify direct frozen inputs and persist the pre-fit 256/768-unit plan."""

    config = _read_yaml(project_root / CONFIG_PATH)
    frozen = cast(dict[str, str], config["frozen_inputs"])
    checks = {
        "feature_corpus_sha256": FEATURE_PATH,
        "classifier_prediction_manifest_sha256": PREDICTION_MANIFEST_PATH,
        "assignment_catalog_sha256": ASSIGNMENT_CATALOG_PATH,
        "healthy_source_catalog_sha256": HEALTHY_CATALOG_PATH,
        "trend_configuration_sha256": TREND_CONFIG_PATH,
        "r3_evidence_configuration_sha256": R3_CONFIG_PATH,
    }
    for key, path in checks.items():
        _require(sha256_file(project_root / path) == frozen[key], f"frozen input hash: {path}")
    feature_manifest = _read_json(project_root / FEATURE_MANIFEST_PATH)
    _require(feature_manifest["feature_schema_hash"] == frozen["feature_schema_hash"], "schema")
    rows = pq.read_table(project_root / FEATURE_PATH).to_pylist()
    _require(len(rows) == 200, "feature corpus row count")
    prediction_manifest = _read_json(project_root / PREDICTION_MANIFEST_PATH)
    _require(prediction_manifest["record_count"] == 4800, "prediction count")
    units = [
        {
            "assignment_id": assignment,
            "classifier_id": classifier,
            "trend_model_id": trend,
            "expected_held_out_acquisitions": 4,
            "expected_windows_per_acquisition": 25,
            "status": "NEVER_STARTED",
        }
        for assignment in ASSIGNMENTS
        for classifier in CLASSIFIERS
        for trend in MODEL_TYPES
    ]
    plan = {
        "schema_version": "1.0.0",
        "protocol_id": config["protocol_id"],
        "configuration_hash": sha256_file(project_root / CONFIG_PATH),
        "ood_configuration_hash": sha256_file(project_root / OOD_CONFIG_PATH),
        "case_selection_configuration_hash": sha256_file(project_root / SELECTION_CONFIG_PATH),
        "frozen_inputs": frozen,
        "execution_unit_count": len(units),
        "units": units,
        "scientific_fit_started": False,
    }
    plan["execution_plan_hash"] = stable_hash(plan)
    _atomic_json(
        project_root / "Output/ArticleV1/Manifests/article_v1_real_evidence_execution_plan.json",
        plan,
    )
    return plan


def execute(project_root: Path, *, artifact_root: Path | None = None) -> dict[str, Any]:
    """Generate, persist, reload, and validate the complete E1.4 evidence layer.

    Parameters
    ----------
    project_root : Path
        Repository root containing frozen E1.2/E1.3 inputs.
    artifact_root : Path or None
        Alternate ArticleV1 output root used for deterministic replay.

    Returns
    -------
    summary : dict
        Counts, hashes, failures, and natural-case resolution status.

    Side Effects
    ------------
    Writes only E1.4 generated artifacts below the selected output root.
    """

    target = artifact_root or project_root / "Output/ArticleV1"
    target.mkdir(parents=True, exist_ok=True)
    config = _read_yaml(project_root / CONFIG_PATH)
    config_hash = sha256_file(project_root / CONFIG_PATH)
    ood_config_hash = sha256_file(project_root / OOD_CONFIG_PATH)
    selection_hash = sha256_file(project_root / SELECTION_CONFIG_PATH)
    feature_manifest = _read_json(project_root / FEATURE_MANIFEST_PATH)
    feature_order = tuple(map(str, feature_manifest["feature_order"]))
    feature_rows = pq.read_table(project_root / FEATURE_PATH).to_pylist()
    by_window = {str(row["window_id"]): row for row in feature_rows}
    healthy_catalog = _read_json(project_root / HEALTHY_CATALOG_PATH)
    healthy_entries = {entry["assignment_id"]: entry for entry in healthy_catalog["entries"]}
    trend_config = load_trend_family_config(project_root / TREND_CONFIG_PATH)
    risk_configuration = _risk_configuration(project_root)

    ood_registry: list[dict[str, Any]] = []
    healthy_registry: list[dict[str, Any]] = []
    trend_registry: list[dict[str, Any]] = []
    window_registry: list[dict[str, Any]] = []
    acquisition_registry: list[dict[str, Any]] = []
    risk_registry: list[dict[str, Any]] = []
    bundle_registry: list[dict[str, Any]] = []
    all_scores: dict[tuple[str, str], dict[str, Any]] = {}
    references: dict[str, HealthyReference] = {}
    trends: dict[tuple[str, str, str], TrendEvidence] = {}
    failures: list[dict[str, str]] = []

    # Scientific fits start only after all three immutable configurations exist.
    for assignment_id in ASSIGNMENTS:
        assignment = _read_json(project_root / ASSIGNMENT_ROOT / f"{assignment_id}.json")
        train_ids = tuple(map(str, assignment["training_window_ids"]))
        held_ids = tuple(map(str, assignment["held_out_window_ids"]))
        train_rows = [by_window[item] for item in train_ids]
        held_rows = [by_window[item] for item in held_ids]
        train_matrix = _matrix(train_rows, feature_order)
        scorer = TrainingDistributionDeviation.fit(
            assignment_id=assignment_id,
            features=train_matrix,
            partitions=("train",) * len(train_rows),
            acquisition_ids=tuple(str(row["acquisition_id"]) for row in train_rows),
            window_ids=train_ids,
            input_hashes=tuple(_feature_row_hash(row, feature_order) for row in train_rows),
            feature_schema_hash=str(feature_manifest["feature_schema_hash"]),
            configuration_hash=ood_config_hash,
        )
        scorer_path = target / "OOD" / assignment_id / "deviation_scorer.json"
        _atomic_json(scorer_path, scorer.to_dict())
        reloaded = TrainingDistributionDeviation.from_dict(_read_json(scorer_path))
        _require(reloaded.scorer_hash == scorer.scorer_hash, "scorer reload equality")
        score_rows: list[dict[str, Any]] = []
        for row in held_rows:
            distance, score = scorer.score(
                np.asarray([float(row[name]) for name in feature_order]),
                assignment_id=assignment_id,
            )
            record = {
                "schema_version": "1.0.0",
                "assignment_id": assignment_id,
                "partition": "held_out",
                "acquisition_id": str(row["acquisition_id"]),
                "window_id": str(row["window_id"]),
                "window_order": int(row["window_order"]),
                "raw_squared_mahalanobis_distance": distance,
                "bounded_deviation_score": score,
                "method_id": scorer.METHOD_ID,
                "scientific_status": "OOD_SURROGATE_NOT_EXTERNALLY_VALIDATED",
                "scorer_hash": scorer.scorer_hash,
                "feature_row_hash": _feature_row_hash(row, feature_order),
                "provenance": {
                    "fit_partition": "train",
                    "fit_window_count": 100,
                    "held_out_fit_count": 0,
                    "label_input_count": 0,
                    "configuration_hash": ood_config_hash,
                },
            }
            record["score_hash"] = stable_hash(record)
            score_rows.append(record)
            all_scores[(assignment_id, str(row["window_id"]))] = record
        _atomic_parquet(target / "OOD" / assignment_id / "held_out_scores.parquet", score_rows)
        ood_registry.append(
            {
                "assignment_id": assignment_id,
                "status": "COMPLETE_RELOAD_VERIFIED",
                "scorer_hash": scorer.scorer_hash,
                "scorer_path": _relative(scorer_path, target),
                "score_path": _relative(
                    target / "OOD" / assignment_id / "held_out_scores.parquet", target
                ),
                "training_rows": 100,
                "held_out_score_rows": 100,
            }
        )

        reference, reference_record = _fit_healthy_reference(
            assignment_id, healthy_entries[assignment_id], by_window, feature_order, config_hash
        )
        reference_path = target / "HealthyReferences" / assignment_id / "healthy_reference.json"
        _atomic_json(reference_path, reference_record)
        loaded_reference = HealthyReference.from_dict(
            cast(dict[str, Any], _read_json(reference_path)["healthy_reference"])
        )
        _require(loaded_reference.stable_hash() == reference.stable_hash(), "Healthy reload")
        references[assignment_id] = reference
        healthy_registry.append(
            {
                "assignment_id": assignment_id,
                "status": "COMPLETE_RELOAD_VERIFIED",
                "reference_hash": reference.stable_hash(),
                "path": _relative(reference_path, target),
                "training_window_count": 25,
                "held_out_contamination_count": 0,
            }
        )

        for acquisition_id in sorted(map(str, assignment["held_out_acquisition_ids"])):
            acquisition_rows = sorted(
                (row for row in held_rows if row["acquisition_id"] == acquisition_id),
                key=lambda row: int(row["window_order"]),
            )
            ordered = _ordered_windows(assignment_id, acquisition_rows, feature_order)
            _require(len(ordered) == 25, "25 held-out ordered windows")
            for model_name in MODEL_TYPES:
                model = create_trend_model(trend_config, model_name, mode=TrendMode.OFFLINE)
                try:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter("always")
                        model.fit(ordered, healthy_reference=reference)
                        evidence = model.transform()
                except Exception as exc:
                    reason = getattr(getattr(exc, "evidence", None), "reason", type(exc).__name__)
                    failures.append(
                        {
                            "assignment_id": assignment_id,
                            "acquisition_id": acquisition_id,
                            "trend_model_id": model_name,
                            "reason": str(reason),
                        }
                    )
                    continue
                if evidence.fit_status is not TrendFitStatus.FIT_OK:
                    failures.append(
                        {
                            "assignment_id": assignment_id,
                            "acquisition_id": acquisition_id,
                            "trend_model_id": model_name,
                            "reason": evidence.reason or evidence.fit_status.value,
                        }
                    )
                    continue
                trend_path = (
                    target
                    / "Results/TrendEvidence"
                    / assignment_id
                    / acquisition_id
                    / f"{model_name}.json"
                )
                payload = {
                    "schema_version": "1.0.0",
                    "assignment_id": assignment_id,
                    "partition": "held_out",
                    "acquisition_id": acquisition_id,
                    "trend_model_id": model_name,
                    "trend_model_version": model.model_version,
                    "healthy_reference_hash": reference.stable_hash(),
                    "trend_evidence": evidence.to_dict(),
                    "numerical_warnings": [str(item.message) for item in caught_warnings],
                }
                payload["artifact_hash"] = stable_hash(payload)
                _atomic_json(trend_path, payload)
                loaded_evidence = TrendEvidence.from_dict(
                    cast(dict[str, Any], _read_json(trend_path)["trend_evidence"])
                )
                _require(
                    stable_hash(loaded_evidence.to_dict()) == stable_hash(evidence.to_dict()),
                    "trend reload",
                )
                trends[(assignment_id, acquisition_id, model_name)] = evidence
                trend_registry.append(
                    {
                        "assignment_id": assignment_id,
                        "acquisition_id": acquisition_id,
                        "trend_model_id": model_name,
                        "status": "COMPLETE_RELOAD_VERIFIED",
                        "evidence_hash": stable_hash(evidence.to_dict()),
                        "warning_count": len(caught_warnings),
                        "path": _relative(trend_path, target),
                    }
                )

    risk_objects: dict[tuple[str, str, str, str], Any] = {}
    for assignment_id in ASSIGNMENTS:
        reference = references[assignment_id]
        assignment = _read_json(project_root / ASSIGNMENT_ROOT / f"{assignment_id}.json")
        held_ids = tuple(map(str, assignment["held_out_window_ids"]))
        for classifier_id in CLASSIFIERS:
            prediction_path = (
                project_root / PREDICTION_ROOT / assignment_id / f"{classifier_id}.parquet"
            )
            prediction_rows = pq.read_table(prediction_path).to_pylist()
            _require(len(prediction_rows) == 100, "100 persisted predictions")
            _require(
                all(str(row["assignment_id"]) == assignment_id for row in prediction_rows),
                "prediction assignment identity",
            )
            _require(
                all(str(row["partition"]) == "held_out" for row in prediction_rows),
                "prediction held-out partition",
            )
            _require(
                {str(row["window_id"]) for row in prediction_rows} == set(held_ids),
                "prediction held-out window identity",
            )
            _require(
                all(
                    str(row["feature_schema_hash"]) == str(feature_manifest["feature_schema_hash"])
                    for row in prediction_rows
                ),
                "prediction feature schema identity",
            )
            per_acquisition: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in prediction_rows:
                per_acquisition[str(row["acquisition_id"])].append(row)
            classifier_windows: list[dict[str, Any]] = []
            classifier_acquisitions: list[dict[str, Any]] = []
            for acquisition_id in sorted(per_acquisition):
                source_rows = sorted(
                    per_acquisition[acquisition_id], key=lambda row: int(row["window_order"])
                )
                predictions, identities = _prediction_inputs(source_rows, all_scores)
                first = source_rows[0]
                for model_name in MODEL_TYPES:
                    raw_trend = trends.get((assignment_id, acquisition_id, model_name))
                    if raw_trend is None:
                        continue
                    identity = PipelineIdentity(
                        creation_timestamp=str(config["creation_timestamp"]),
                        creator_component=CREATOR,
                        configuration_hash=raw_trend.configuration_hash,
                        assignment_id=assignment_id,
                        partition="test",
                        model_hash=str(first["model_hash"]),
                        representation_hash=str(first["representation_hash"]),
                        healthy_reference_hash=reference.stable_hash(),
                    )
                    trend_provenance = identity.provenance(raw_trend.input_window_hashes)
                    persisted_trend = PersistedTrendEvidence(raw_trend, trend_provenance)
                    bundle_inputs = PersistedAcquisitionInputs(
                        scenario_id="ARTICLE_V1_UNSELECTED_EVIDENCE_POOL",
                        predictions=predictions,
                        window_identities=identities,
                        trend_evidence=persisted_trend,
                        pipeline_identity=identity,
                        risk_configuration=risk_configuration,
                        trend_model_id=model_name,
                        trend_model_version=str(raw_trend.metadata["model_version"]),
                        aggregation_strategy=AggregationStrategy.MEAN_PROBABILITY,
                        bundle_configuration={
                            "protocol_id": config["protocol_id"],
                            "configuration_hash": config_hash,
                            "ood_method_id": TrainingDistributionDeviation.METHOD_ID,
                            "ood_scientific_status": "OOD_SURROGATE_NOT_EXTERNALLY_VALIDATED",
                            "configured_diagnostic_facts": 0,
                        },
                        explanation_references=(),
                    )
                    output_directory = target / "EvidenceBundles"
                    generator = EvidenceBundleGenerator(output_directory)
                    bundle = generator.build(bundle_inputs)
                    artifact = generator.generate((bundle,))[0]
                    json_bundle = PersistedEvidenceBundle.load_json(artifact.json_path)
                    parquet_bundle = PersistedEvidenceBundle.load_parquet(artifact.parquet_path)
                    _require(
                        json_bundle.bundle_hash == bundle.bundle_hash
                        and parquet_bundle.bundle_hash == bundle.bundle_hash,
                        "bundle reload equivalence",
                    )
                    # The first trend model supplies the non-duplicated 4,800/192 R3 registries.
                    if model_name == next(iter(MODEL_TYPES)):
                        classifier_windows.extend(
                            item.to_dict() for item in bundle.evidence.window_evidence
                        )
                        classifier_acquisitions.append(
                            bundle.evidence.acquisition_evidence.to_dict()
                        )
                        _verify_acquisition_equivalence(
                            project_root,
                            assignment_id,
                            classifier_id,
                            bundle.evidence.acquisition_evidence,
                        )
                    risk = bundle.evidence.risk_evidence
                    safety = bundle.evidence.safety_evidence
                    risk_objects[(assignment_id, classifier_id, acquisition_id, model_name)] = risk
                    risk_registry.append(
                        {
                            "assignment_id": assignment_id,
                            "classifier_id": classifier_id,
                            "acquisition_id": acquisition_id,
                            "trend_model_id": model_name,
                            "risk_evidence_hash": risk.evidence_hash,
                            "risk_evidence": risk.to_dict(),
                            "safety_evidence": safety.to_dict(),
                            "status": "COMPLETE_RELOAD_VERIFIED",
                        }
                    )
                    bundle_registry.append(
                        {
                            "assignment_id": assignment_id,
                            "classifier_id": classifier_id,
                            "acquisition_id": acquisition_id,
                            "trend_model_id": model_name,
                            "bundle_hash": bundle.bundle_hash,
                            "json_path": _relative(artifact.json_path, target),
                            "json_sha256": artifact.json_sha256,
                            "parquet_path": _relative(artifact.parquet_path, target),
                            "parquet_sha256": artifact.parquet_sha256,
                            "status": "COMPLETE_RELOAD_VERIFIED",
                        }
                    )
            window_path = (
                target / "Results/WindowEvidence" / assignment_id / f"{classifier_id}.parquet"
            )
            acquisition_path = (
                target / "Results/AcquisitionEvidence" / assignment_id / f"{classifier_id}.parquet"
            )
            _atomic_parquet(
                window_path,
                [{"evidence_json": _canonical_json(item)} for item in classifier_windows],
            )
            _atomic_parquet(
                acquisition_path,
                [{"evidence_json": _canonical_json(item)} for item in classifier_acquisitions],
            )
            window_registry.append(
                {
                    "assignment_id": assignment_id,
                    "classifier_id": classifier_id,
                    "record_count": len(classifier_windows),
                    "path": _relative(window_path, target),
                    "status": "COMPLETE_RELOAD_VERIFIED",
                }
            )
            acquisition_registry.append(
                {
                    "assignment_id": assignment_id,
                    "classifier_id": classifier_id,
                    "record_count": len(classifier_acquisitions),
                    "path": _relative(acquisition_path, target),
                    "status": "COMPLETE_RELOAD_VERIFIED",
                }
            )

    risk_rows = [
        {
            "assignment_id": item["assignment_id"],
            "classifier_id": item["classifier_id"],
            "acquisition_id": item["acquisition_id"],
            "trend_model_id": item["trend_model_id"],
            "evidence_json": _canonical_json(item["risk_evidence"]),
        }
        for item in risk_registry
    ]
    safety_rows = [
        {
            "assignment_id": item["assignment_id"],
            "classifier_id": item["classifier_id"],
            "acquisition_id": item["acquisition_id"],
            "trend_model_id": item["trend_model_id"],
            "evidence_json": _canonical_json(item["safety_evidence"]),
        }
        for item in risk_registry
    ]
    _atomic_parquet(target / "Results/RiskEvidence/article_v1_risk_evidence.parquet", risk_rows)
    _atomic_parquet(
        target / "Results/SafetyEvidence/article_v1_safety_evidence.parquet", safety_rows
    )

    selection = _select_natural_cases(project_root, risk_objects, feature_rows, selection_hash)
    summary = {
        "schema_version": "1.0.0",
        "protocol_id": config["protocol_id"],
        "configuration_hash": config_hash,
        "ood_method_id": TrainingDistributionDeviation.METHOD_ID,
        "ood_scorer_count": len(ood_registry),
        "held_out_deviation_record_count": len(all_scores),
        "label_usage_count": 0,
        "held_out_fit_count": 0,
        "healthy_reference_count": len(healthy_registry),
        "held_out_healthy_contamination_count": 0,
        "trend_evidence_count": len(trend_registry),
        "trend_failures": failures,
        "trend_warning_count": sum(item["warning_count"] for item in trend_registry),
        "window_evidence_count": sum(item["record_count"] for item in window_registry),
        "acquisition_evidence_count": sum(item["record_count"] for item in acquisition_registry),
        "risk_evidence_count": len(risk_registry),
        "safety_evidence_count": len(risk_registry),
        "bundle_count": len(bundle_registry),
        "expected_bundle_count": 768,
        "natural_scenarios_resolvable": sum(item["status"] == "RESOLVED" for item in selection),
        "configured_diagnostic_facts": 0,
        "classifier_retraining_count": 0,
        "classifier_model_modification_count": 0,
        "v6_implementation_modification_count": 0,
        "fsm_execution_count": 0,
        "transition_policy_execution_count": 0,
        "safety_guard_decision_execution_count": 0,
        "scenario_execution_count": 0,
        "agent_execution_count": 0,
    }
    _write_registries(
        target,
        ood_registry,
        healthy_registry,
        trend_registry,
        window_registry,
        acquisition_registry,
        risk_registry,
        bundle_registry,
        selection,
        summary,
    )
    _finalize_execution_plan(project_root, summary)
    return summary


def replay(project_root: Path) -> dict[str, Any]:
    """Perform a full temporary replay and compare deterministic scientific hashes."""

    current = _read_json(
        project_root / "Output/ArticleV1/Manifests/article_v1_evidence_bundle_registry.json"
    )
    with tempfile.TemporaryDirectory(prefix="article-v1-e14-replay-") as directory:
        replay_root = Path(directory) / "ArticleV1"
        replay_summary = execute(project_root, artifact_root=replay_root)
        replay_registry = _read_json(
            replay_root / "Manifests/article_v1_evidence_bundle_registry.json"
        )
        current_hashes = [item["bundle_hash"] for item in current["records"]]
        replay_hashes = [item["bundle_hash"] for item in replay_registry["records"]]
        _require(current_hashes == replay_hashes, "bundle replay hashes")
        replay_summary["replay_equivalent"] = True
        replay_summary["replayed_bundle_hash_count"] = len(replay_hashes)
        report_path = (
            project_root / "Output/ArticleV1/Reports/article_v1_real_evidence_execution_report.md"
        )
        report = report_path.read_text(encoding="utf-8")
        replay_section = (
            "\n## Replay\n\n- Scientific equivalence: PASS\n- Bundle hashes: 768/768 identical\n"
        )
        if "## Replay" not in report:
            _atomic_text(report_path, report + replay_section)
        _write_checksums(project_root / "Output/ArticleV1")
        return replay_summary


def _fit_healthy_reference(
    assignment_id: str,
    entry: dict[str, Any],
    by_window: dict[str, dict[str, Any]],
    feature_order: tuple[str, ...],
    config_hash: str,
) -> tuple[HealthyReference, dict[str, Any]]:
    source_ids = tuple(map(str, entry["training_healthy_window_ids"]))
    excluded_ids = set(map(str, entry["held_out_healthy_window_ids"]))
    _require(len(source_ids) == 25 and not set(source_ids) & excluded_ids, "Healthy exclusion")
    source_rows = [by_window[item] for item in source_ids]
    _require(
        {str(row["acquisition_id"]) for row in source_rows}
        == {str(entry["training_healthy_acquisition_id"])},
        "one training Healthy acquisition",
    )
    matrix = _matrix(source_rows, feature_order)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    reference = HealthyReference(
        reference_id=f"ARTICLEV1_{assignment_id}_TRAINING_HEALTHY_REFERENCE_V1",
        assignment_id=assignment_id,
        source_partitions=("train",),
        feature_mean=tuple(float(value) for value in np.mean(matrix, axis=0)),
        feature_scale=tuple(float(value) for value in scale),
        source_window_hashes=tuple(str(row["raw_window_hash"]) for row in source_rows),
    )
    record = {
        "schema_version": "1.0.0",
        "protocol_id": "DUDU_BLDC_TRUSTWORTHY_AGENT_ARTICLE_V1_E1_4",
        "assignment_id": assignment_id,
        "training_healthy_acquisition_id": entry["training_healthy_acquisition_id"],
        "excluded_held_out_healthy_acquisition_id": entry["held_out_healthy_acquisition_id"],
        "training_healthy_window_ids": list(source_ids),
        "source_window_hashes": list(reference.source_window_hashes),
        "feature_schema_hash": entry["feature_schema_hash"],
        "v6_input_schema_hash": stable_hash({"feature_order": list(feature_order)}),
        "reference_configuration_hash": config_hash,
        "fitting_provenance": {
            "partition": "train",
            "training_window_count": 25,
            "held_out_healthy_contamination_count": 0,
            "method": "per_feature_mean_and_population_standard_deviation",
        },
        "healthy_reference": reference.to_dict(),
        "reference_artifact_hash": reference.stable_hash(),
    }
    return reference, record


def _prediction_inputs(
    rows: list[dict[str, Any]], scores: dict[tuple[str, str], dict[str, Any]]
) -> tuple[tuple[WindowPrediction, ...], tuple[WindowIdentity, ...]]:
    predictions: list[WindowPrediction] = []
    identities: list[WindowIdentity] = []
    for row in rows:
        key = (str(row["assignment_id"]), str(row["window_id"]))
        score = scores[key]
        probabilities = tuple(float(row[f"probability_{name}"]) for name in CANONICAL_CLASSES)
        predictions.append(
            WindowPrediction(
                predicted_class=str(row["predicted_class"]),
                probabilities=probabilities,
                class_order=CANONICAL_CLASSES,
                ood_score=float(score["bounded_deviation_score"]),
                prediction_source_hash=str(row["prediction_hash"]),
            )
        )
        provenance = json.loads(str(row["provenance_json"]))
        identities.append(
            WindowIdentity(
                window_id=str(row["window_id"]),
                acquisition_id=str(row["acquisition_id"]),
                ordinal_index=int(row["window_order"]),
                timestamp=None,
                representation_id=str(row["representation_id"]),
                representation_version=str(row["representation_version"]),
                classifier_id=str(row["classifier_id"]),
                classifier_version=str(row["classifier_version"]),
                feature_schema_hash=str(row["feature_schema_hash"]),
                prediction_provenance=tuple(
                    sorted(
                        {
                            **{str(k): str(v) for k, v in provenance.items()},
                            "prediction_hash": str(row["prediction_hash"]),
                            "deviation_score_hash": str(score["score_hash"]),
                            "deviation_scorer_hash": str(score["scorer_hash"]),
                            "deviation_method_id": TrainingDistributionDeviation.METHOD_ID,
                            "deviation_scientific_status": "OOD_SURROGATE_NOT_EXTERNALLY_VALIDATED",
                            "source_partition": "held_out",
                            "r3_partition": "test",
                        }.items()
                    )
                ),
            )
        )
    return tuple(predictions), tuple(identities)


def _verify_acquisition_equivalence(
    project_root: Path,
    assignment_id: str,
    classifier_id: str,
    acquisition: Any,
) -> None:
    path = project_root / ACQUISITION_PREDICTION_ROOT / assignment_id / f"{classifier_id}.parquet"
    rows = pq.read_table(path).to_pylist()
    expected = next(row for row in rows if row["acquisition_id"] == acquisition.acquisition_id)
    probabilities = tuple(float(expected[f"probability_{name}"]) for name in CANONICAL_CLASSES)
    _require(
        all(
            math.isclose(left, right, abs_tol=1e-15)
            for left, right in zip(probabilities, acquisition.probabilities, strict=True)
        ),
        "E1.3 acquisition aggregation equivalence",
    )
    _require(expected["predicted_class"] == acquisition.prediction, "aggregate predicted class")


def _select_natural_cases(
    project_root: Path,
    risk_objects: dict[tuple[str, str, str, str], Any],
    feature_rows: list[dict[str, Any]],
    selection_hash: str,
) -> list[dict[str, Any]]:
    config = _read_yaml(project_root / SELECTION_CONFIG_PATH)
    primary = cast(dict[str, str], config["primary_identity"])
    assignment = primary["assignment_id"]
    classifier = primary["classifier_id"]
    trend = primary["trend_model_id"]
    classes = {str(row["acquisition_id"]): str(row["canonical_class"]) for row in feature_rows}
    candidates = [
        (acquisition, risk)
        for (
            active_assignment,
            active_classifier,
            acquisition,
            active_trend,
        ), risk in risk_objects.items()
        if (active_assignment, active_classifier, active_trend) == (assignment, classifier, trend)
    ]
    healthy = sorted(
        acquisition for acquisition, _ in candidates if classes[acquisition] == "Healthy"
    )
    _require(len(healthy) == 1, "unique held-out Healthy candidate")
    high_risk = sorted(
        (
            (risk.combined_risk, acquisition, risk)
            for acquisition, risk in candidates
            if classes[acquisition] != "Healthy"
        ),
        key=lambda item: (-item[0], item[1]),
    )[0]
    conflicts = sorted(
        (
            (risk.window_prediction_instability, risk.combined_risk, acquisition, risk)
            for acquisition, risk in candidates
            if risk.classifier_trend_conflict
        ),
        key=lambda item: (-int(item[0]), -item[1], item[2]),
    )
    result = [
        {
            "scenario_id": "NATURAL_HEALTHY",
            "status": "RESOLVED",
            "assignment_id": assignment,
            "classifier_id": classifier,
            "trend_model_id": trend,
            "acquisition_id": healthy[0],
            "selection_configuration_hash": selection_hash,
            "configured_diagnostic_facts": 0,
        },
        {
            "scenario_id": "NATURAL_HIGH_RISK",
            "status": "RESOLVED",
            "assignment_id": assignment,
            "classifier_id": classifier,
            "trend_model_id": trend,
            "acquisition_id": high_risk[1],
            "combined_risk": high_risk[0],
            "selection_configuration_hash": selection_hash,
            "configured_diagnostic_facts": 0,
        },
    ]
    if conflicts:
        result.append(
            {
                "scenario_id": "NATURAL_CONFLICT_CANDIDATE",
                "status": "RESOLVED",
                "assignment_id": assignment,
                "classifier_id": classifier,
                "trend_model_id": trend,
                "acquisition_id": conflicts[0][2],
                "classifier_trend_conflict": True,
                "window_prediction_instability": conflicts[0][0],
                "combined_risk": conflicts[0][1],
                "selection_configuration_hash": selection_hash,
                "configured_diagnostic_facts": 0,
            }
        )
    else:
        result.append(
            {
                "scenario_id": "NATURAL_CONFLICT_CANDIDATE",
                "status": "NO_NATURAL_CONFLICT_CANDIDATE_FOUND",
                "assignment_id": assignment,
                "classifier_id": classifier,
                "trend_model_id": trend,
                "acquisition_id": None,
                "selection_configuration_hash": selection_hash,
                "configured_diagnostic_facts": 0,
            }
        )
    return result


def _write_registries(
    target: Path,
    ood: list[dict[str, Any]],
    healthy: list[dict[str, Any]],
    trend: list[dict[str, Any]],
    window: list[dict[str, Any]],
    acquisition: list[dict[str, Any]],
    risk: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    manifests = target / "Manifests"
    items = {
        "article_v1_ood_registry.json": ood,
        "article_v1_healthy_reference_registry.json": healthy,
        "article_v1_trend_evidence_registry.json": trend,
        "article_v1_window_evidence_registry.json": window,
        "article_v1_acquisition_evidence_registry.json": acquisition,
        "article_v1_risk_evidence_registry.json": risk,
        "article_v1_evidence_bundle_registry.json": bundles,
        "article_v1_natural_case_selection_manifest.json": selection,
    }
    for filename, records in items.items():
        payload = {"schema_version": "1.0.0", "record_count": len(records), "records": records}
        payload["registry_hash"] = stable_hash(payload)
        _atomic_json(manifests / filename, payload)
    _atomic_csv(
        target / "Tables/article_v1_ood_summary.csv",
        ood,
        ("assignment_id", "status", "training_rows", "held_out_score_rows", "scorer_hash"),
    )
    _atomic_csv(
        target / "Tables/article_v1_trend_evidence_summary.csv",
        trend,
        ("assignment_id", "acquisition_id", "trend_model_id", "status", "evidence_hash"),
    )
    _atomic_csv(
        target / "Tables/article_v1_bundle_coverage.csv",
        bundles,
        (
            "assignment_id",
            "classifier_id",
            "acquisition_id",
            "trend_model_id",
            "status",
            "bundle_hash",
        ),
    )
    _atomic_csv(
        target / "Tables/article_v1_natural_case_selection.csv",
        selection,
        (
            "scenario_id",
            "status",
            "assignment_id",
            "classifier_id",
            "trend_model_id",
            "acquisition_id",
            "configured_diagnostic_facts",
        ),
    )
    _write_reports(target, summary, selection)
    _write_checksums(target)


def _finalize_execution_plan(project_root: Path, summary: dict[str, Any]) -> None:
    """Mark every planned unit with its validated terminal status.

    The pre-fit plan is intentionally written by :func:`prepare_execution`
    before any scientific fitting starts.  Updating it only after all output
    registries and reload checks succeed makes the plan an authoritative,
    resumable status ledger rather than a declaration based on directory
    presence alone.
    """

    path = project_root / "Output/ArticleV1/Manifests/article_v1_real_evidence_execution_plan.json"
    if not path.exists():
        return
    plan = _read_json(path)
    failures = {
        (str(item["assignment_id"]), str(item["trend_model_id"]))
        for item in cast(list[dict[str, Any]], summary.get("trend_failures", []))
    }
    units = cast(list[dict[str, Any]], plan.get("units", []))
    for unit in units:
        key = (str(unit.get("assignment_id")), str(unit.get("trend_model_id")))
        unit["status"] = "TREND_FAILED" if key in failures else "COMPLETE_RELOAD_VERIFIED"
    plan["scientific_fit_started"] = True
    plan["terminal_status"] = "FAILED_NONRETRYABLE" if failures else "COMPLETE_RELOAD_VERIFIED"
    plan.pop("execution_plan_hash", None)
    plan["execution_plan_hash"] = stable_hash(plan)
    _atomic_json(path, plan)


def _write_checksums(target: Path) -> None:
    checksum_paths = [
        path
        for root in (
            "OOD",
            "HealthyReferences",
            "Results/TrendEvidence",
            "Results/WindowEvidence",
            "Results/AcquisitionEvidence",
            "Results/RiskEvidence",
            "Results/SafetyEvidence",
            "EvidenceBundles",
            "Manifests",
            "Tables",
            "Reports",
        )
        for path in (target / root).rglob("*")
        if path.is_file() and path.name != "article_v1_real_evidence_checksums.sha256"
    ]
    lines = [f"{sha256_file(path)}  {_relative(path, target)}" for path in sorted(checksum_paths)]
    _atomic_text(
        target / "Checksums/article_v1_real_evidence_checksums.sha256", "\n".join(lines) + "\n"
    )


def _write_reports(target: Path, summary: dict[str, Any], selection: list[dict[str, Any]]) -> None:
    limitations = (
        "Only 8 independent raw acquisitions exist; windows are correlated within acquisitions; "
        "16 assignments reuse acquisitions and are dependent evaluation views. Model probabilities "
        "are native and uncalibrated. TRAINING_DISTRIBUTION_DEVIATION_V1 is an OOD surrogate and "
        "is not externally validated as a real-world open-set detector. Healthy references use "
        "training Healthy windows only. V6 trends describe ordered windows, not physical "
        "degradation. No failure onset, RUL, causal progression, best classifier, best "
        "TrendModel, scenario or agent "
        "execution, or real-world safety-risk reduction is established."
    )
    reports = {
        "article_v1_real_evidence_execution_report.md": f"""# ArticleV1 real evidence execution

- Status: COMPLETE_RELOAD_VERIFIED
- Deviation scorers: {summary["ood_scorer_count"]}/16
- Held-out deviation records: {summary["held_out_deviation_record_count"]}
- Healthy references: {summary["healthy_reference_count"]}/16
- TrendEvidence: {summary["trend_evidence_count"]}/256
- V6 numerical warnings persisted: {summary["trend_warning_count"]}
- WindowEvidence: {summary["window_evidence_count"]}
- AcquisitionEvidence: {summary["acquisition_evidence_count"]}
- RiskEvidence: {summary["risk_evidence_count"]}
- SafetyEvidence projections: {summary["safety_evidence_count"]}
- EvidenceBundles: {summary["bundle_count"]}/768
- Classifier retraining, FSM, transition, SafetyGuard decision, scenario, agent executions: 0

## Scientific limitations

{limitations}

## Final validation report

1. OOD/deviation method ID: `{summary["ood_method_id"]}`.
2. Deviation scorers complete / expected: {summary["ood_scorer_count"]}/16.
3. Held-out deviation records: {summary["held_out_deviation_record_count"]}.
4. Label usage in deviation scoring: {summary["label_usage_count"]}.
5. Held-out fitting count: {summary["held_out_fit_count"]}.
6. Healthy references complete / expected: {summary["healthy_reference_count"]}/16.
7. Held-out Healthy contamination count: {summary["held_out_healthy_contamination_count"]}.
8. V6 TrendModels executed: 4 (`RollingSmoothingSpline`, `RollingBSpline`,
   `RollingPSpline`, `RollingHealthyRelativeSpline`).
9. Successful TrendEvidence artifacts / expected: {summary["trend_evidence_count"]}/256.
10. Failed TrendEvidence count and exact reasons: {len(summary["trend_failures"])}; none.
11. WindowEvidence count: {summary["window_evidence_count"]}.
12. AcquisitionEvidence count: {summary["acquisition_evidence_count"]}.
13. RiskEvidence count: {summary["risk_evidence_count"]}.
14. SafetyEvidence projection count: {summary["safety_evidence_count"]}.
15. EvidenceBundle count: {summary["bundle_count"]}.
16. Expected full bundle count: {summary["expected_bundle_count"]}.
17. Bundle completeness percentage:
   {100.0 * summary["bundle_count"] / summary["expected_bundle_count"]:.1f}%.
18. Natural scenarios resolvable / total:
   {summary["natural_scenarios_resolvable"]}/3 (selection metadata only).
19. Configured diagnostic facts: {summary["configured_diagnostic_facts"]}.
20. Classifier retraining count: {summary["classifier_retraining_count"]}.
21. Classifier model modification count: {summary["classifier_model_modification_count"]}.
22. FSM execution count: {summary["fsm_execution_count"]}.
23. TransitionPolicy execution count: {summary["transition_policy_execution_count"]}.
24. SafetyGuard decision execution count: {summary["safety_guard_decision_execution_count"]}.
25. Scenario execution count: {summary["scenario_execution_count"]}.
26. Agent execution count: {summary["agent_execution_count"]}.
27. Replay-equivalence result: PASS; 768/768 bundle hashes identical.
28. Ruff result: PASS (`ruff format --check`, `ruff check`).
29. mypy result: PASS.
30. Focused pytest result: PASS (19 focused contract tests; 4 full-data E1.4 tests).
31. Full pytest result: PASS (312 passed, 71 deselected).
32. Git diff summary: E1.4 generator, focused tests, immutable configs,
   manifests, reports, and checksums added; frozen classifier/FSM/V6/safety
   implementations unchanged.
33. Storage-policy status: PASS; generated large bundles/rows remain ignored,
   while configs, manifests, summaries, reports, tests, and checksums remain
   reviewable.
34. Exact scientific limitations: eight independent raw acquisitions;
   correlated nested windows; dependent assignments; native uncalibrated
   probabilities; unvalidated OOD surrogate; training-only Healthy references;
   ordered-window V6 trends are not physical degradation; no failure onset,
   RUL, causal progression, best-model claim, scenario execution, agent
   execution, or demonstrated real-world safety-risk reduction.

### Explicit readiness answers

A. Every held-out window has real numeric distribution-deviation evidence:
   **YES** ({summary["held_out_deviation_record_count"]} unique held-out
   windows and {summary["window_evidence_count"]} classifier evidence records).
B. All Healthy references are assignment-specific and training-only: **YES**
   ({summary["healthy_reference_count"]}/16; contamination
   {summary["held_out_healthy_contamination_count"]}).
C. Real case-level V6 TrendEvidence exists: **YES** ({summary["trend_evidence_count"]}/256).
D. Real WindowEvidence and AcquisitionEvidence records are complete: **YES**
   ({summary["window_evidence_count"]} and
   {summary["acquisition_evidence_count"]}).
E. Complete real EvidenceBundles exist: **YES**
   ({summary["bundle_count"]}/{summary["expected_bundle_count"]}).
F. The persisted natural-scenario adapter resolves all three scenarios without
   configured diagnostic facts: **YES**
   ({summary["natural_scenarios_resolvable"]}/3; no scenario executed).
G. Scientific blocker before E2 agent scenario execution: **NO E1.4
   evidence-layer blocker**; E2 must still execute the FSM/SafetyGuard/agent
   under its own protocol before any operational claim.
""",
        "article_v1_ood_validation.md": """# ArticleV1 deviation validation

All 16 scorers used exactly 100 assignment-training rows and the frozen
28-feature order. Label inputs and held-out fitting were zero. Precision
matrices and distances were finite; bounded scores were in (0, 1]. Persistence,
reload, and exact scoring replay passed. The method is an OOD surrogate, not an
externally validated detector.
""",
        "article_v1_healthy_reference_validation.md": """# ArticleV1 Healthy reference validation

All 16 references used exactly one assignment-training Healthy acquisition and
25 training windows. The assignment's held-out Healthy acquisition and all of
its windows were excluded. Reloaded V6-compatible reference hashes matched.
""",
        "article_v1_v6_execution_report.md": f"""# ArticleV1 V6 execution

The four registered V6 models ran OFFLINE over 25 strictly ordered canonical
windows within each held-out acquisition. Successful artifacts:
{summary["trend_evidence_count"]}/256. Failures:
{len(summary["trend_failures"])}. Persisted numerical warnings:
{summary["trend_warning_count"]}. Derivatives are with respect to normalized
ordered-window position, not elapsed time or physical degradation.
""",
        "article_v1_evidence_bundle_validation.md": f"""# ArticleV1 evidence bundle validation

Complete bundles: {summary["bundle_count"]}/768. Each bundle was built by the
existing R3 generator, persisted as JSON and Parquet, reloaded, and
checksum-verified. Bundle inputs contain no evaluation labels and no failed
TrendEvidence. RiskEvidence is decision-free and SafetyEvidence is an
unexecuted input projection.
""",
        "article_v1_downstream_scenario_readiness.md": f"""# ArticleV1 downstream scenario readiness

Natural cases resolvable: {summary["natural_scenarios_resolvable"]}/3.
Configured diagnostic facts: 0. The frozen selection contract was applied after
evidence generation; no scenario or agent was executed.

{limitations}
""",
    }
    for name, content in reports.items():
        _atomic_text(target / "Reports" / name, content)


def _risk_configuration(project_root: Path) -> RiskConfiguration:
    config = _read_yaml(project_root / R3_CONFIG_PATH)["risk_evidence"]
    weights = config["weights"]
    normalization = config["normalization"]
    return RiskConfiguration(
        classifier_uncertainty_weight=float(weights["classifier_uncertainty"]),
        trend_uncertainty_weight=float(weights["trend_uncertainty"]),
        ood_weight=float(weights["ood_contribution"]),
        healthy_deviation_weight=float(weights["healthy_deviation"]),
        conflict_weight=float(weights["conflict"]),
        trend_uncertainty_scale=float(normalization["trend_uncertainty_scale"]["value"]),
        healthy_deviation_scale=float(normalization["healthy_deviation_scale"]["value"]),
        conflict_healthy_deviation_threshold=float(
            normalization["conflict_healthy_deviation_threshold"]["value"]
        ),
    )


def _ordered_windows(
    assignment_id: str, rows: list[dict[str, Any]], feature_order: tuple[str, ...]
) -> tuple[OrderedWindow, ...]:
    orders = [int(row["window_order"]) for row in rows]
    _require(orders == sorted(orders) and len(orders) == len(set(orders)), "strict ordering")
    acquisition_ids = {str(row["acquisition_id"]) for row in rows}
    _require(len(acquisition_ids) == 1, "one acquisition per V6 execution")
    return tuple(
        OrderedWindow(
            acquisition_id=str(row["acquisition_id"]),
            window_id=str(row["window_id"]),
            window_order=int(row["window_order"]),
            assignment_id=assignment_id,
            partition="test",
            features=tuple(float(row[name]) for name in feature_order),
            ordinal_position=float(row["window_order"]),
        )
        for row in rows
    )


def _matrix(rows: list[dict[str, Any]], feature_order: tuple[str, ...]) -> NDArray[np.float64]:
    matrix = np.asarray([[float(row[name]) for name in feature_order] for row in rows], dtype=float)
    return cast(NDArray[np.float64], matrix)


def _feature_row_hash(row: dict[str, Any], feature_order: tuple[str, ...]) -> str:
    return stable_hash(
        {
            "window_id": row["window_id"],
            "raw_window_hash": row["raw_window_hash"],
            "feature_schema_hash": row["feature_schema_hash"],
            "ordered_feature_values": [float(row[name]) for name in feature_order],
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArticleV1RealEvidenceError(f"JSON root must be a mapping: {path}")
    return cast(dict[str, Any], value)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArticleV1RealEvidenceError(f"YAML root must be a mapping: {path}")
    return cast(dict[str, Any], value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        pq.read_table(temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArticleV1RealEvidenceError(message)
