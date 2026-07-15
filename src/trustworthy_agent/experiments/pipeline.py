"""Leakage-safe E0-E7 experiment pipeline primitives."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.data.manifest import CANONICAL_CLASSES, CANONICAL_FILE_CLASS_MAP
from trustworthy_agent.data.splits import (
    SplitProvenance,
    analyze_duplicates,
    create_leakage_safe_split,
)
from trustworthy_agent.data.validation import source_data_root
from trustworthy_agent.exceptions import DataQualityError, LeakageError, ReproducibilityError
from trustworthy_agent.strategies.spline._core import fit_training_healthy_reference
from trustworthy_agent.strategies.spline.smoothing_spline import SmoothingSplineStrategy

ExperimentStatus = Literal["COMPLETED", "SKIPPED_DATA_UNAVAILABLE", "FAILED"]
RepresentationId = Literal["V1_CLASSICAL", "V2_SPLINE", "V3_HYBRID"]
ModelFamily = Literal["logistic_regression", "random_forest"]

LABEL_COLUMNS = ("class", "label", "target", "diagnosis", "fault_type", "condition")
IDENTIFIER_TOKENS = ("id", "index", "motor", "scenario", "case", "window", "group", "file")


@dataclass(frozen=True)
class ExperimentResult:
    """Structured result for one protocol experiment.

    Purpose:
        Represent executed, skipped, and failed experiments without fabricating
        metrics when source data are unavailable.
    Parameters:
        experiment_id: E0-E7 identifier.
        status: Completion status.
        metrics: Computed metrics only when execution genuinely completed.
        artifacts: Artifact paths or in-memory structured outputs.
        provenance: Split, seed, representation, and leakage evidence.
        reason_codes: Machine-readable reasons for skips or failures.
    Return value:
        Immutable result object.
    Raised exceptions:
        None during construction.
    Scientific assumptions:
        Empty metrics on a skipped experiment mean "not run", not zero
        performance.
    Side effects:
        None.
    Reproducibility implications:
        JSON serialization preserves audit-friendly execution evidence.
    """

    experiment_id: str
    status: ExperimentStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable experiment evidence."""

        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "provenance": self.provenance,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class TabularDataset:
    """In-memory analysis table used by experiment tests and runners.

    Purpose:
        Carry rows, label identity, and verified grouping candidates without
        inventing dataset schema properties.
    Parameters:
        rows: Immutable tuple of row mappings.
        label_column: Verified label column.
        candidate_group_columns: Candidate identifiers discovered from actual
            headers/content.
    Return value:
        Immutable dataset view.
    Raised exceptions:
        None during construction.
    Scientific assumptions:
        Canonical labels are enforced by experiment execution, not by silently
        relabeling inputs.
    Side effects:
        None.
    Reproducibility implications:
        Dataset hash is computed from row contents.
    """

    rows: tuple[Mapping[str, Any], ...]
    label_column: str
    candidate_group_columns: tuple[str, ...] = ()

    def dataset_hash(self) -> str:
        """Return a stable hash over row content and label identity."""

        return _stable_hash(
            {
                "rows": [_row_to_json_dict(row) for row in self.rows],
                "label_column": self.label_column,
                "candidate_group_columns": self.candidate_group_columns,
            }
        )


@dataclass(frozen=True)
class S3ArtifactSet:
    """Persisted S3 artifact metadata for a generated spline feature table."""

    dataset: TabularDataset
    feature_names: tuple[str, ...]
    artifact_rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SupervisedModelArtifact:
    """Leakage-safe fitted model artifact for metrics, XAI, and ablations."""

    estimator: Any
    representation_id: RepresentationId
    model_family: ModelFamily
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    x_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_train: NDArray[Any]
    y_test: NDArray[Any]
    test_indices: tuple[int, ...]
    predictions: tuple[str, ...]
    probabilities: NDArray[np.float64] | None
    split: SplitProvenance
    model_hash: str
    feature_hash: str


class TrainingOnlyImputer:
    """Mean imputer whose fit method is guarded against non-training data.

    Purpose:
        Implement leakage-safe imputation for baseline experiments.
    Parameters:
        None.
    Return value:
        Fitted transformer after `fit`.
    Raised exceptions:
        LeakageError if fit is requested on any split other than `train`.
        DataQualityError if transform is called before fit.
    Scientific assumptions:
        Mean imputation is an engineering baseline, not a domain claim.
    Side effects:
        Stores training means in memory.
    Reproducibility implications:
        Fit scope is explicit and recorded by callers.
    """

    def __init__(self) -> None:
        self._means: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], *, split_name: str) -> TrainingOnlyImputer:
        """Fit column means on the training split only."""

        if split_name != "train":
            raise LeakageError("Imputer fit attempted outside the training split.")
        means = np.nanmean(x, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        self._means = cast(NDArray[np.float64], means.astype(float))
        return self

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fill missing/non-finite values using training means."""

        if self._means is None:
            raise DataQualityError("Imputer must be fit before transform.")
        clean = x.astype(float, copy=True)
        missing = ~np.isfinite(clean)
        if np.any(missing):
            clean[missing] = np.take(self._means, np.where(missing)[1])
        return cast(NDArray[np.float64], clean)


class TrainingOnlyStandardScaler:
    """Standard scaler whose fit method is guarded against leakage.

    Purpose:
        Fit centering and scaling parameters on training data only.
    Parameters:
        None.
    Return value:
        Fitted transformer after `fit`.
    Raised exceptions:
        LeakageError if fit is requested outside `train`.
        DataQualityError if transform is called before fit.
    Scientific assumptions:
        Scaling statistics learned from validation/test data would leak
        distribution information and are therefore prohibited.
    Side effects:
        Stores training mean and standard deviation in memory.
    Reproducibility implications:
        Fit scope is deterministic and auditable.
    """

    def __init__(self) -> None:
        self._mean: NDArray[np.float64] | None = None
        self._scale: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], *, split_name: str) -> TrainingOnlyStandardScaler:
        """Fit mean and scale on the training split only."""

        if split_name != "train":
            raise LeakageError("Scaler fit attempted outside the training split.")
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        self._mean = cast(NDArray[np.float64], mean.astype(float))
        self._scale = cast(NDArray[np.float64], np.where(scale <= 0.0, 1.0, scale).astype(float))
        return self

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply training-fitted scaling parameters."""

        if self._mean is None or self._scale is None:
            raise DataQualityError("Scaler must be fit before transform.")
        return cast(NDArray[np.float64], ((x - self._mean) / self._scale).astype(float))


class LeakageSafePreprocessor:
    """Baseline preprocessing chain fitted only on the training split."""

    def __init__(self) -> None:
        self.imputer = TrainingOnlyImputer()
        self.scaler = TrainingOnlyStandardScaler()

    def fit_transform_train(self, x_train: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit imputation/scaling on train and return transformed train rows."""

        imputed = self.imputer.fit(x_train, split_name="train").transform(x_train)
        return self.scaler.fit(imputed, split_name="train").transform(imputed)

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform validation/test rows without refitting."""

        return self.scaler.transform(self.imputer.transform(x))


def load_csv_dataset(path: Path, *, label_column: str | None = None) -> TabularDataset:
    """Load a small analysis CSV without assuming DUDU-BLDC feature count."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = tuple(cast(Mapping[str, Any], dict(row)) for row in reader)
    if not rows:
        raise DataQualityError(f"No rows found in analysis dataset: {path}")
    resolved_label = label_column or _discover_label_column(rows)
    candidate_groups = tuple(
        column for column in rows[0] if any(token in column.lower() for token in IDENTIFIER_TOKENS)
    )
    return TabularDataset(
        rows=rows,
        label_column=resolved_label,
        candidate_group_columns=candidate_groups,
    )


def load_dudu_bldc_analysis_dataset(project_root: Path) -> TabularDataset:
    """Load the four canonical DUDU-BLDC analytical class CSV files.

    Purpose:
        Build an analysis table from verified local DUDU-BLDC source files
        without modifying immutable input data.
    Parameters:
        project_root: Repository root.
    Return value:
        TabularDataset with canonical class labels and source case identity.
    Raised exceptions:
        FileNotFoundError if the canonical source root is unavailable.
        DataQualityError if a required analytical file is missing or empty.
    Scientific assumptions:
        Class labels are mapped only from the four canonical analytical files
        declared in the protocol.
    Side effects:
        None.
    Reproducibility implications:
        Case IDs preserve source file and `Experiment ID` identity.
    """

    root = source_data_root(project_root)
    rows: list[Mapping[str, Any]] = []
    for filename, class_label in CANONICAL_FILE_CLASS_MAP.items():
        path = _find_required_file(root, filename)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            file_rows = list(reader)
        if not file_rows:
            raise DataQualityError(f"Canonical DUDU-BLDC analytical file is empty: {path}")
        for row_number, row in enumerate(file_rows, start=1):
            experiment_id = str(row.get("Experiment ID", row_number))
            rows.append(
                {
                    **dict(row),
                    "Class": class_label,
                    "source_file": filename,
                    "case_id": f"{filename}:{experiment_id}",
                    "window_id": experiment_id,
                }
            )
    return TabularDataset(
        rows=tuple(rows),
        label_column="Class",
        candidate_group_columns=("Experiment ID",),
    )


def generate_s3_spline_artifacts(
    dataset: TabularDataset,
    split: SplitProvenance,
    *,
    project_root: Path | None,
    experiment_id: str,
    config: Mapping[str, Any],
) -> S3ArtifactSet:
    """Generate S3 spline-derived features using training-only Healthy reference.

    Purpose:
        Convert validated DUDU-BLDC analytical rows into the mandatory S3
        spline feature contract for V2/V3 experiments.
    Parameters:
        dataset: Analysis dataset with canonical labels.
        split: Exact train/validation/test split provenance.
        project_root: Optional repository root for artifact persistence.
        experiment_id: Experiment identity recorded in provenance.
        config: Resolved S3 strategy YAML mapping.
    Return value:
        S3ArtifactSet containing augmented dataset and persisted manifest data.
    Raised exceptions:
        DataQualityError if no training Healthy rows are available.
        StrategyExecutionError from the S3 strategy for invalid references.
    Scientific assumptions:
        The ordered series is a constructed ordered diagnostic feature sequence
        across validated current/speed feature columns; it is not real
        longitudinal degradation time.
    Side effects:
        Writes S3 CSV/JSON artifacts under `Output/` when `project_root` is set.
    Reproducibility implications:
        Records split hash, strategy identity, input features, reference hash,
        fit statuses, and artifact hash.
    """

    input_features = _feature_columns(
        dataset.rows, dataset.label_column, dataset.candidate_group_columns
    )
    healthy_train_indices = tuple(
        index
        for index in split.train_indices
        if str(dataset.rows[index].get(dataset.label_column)) == "Healthy"
    )
    healthy_train_series = [
        _series_for_row(dataset.rows[index], input_features) for index in healthy_train_indices
    ]
    if not healthy_train_series:
        raise DataQualityError("S3 Healthy reference requires training-fold Healthy observations.")
    healthy_reference = fit_training_healthy_reference(
        healthy_train_series,
        reference_id=f"{experiment_id}:{split.split_hash}:healthy_reference",
    )
    reference_hash = _stable_hash(healthy_reference)
    strategy = SmoothingSplineStrategy()
    strategy.validate_config(config)
    augmented_rows: list[Mapping[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(dataset.rows):
        series = _series_for_row(row, input_features)
        context = AgentContext(
            experiment_id=experiment_id,
            case_id=str(row.get("case_id", row_index)),
            window_id=str(row.get("window_id", row.get("Experiment ID", row_index))),
            dataset_id="DUDU-BLDC",
            dataset_version="v1",
            raw_record_reference=str(row.get("case_id", row_index)),
            classical_features={name: row.get(name) for name in input_features},
            derived_facts={
                "ordered_series": series,
                "healthy_reference": healthy_reference,
                "input_available": True,
                "validation_passed": True,
            },
        )
        result = strategy.execute(context)
        spline_features = cast(Mapping[str, Any], result.facts["spline_features"])
        prefixed = {f"spline_{key}": value for key, value in spline_features.items()}
        augmented_rows.append({**dict(row), **prefixed})
        artifact = {
            "experiment_id": experiment_id,
            "split_id": split.split_id,
            "split_hash": split.split_hash,
            "case_id": str(row.get("case_id", row_index)),
            "window_id": str(row.get("window_id", row.get("Experiment ID", row_index))),
            "split_role": _split_role(row_index, split),
            "strategy_name": strategy.strategy_name,
            "strategy_version": strategy.strategy_version,
            "resolved_parameters": dict(config),
            "input_feature_names": list(input_features),
            "healthy_reference_provenance": {
                "fit_scope": "training_only",
                "training_indices": list(healthy_train_indices),
                "training_healthy_count": len(healthy_train_series),
                "healthy_reference_hash": reference_hash,
            },
            "fit_status": spline_features.get("fit_status"),
            "features": dict(spline_features),
        }
        artifact["artifact_hash"] = _stable_hash(artifact)
        artifact_rows.append(artifact)
    spline_feature_names = [f"spline_{name}" for name in _s3_output_names()]
    manifest = {
        "experiment_id": experiment_id,
        "split_id": split.split_id,
        "split_hash": split.split_hash,
        "strategy_name": strategy.strategy_name,
        "strategy_version": strategy.strategy_version,
        "input_feature_names": list(input_features),
        "spline_feature_names": spline_feature_names,
        "healthy_reference_hash": reference_hash,
        "healthy_reference_fit_scope": "training_only",
        "healthy_reference_training_indices": list(healthy_train_indices),
        "ordering_semantics": "constructed_ordered_diagnostic_feature_sequence",
        "allow_real_longitudinal_claim": False,
        "fit_status_counts": _count_values(
            str(artifact["fit_status"]) for artifact in artifact_rows
        ),
    }
    manifest["artifact_hash"] = _stable_hash({"manifest": manifest, "rows": artifact_rows})
    artifact_set = S3ArtifactSet(
        dataset=TabularDataset(
            rows=tuple(augmented_rows),
            label_column=dataset.label_column,
            candidate_group_columns=dataset.candidate_group_columns,
        ),
        feature_names=tuple(spline_feature_names),
        artifact_rows=tuple(artifact_rows),
        manifest=manifest,
    )
    if project_root is not None:
        _persist_s3_artifacts(project_root, artifact_set)
    return artifact_set


def build_representation(
    dataset: TabularDataset,
    representation_id: RepresentationId,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Build V1, V2, or V3 feature matrices with deterministic column order."""

    if not dataset.rows:
        return np.empty((0, 0), dtype=float), ()
    columns = _feature_columns(dataset.rows, dataset.label_column, dataset.candidate_group_columns)
    classical = tuple(column for column in columns if not _is_spline_feature(column))
    spline = tuple(column for column in columns if _is_spline_feature(column))
    if representation_id == "V1_CLASSICAL":
        selected = classical
    elif representation_id == "V2_SPLINE":
        selected = spline
    else:
        selected = classical + spline
    if not selected:
        raise DataQualityError(f"No numeric columns available for {representation_id}.")
    matrix = np.asarray(
        [[_coerce_float(row.get(column)) for column in selected] for row in dataset.rows],
        dtype=float,
    )
    return cast(NDArray[np.float64], matrix), selected


def run_e0_dataset_audit(dataset: TabularDataset | None) -> ExperimentResult:
    """Run E0 dataset audit or explicitly skip when no data are available."""

    if dataset is None:
        return _skipped("E0_dataset_audit", "DATASET_UNAVAILABLE")
    labels = [str(row.get(dataset.label_column, "")) for row in dataset.rows]
    feature_columns = _feature_columns(
        dataset.rows, dataset.label_column, dataset.candidate_group_columns
    )
    duplicate_analysis = analyze_duplicates(
        dataset.rows,
        feature_columns=feature_columns,
        tolerance=1.0e-12,
    )
    class_distribution = {label: labels.count(label) for label in sorted(set(labels))}
    return ExperimentResult(
        experiment_id="E0_dataset_audit",
        status="COMPLETED",
        metrics={
            "row_count": len(dataset.rows),
            "column_count": len(dataset.rows[0]) if dataset.rows else 0,
            "class_distribution": class_distribution,
            "exact_duplicate_rows": duplicate_analysis.exact_duplicate_rows,
            "near_duplicate_pairs": duplicate_analysis.near_duplicate_pairs,
        },
        artifacts={
            "feature_columns": list(feature_columns),
            "candidate_group_columns": list(dataset.candidate_group_columns),
        },
        provenance={"dataset_hash": dataset.dataset_hash()},
    )


def create_experiment_split(dataset: TabularDataset, *, seed: int) -> SplitProvenance:
    """Create and return the canonical leakage-safe experiment split."""

    features = _feature_columns(dataset.rows, dataset.label_column, dataset.candidate_group_columns)
    return create_leakage_safe_split(
        dataset.rows,
        label_column=dataset.label_column,
        candidate_group_columns=dataset.candidate_group_columns,
        feature_columns=features,
        seed=seed,
        split_id="canonical_holdout_v1",
    )


def run_supervised_experiment(
    dataset: TabularDataset | None,
    *,
    experiment_id: str,
    representation_id: RepresentationId,
    model_family: ModelFamily,
    seed: int,
    split: SplitProvenance | None = None,
) -> ExperimentResult:
    """Run leakage-safe baseline classification for E1, E2, or E3."""

    if dataset is None:
        return _skipped(experiment_id, "DATASET_UNAVAILABLE")
    artifact = train_supervised_model_artifact(
        dataset,
        representation_id=representation_id,
        model_family=model_family,
        seed=seed,
        split=split,
    )
    from trustworthy_agent.experiments.metrics import compute_classification_metrics

    metrics = compute_classification_metrics(
        y_true=[str(label) for label in artifact.y_test],
        y_pred=list(artifact.predictions),
        labels=CANONICAL_CLASSES,
        probabilities=artifact.probabilities,
        probability_classes=artifact.classes,
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        status="COMPLETED",
        metrics=metrics,
        artifacts={
            "feature_names": list(artifact.feature_names),
            "feature_count": len(artifact.feature_names),
            "feature_hash": artifact.feature_hash,
        },
        provenance={
            "representation_id": representation_id,
            "model_family": model_family,
            "seed": seed,
            "dataset_hash": dataset.dataset_hash(),
            "split": artifact.split.to_json_dict(),
            "learned_operations_fit_scope": {
                "imputation": "training_only",
                "scaling": "training_only",
                "model_fit": "training_only",
                "calibration": "not_used",
                "feature_selection": "not_used",
                "threshold_optimization": "not_used",
                "ood_detector": "not_used",
            },
            "model_hash": artifact.model_hash,
        },
    )


def train_supervised_model_artifact(
    dataset: TabularDataset,
    *,
    representation_id: RepresentationId,
    model_family: ModelFamily,
    seed: int,
    split: SplitProvenance | None = None,
) -> SupervisedModelArtifact:
    """Fit a classifier using only training rows and return auditable internals."""

    _validate_canonical_labels(dataset)
    active_split = split or create_experiment_split(dataset, seed=seed)
    x_all, feature_names = build_representation(dataset, representation_id)
    y_all = np.asarray([str(row[dataset.label_column]) for row in dataset.rows])
    train_idx = np.asarray(active_split.train_indices, dtype=int)
    test_idx = np.asarray(active_split.test_indices, dtype=int)
    if train_idx.size == 0 or test_idx.size == 0:
        raise DataQualityError("Train and test splits must both contain rows.")
    preprocessor = LeakageSafePreprocessor()
    x_train = preprocessor.fit_transform_train(cast(NDArray[np.float64], x_all[train_idx]))
    x_test = preprocessor.transform(cast(NDArray[np.float64], x_all[test_idx]))
    estimator = _build_estimator(model_family, seed)
    estimator.fit(x_train, y_all[train_idx])
    predictions = tuple(str(label) for label in estimator.predict(x_test))
    probabilities = (
        cast(NDArray[np.float64], estimator.predict_proba(x_test))
        if hasattr(estimator, "predict_proba")
        else None
    )
    classes = tuple(str(label) for label in estimator.classes_)
    feature_hash = _stable_hash(list(feature_names))
    model_hash = _stable_hash(
        {
            "model_family": model_family,
            "feature_names": feature_names,
            "classes": classes,
            "seed": seed,
            "split_hash": active_split.split_hash,
        }
    )
    return SupervisedModelArtifact(
        estimator=estimator,
        representation_id=representation_id,
        model_family=model_family,
        feature_names=feature_names,
        classes=classes,
        x_train=x_train,
        x_test=x_test,
        y_train=cast(NDArray[Any], y_all[train_idx]),
        y_test=cast(NDArray[Any], y_all[test_idx]),
        test_indices=tuple(int(index) for index in test_idx),
        predictions=predictions,
        probabilities=probabilities,
        split=active_split,
        model_hash=model_hash,
        feature_hash=feature_hash,
    )


def run_e1_classical(
    dataset: TabularDataset | None, *, seed: int, split: SplitProvenance | None = None
) -> tuple[ExperimentResult, ...]:
    """Run E1 V1 Classical with Logistic Regression and Random Forest."""

    return (
        run_supervised_experiment(
            dataset,
            experiment_id="E1_classical_logistic_regression",
            representation_id="V1_CLASSICAL",
            model_family="logistic_regression",
            seed=seed,
            split=split,
        ),
        run_supervised_experiment(
            dataset,
            experiment_id="E1_classical_random_forest",
            representation_id="V1_CLASSICAL",
            model_family="random_forest",
            seed=seed,
            split=split,
        ),
    )


def run_e2_spline(
    dataset: TabularDataset | None, *, seed: int, split: SplitProvenance | None = None
) -> ExperimentResult:
    """Run E2 V2 Spline-only baseline when spline features exist."""

    return run_supervised_experiment(
        dataset,
        experiment_id="E2_spline_random_forest",
        representation_id="V2_SPLINE",
        model_family="random_forest",
        seed=seed,
        split=split,
    )


def run_e3_hybrid(
    dataset: TabularDataset | None, *, seed: int, split: SplitProvenance | None = None
) -> ExperimentResult:
    """Run E3 V3 Hybrid baseline."""

    return run_supervised_experiment(
        dataset,
        experiment_id="E3_hybrid_random_forest",
        representation_id="V3_HYBRID",
        model_family="random_forest",
        seed=seed,
        split=split,
    )


def run_e7_reproduction_verification(results: Sequence[ExperimentResult]) -> ExperimentResult:
    """Verify that result hashes are stable enough for reproduction checks."""

    if not results:
        return _skipped("E7_reproduction_verification", "NO_RESULTS_TO_VERIFY")
    hashes = [_stable_hash(result.to_json_dict()) for result in results]
    return ExperimentResult(
        experiment_id="E7_reproduction_verification",
        status="COMPLETED",
        metrics={"result_count": len(results), "unique_result_hashes": len(set(hashes))},
        provenance={"result_hashes": hashes},
    )


def compare_reproduction_results(
    left: Sequence[ExperimentResult],
    right: Sequence[ExperimentResult],
) -> ExperimentResult:
    """Compare two deterministic result sequences for E7 replay/reproduction."""

    left_hash = _stable_hash([result.to_json_dict() for result in left])
    right_hash = _stable_hash([result.to_json_dict() for result in right])
    if left_hash != right_hash:
        raise ReproducibilityError("Reproduction result hashes differ.")
    return ExperimentResult(
        experiment_id="E7_reproduction_verification",
        status="COMPLETED",
        metrics={"reproduction_match": True},
        provenance={"left_hash": left_hash, "right_hash": right_hash},
    )


def run_e7_deterministic_reproduction(
    dataset: TabularDataset | None,
    *,
    seed: int,
    split: SplitProvenance | None,
) -> ExperimentResult:
    """Run and rerun one deterministic E3 experiment and compare fingerprints."""

    if dataset is None or split is None:
        return _skipped("E7_reproduction_verification", "DATASET_UNAVAILABLE")
    first = run_e3_hybrid(dataset, seed=seed, split=split)
    second = run_e3_hybrid(dataset, seed=seed, split=split)
    comparison = compare_reproduction_results((first,), (second,))
    return ExperimentResult(
        experiment_id="E7_reproduction_verification",
        status="COMPLETED",
        metrics={
            "reproduction_match": True,
            "deterministic_experiment": "E3_hybrid_random_forest",
            "accuracy": first.metrics.get("accuracy"),
            "macro_f1": first.metrics.get("macro_f1"),
        },
        provenance={
            **comparison.provenance,
            "split_hash": split.split_hash,
            "seed": seed,
            "volatile_fields_excluded": ["run_id", "timestamp", "host_temporary_paths"],
        },
    )


def _build_estimator(model_family: ModelFamily, seed: int) -> Any:
    if model_family == "logistic_regression":
        return LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        )
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def persist_experiment_outputs(project_root: Path, results: Sequence[ExperimentResult]) -> None:
    """Persist real experiment result tables and manifests under `Output/`."""

    results_root = project_root / "Output/Results"
    models_root = project_root / "Output/Models"
    manifests_root = project_root / "Output/Manifests"
    tables_root = project_root / "Output/Tables"
    results_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    classification_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    agent_metric_rows: list[dict[str, Any]] = []
    for result in results:
        if result.experiment_id.startswith("E6_"):
            execution = result.artifacts.get("execution")
            if isinstance(execution, Mapping):
                scenario_rows.append(
                    {
                        "scenario_id": execution.get("scenario_id"),
                        "case_id": execution.get("case_id"),
                        "source_kind": execution.get("source_kind"),
                        "true_class": execution.get("true_class"),
                        "predicted_class": execution.get("predicted_class"),
                        "confidence": execution.get("confidence"),
                        "data_quality": execution.get("data_quality"),
                        "explanation_score": execution.get("explanation_score"),
                        "explanation_artifact_ref": execution.get("explanation_artifact_ref"),
                        "model_agreement": execution.get("model_agreement"),
                        "risk_score": execution.get("risk_score"),
                        "ood_score": execution.get("ood_score"),
                        "state_path": execution.get("state_path"),
                        "safety_rules_triggered": execution.get("safety_rules_triggered"),
                        "final_action": execution.get("final_action"),
                        "decision_reason": execution.get("decision_reason"),
                        "ordering_provenance": execution.get("ordering_provenance"),
                        "ordering_provenance_complete": execution.get(
                            "ordering_provenance_complete"
                        ),
                        "unsupported_temporal_claims": execution.get("unsupported_temporal_claims"),
                        "perturbation_provenance": execution.get("perturbation_provenance"),
                        "conflict_evidence": execution.get("conflict_evidence"),
                        "high_risk_evidence": execution.get("high_risk_evidence"),
                        "data_quality_failure_evidence": execution.get(
                            "data_quality_failure_evidence"
                        ),
                        "pass_fail": result.metrics.get("pass_fail"),
                        "audit_log": execution.get("audit_log"),
                    }
                )
        if result.experiment_id == "E4_transition_policy_ablation":
            for item in result.artifacts.get("policy_results", []):
                if isinstance(item, Mapping):
                    transition_rows.append(
                        {
                            "policy_name": item.get("policy_name"),
                            "policy_version": item.get("policy_version"),
                            **dict(item.get("metrics", {})),
                            "audit_logs": item.get("audit_logs"),
                        }
                    )
        if result.experiment_id == "E5_strategy_ablation":
            for item in result.artifacts.get("strategy_results", []):
                if isinstance(item, Mapping):
                    strategy_rows.append(
                        {
                            "state": item.get("state"),
                            "experiment_id": item.get("experiment_id"),
                            "changed_state": item.get("changed_state"),
                            "baseline_strategy": item.get("baseline_strategy"),
                            "alternate_strategy": item.get("alternate_strategy"),
                            "strategy_version": item.get("strategy_version"),
                            "active_strategy_name": item.get("active_strategy_name"),
                            "active_strategy_version": item.get("active_strategy_version"),
                            "configuration_hash": item.get("configuration_hash"),
                            "full_resolved_config_hash": item.get("full_resolved_config_hash"),
                            "dataset_version": item.get("dataset_version"),
                            "split_hash": item.get("split_hash"),
                            "model_hash": item.get("model_hash"),
                            "scientific_metrics": item.get("scientific_metrics"),
                            "state_paths": item.get("state_paths"),
                            "final_actions": item.get("final_actions"),
                            "audit_references": item.get("audit_references"),
                            "result_fingerprint": item.get("result_fingerprint"),
                            **dict(item.get("metrics", {})),
                            "audit_logs": item.get("audit_logs"),
                        }
                    )
        if any(key.endswith("_rate") or key.endswith("_completeness") for key in result.metrics):
            agent_metric_rows.append({"experiment_id": result.experiment_id, **result.metrics})
        if (
            result.status != "COMPLETED"
            or "accuracy" not in result.metrics
            or "representation_id" not in result.provenance
        ):
            continue
        classification_rows.append(
            {
                "experiment_id": result.experiment_id,
                "representation_id": result.provenance.get("representation_id"),
                "model_family": result.provenance.get("model_family"),
                "accuracy": result.metrics.get("accuracy"),
                "macro_f1": result.metrics.get("macro_f1"),
                "balanced_accuracy": result.metrics.get("balanced_accuracy"),
                "log_loss": result.metrics.get("log_loss"),
                "brier_score": result.metrics.get("brier_score"),
                "expected_calibration_error": result.metrics.get("expected_calibration_error"),
                "split_hash": _nested(result.provenance, "split", "split_hash"),
                "model_hash": result.provenance.get("model_hash"),
            }
        )
        per_class = result.metrics.get("per_class", {})
        if isinstance(per_class, Mapping):
            for label, metrics in per_class.items():
                if isinstance(metrics, Mapping):
                    per_class_rows.append(
                        {
                            "experiment_id": result.experiment_id,
                            "class_label": label,
                            **dict(metrics),
                        }
                    )
        model_manifest = {
            "experiment_id": result.experiment_id,
            "model_type": result.provenance.get("model_family"),
            "strategy_name": result.provenance.get("model_family"),
            "strategy_version": "1.0.0",
            "feature_representation": result.provenance.get("representation_id"),
            "input_feature_names": result.artifacts.get("feature_names", []),
            "training_split_hash": _nested(result.provenance, "split", "split_hash"),
            "random_seed": result.provenance.get("seed"),
            "artifact_sha256": result.provenance.get("model_hash"),
        }
        _write_json(models_root / f"{result.experiment_id}_model_manifest.json", model_manifest)
    _write_table(results_root / "classification_metrics.csv", classification_rows)
    _write_table(results_root / "per_class_metrics.csv", per_class_rows)
    _write_table(results_root / "agent_metrics.csv", agent_metric_rows)
    _write_table(tables_root / "table_agent_scenarios.csv", scenario_rows)
    _write_table(tables_root / "table_transition_policy_ablation.csv", transition_rows)
    _write_table(tables_root / "table_strategy_ablation.csv", strategy_rows)
    _write_table(tables_root / "table_classical_vs_spline_vs_hybrid.csv", classification_rows)
    _write_json(
        manifests_root / "run_manifest.json",
        {"results": [result.to_json_dict() for result in results]},
    )


def _feature_columns(
    rows: tuple[Mapping[str, Any], ...],
    label_column: str,
    candidate_group_columns: tuple[str, ...],
) -> tuple[str, ...]:
    if not rows:
        return ()
    excluded = {label_column, *candidate_group_columns}
    columns = sorted(column for column in rows[0] if column not in excluded)
    return tuple(
        column
        for column in columns
        if column.lower() not in LABEL_COLUMNS
        and not any(token in column.lower() for token in IDENTIFIER_TOKENS)
        and all(_is_number_like(row.get(column)) or _is_missing(row.get(column)) for row in rows)
        and any(_is_number_like(row.get(column)) for row in rows)
    )


def _find_required_file(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    analysis_matches = [path for path in matches if "AnalysisData" in path.parts]
    if analysis_matches:
        return analysis_matches[0]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Required DUDU-BLDC analytical file not found: {filename}")


def _series_for_row(row: Mapping[str, Any], feature_names: Sequence[str]) -> list[float]:
    return [_coerce_float(row.get(name)) for name in feature_names]


def _s3_output_names() -> tuple[str, ...]:
    return (
        "smoothed_value",
        "slope",
        "second_derivative",
        "curvature",
        "max_curvature",
        "distance_from_healthy",
        "area_deviation_from_healthy",
        "first_threshold_crossing",
        "fit_quality",
        "fit_status",
    )


def _split_role(index: int, split: SplitProvenance) -> str:
    if index in set(split.train_indices):
        return "train"
    if index in set(split.validation_indices):
        return "validation"
    if index in set(split.test_indices):
        return "test"
    return "unassigned"


def _count_values(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _persist_s3_artifacts(project_root: Path, artifact_set: S3ArtifactSet) -> None:
    intermediate_root = project_root / "Data/IntermediateData"
    analysis_root = project_root / "Data/AnalysisData"
    results_root = project_root / "Output/Results"
    manifests_root = project_root / "Output/Manifests"
    for path in (intermediate_root, analysis_root, results_root, manifests_root):
        path.mkdir(parents=True, exist_ok=True)
    _write_table(intermediate_root / "s3_spline_artifacts.csv", artifact_set.artifact_rows)
    _write_table(
        analysis_root / "dudu_bldc_analysis_with_spline_features.csv", artifact_set.dataset.rows
    )
    _write_json(
        results_root / "s3_spline_artifacts.json", {"rows": list(artifact_set.artifact_rows)}
    )
    _write_json(manifests_root / "s3_spline_manifest.json", artifact_set.manifest)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fieldnames})


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _is_spline_feature(column: str) -> bool:
    lower = column.lower()
    return lower.startswith("spline_") or lower in {
        "smoothed_value",
        "slope",
        "second_derivative",
        "curvature",
        "max_curvature",
        "distance_from_healthy",
        "area_deviation_from_healthy",
        "first_threshold_crossing",
    }


def _discover_label_column(rows: tuple[Mapping[str, Any], ...]) -> str:
    for column in rows[0]:
        if column.lower() in LABEL_COLUMNS:
            return column
    raise DataQualityError("No label column discovered from actual CSV header.")


def _validate_canonical_labels(dataset: TabularDataset) -> None:
    observed = {str(row.get(dataset.label_column, "")) for row in dataset.rows}
    unexpected = sorted(observed - set(CANONICAL_CLASSES))
    if unexpected:
        raise DataQualityError(f"Unexpected class labels: {unexpected}")


def _coerce_float(value: object) -> float:
    if _is_missing(value):
        return float("nan")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _is_number_like(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "na", "nan", "null", "none"}


def _row_to_json_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in sorted(row)}


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _skipped(experiment_id: str, reason_code: str) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        status="SKIPPED_DATA_UNAVAILABLE",
        reason_codes=(reason_code,),
    )
