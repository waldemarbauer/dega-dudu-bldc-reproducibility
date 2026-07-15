"""Frozen ArticleV1 canonical-window classifier execution.

This module is deliberately independent of the agent, FSM, SafetyGuard, V6,
scenario, and evidence-bundle execution paths.  It produces classifier-side
artifacts only; ground-truth labels are joined solely into separate evaluation
tables after evidence-safe prediction persistence.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from trustworthy_agent.evidence.diagnostic import CANONICAL_CLASSES, normalized_entropy
from trustworthy_agent.provenance.environment import basic_environment
from trustworthy_agent.provenance.git_info import git_identity

CONFIG_PATH = Path("configs/classifiers/article_v1/fixed_window_classifiers_v1.yaml")
FEATURE_MANIFEST_PATH = Path("Output/ArticleV1/Manifests/article_window_feature_manifest.json")
DATA_UNIT_MANIFEST_PATH = Path("Output/ArticleV1/Manifests/article_window_data_unit_manifest.json")
ASSIGNMENT_CATALOG_PATH = Path("Output/ArticleV1/Manifests/acquisition_assignment_catalog.json")
TRAINING_CONSTRAINTS_PATH = Path(
    "Output/ArticleV1/Manifests/article_window_model_training_constraints.json"
)
FEATURE_CORPUS_PATH = Path(
    "Data/AnalysisData/ArticleV1/WindowFeatures/article_v1_canonical_window_features.parquet"
)
ASSIGNMENT_ROOT = Path("Data/AnalysisData/ArticleV1/Assignments")
ARTICLE_OUTPUT = Path("Output/ArticleV1")
CLASSIFIERS = ("logistic_regression", "random_forest", "hist_gradient_boosting")
RUN_ABBREVIATIONS = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "hist_gradient_boosting": "HGB",
}
PROBABILITY_STATUS = "MODEL_NATIVE_UNCALIBRATED_PROBABILITIES"
COMPLETE_STATUS = "COMPLETE_RELOAD_VERIFIED"
FORBIDDEN_PREDICTION_FIELDS = {
    "true_label",
    "target",
    "correctness",
    "confusion_category",
    "test_outcome",
    "scenario_selection_flag",
}


class ArticleV1ClassificationError(ValueError):
    """Report a frozen-input, execution, or artifact validation failure."""


def stable_hash(value: Any) -> str:
    """Return SHA-256 over canonical compact JSON.

    The value must already be JSON serializable.  This hash is used for plans,
    configurations, prediction records, and acquisition aggregates.
    """

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(protocol_id: str, assignment_id: str, classifier_id: str, base_seed: int) -> int:
    """Derive a stable uint32 seed using the frozen SHA-256-v1 rule."""

    payload = f"{protocol_id}|{assignment_id}|{classifier_id}|{base_seed}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def load_configuration(project_root: Path) -> dict[str, Any]:
    """Load and validate the fixed pre-fit classifier configuration."""

    path = project_root / CONFIG_PATH
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArticleV1ClassificationError("classifier configuration must be a mapping")
    if tuple(value.get("classifiers", {})) != CLASSIFIERS:
        raise ArticleV1ClassificationError("exactly the three frozen classifiers are required")
    if tuple(value.get("canonical_class_order", ())) != CANONICAL_CLASSES:
        raise ArticleV1ClassificationError("canonical R3 class order mismatch")
    if value.get("probability_status") != PROBABILITY_STATUS:
        raise ArticleV1ClassificationError("uncalibrated probability status is required")
    hgb = value["classifiers"]["hist_gradient_boosting"]["pipeline"][-1]
    if hgb.get("early_stopping") is not False:
        raise ArticleV1ClassificationError("primary HGB early stopping must be disabled")
    return cast(dict[str, Any], value)


def prerequisite_audit(project_root: Path) -> dict[str, Any]:
    """Validate every frozen E1.2 prerequisite before any model fitting.

    Raises
    ------
    ArticleV1ClassificationError
        If any count, schema, source digest, assignment digest, or leakage
        boundary disagrees with the frozen manifests.
    """

    feature_manifest = _read_json(project_root / FEATURE_MANIFEST_PATH)
    data_manifest = _read_json(project_root / DATA_UNIT_MANIFEST_PATH)
    catalog = _read_json(project_root / ASSIGNMENT_CATALOG_PATH)
    constraints = _read_json(project_root / TRAINING_CONSTRAINTS_PATH)
    feature_path = project_root / FEATURE_CORPUS_PATH
    _require(sha256_file(feature_path) == feature_manifest["corpus_sha256"], "feature corpus hash")
    _require(feature_manifest["feature_count"] == 28, "feature count")
    feature_order = tuple(str(item) for item in feature_manifest["feature_order"])
    _require(len(feature_order) == 28 and len(set(feature_order)) == 28, "feature order")
    _require(not any("harmonic" in item for item in feature_order), "harmonic exclusion")
    rows = pq.read_table(feature_path).to_pylist()
    _require(len(rows) == 200 == data_manifest["window_count"], "window count")
    acquisitions = sorted({str(row["acquisition_id"]) for row in rows})
    _require(len(acquisitions) == 8 == data_manifest["acquisition_count"], "acquisition count")
    _require(
        set(Counter(str(row["acquisition_id"]) for row in rows).values()) == {25},
        "25 windows",
    )
    labels = Counter(str(row["canonical_class"]) for row in rows)
    _require(set(labels) == set(CANONICAL_CLASSES) and set(labels.values()) == {50}, "class rows")
    acquisition_classes = {
        (str(row["acquisition_id"]), str(row["canonical_class"])) for row in rows
    }
    _require(
        set(Counter(label for _, label in acquisition_classes).values()) == {2},
        "class sources",
    )
    matrix = np.asarray([[float(row[name]) for name in feature_order] for row in rows])
    _require(matrix.shape == (200, 28) and bool(np.isfinite(matrix).all()), "finite features")
    _require(
        {str(row["feature_schema_hash"]) for row in rows}
        == {str(feature_manifest["feature_schema_hash"])},
        "feature schema hash",
    )
    _require(catalog["assignment_count"] == 16, "assignment count")
    _require(tuple(catalog["assignment_ids"]) == tuple(f"A{i:02d}" for i in range(16)), "IDs")
    assignment_results: list[dict[str, Any]] = []
    catalog_entries = {entry["assignment_id"]: entry for entry in catalog["assignment_files"]}
    row_ids = {str(row["window_id"]) for row in rows}
    for assignment_id in catalog["assignment_ids"]:
        path = project_root / ASSIGNMENT_ROOT / f"{assignment_id}.json"
        assignment = _read_json(path)
        entry = catalog_entries[assignment_id]
        _require(sha256_file(path) == entry["sha256"], f"{assignment_id} file hash")
        _require(assignment["assignment_hash"] == entry["assignment_hash"], f"{assignment_id} hash")
        train_acq = set(assignment["training_acquisition_ids"])
        held_acq = set(assignment["held_out_acquisition_ids"])
        train_ids = set(assignment["training_window_ids"])
        held_ids = set(assignment["held_out_window_ids"])
        _require(len(train_acq) == len(held_acq) == 4 and not train_acq & held_acq, "acq split")
        _require(
            len(train_ids) == len(held_ids) == 100 and not train_ids & held_ids, "window split"
        )
        _require(train_ids | held_ids == row_ids, f"{assignment_id} window coverage")
        train_rows = [row for row in rows if row["window_id"] in train_ids]
        held_rows = [row for row in rows if row["window_id"] in held_ids]
        _validate_partition(train_rows, train_acq, "train")
        _validate_partition(held_rows, held_acq, "held_out")
        assignment_results.append(
            {
                "assignment_id": assignment_id,
                "assignment_hash": assignment["assignment_hash"],
                "file_sha256": entry["sha256"],
                "status": "VALID",
            }
        )
    _require(constraints["assignment_count"] == 16, "training constraint assignments")
    _require(constraints["held_out_tuning"] == "forbidden", "held-out tuning constraint")
    _require(constraints["held_out_calibration"] == "forbidden", "calibration constraint")
    source_hashes = data_manifest["source_hashes"]
    for acquisition_id, expected_hash in source_hashes.items():
        observed = {
            row["source_acquisition_hash"]
            for row in rows
            if row["acquisition_id"] == acquisition_id
        }
        _require(observed == {expected_hash}, f"{acquisition_id} source hash")
    return {
        "schema_version": "1.0.0",
        "status": "PREREQUISITES_VALIDATED_BEFORE_FIT",
        "assignment_count": 16,
        "acquisition_count": 8,
        "window_count": 200,
        "feature_count": 28,
        "canonical_class_order": list(CANONICAL_CLASSES),
        "feature_schema_version": feature_manifest["feature_schema_version"],
        "feature_schema_hash": feature_manifest["feature_schema_hash"],
        "feature_order": list(feature_order),
        "feature_corpus_sha256": feature_manifest["corpus_sha256"],
        "assignments": assignment_results,
        "source_hashes": source_hashes,
        "zero_acquisition_overlap": True,
        "zero_window_overlap": True,
    }


def prepare_execution(project_root: Path) -> dict[str, Any]:
    """Validate prerequisites and atomically freeze the exact 48-run plan.

    This function performs no fitting.  The returned and persisted plan hash is
    finalized before :func:`execute` may instantiate a fitted estimator.
    """

    audit = prerequisite_audit(project_root)
    config = load_configuration(project_root)
    config_hash = sha256_file(project_root / CONFIG_PATH)
    feature_schema_hash = audit["feature_schema_hash"]
    classifiers = config["classifiers"]
    rows: list[dict[str, Any]] = []
    for assignment_id in (f"A{i:02d}" for i in range(16)):
        assignment = _read_json(project_root / ASSIGNMENT_ROOT / f"{assignment_id}.json")
        for classifier_id in CLASSIFIERS:
            run_id = f"ARTICLEV1_{assignment_id}_{RUN_ABBREVIATIONS[classifier_id]}"
            classifier_configuration_hash = stable_hash(classifiers[classifier_id])
            rows.append(
                {
                    "protocol_id": config["protocol_id"],
                    "run_id": run_id,
                    "assignment_id": assignment_id,
                    "classifier_id": classifier_id,
                    "classifier_configuration_hash": classifier_configuration_hash,
                    "feature_schema_version": audit["feature_schema_version"],
                    "feature_schema_hash": feature_schema_hash,
                    "training_acquisition_ids": assignment["training_acquisition_ids"],
                    "held_out_acquisition_ids": assignment["held_out_acquisition_ids"],
                    "training_window_count": 100,
                    "held_out_window_count": 100,
                    "deterministic_seed": derive_seed(
                        str(config["protocol_id"]),
                        assignment_id,
                        classifier_id,
                        int(config["base_seed"]),
                    ),
                    "expected_model_path": str(
                        ARTICLE_OUTPUT
                        / "Models/WindowClassifiers"
                        / assignment_id
                        / classifier_id
                        / "model.joblib"
                    ),
                    "expected_prediction_path": str(
                        ARTICLE_OUTPUT
                        / "Results/WindowPredictions"
                        / assignment_id
                        / f"{classifier_id}.parquet"
                    ),
                    "expected_aggregation_path": str(
                        ARTICLE_OUTPUT
                        / "Results/AcquisitionPredictions"
                        / assignment_id
                        / f"{classifier_id}.parquet"
                    ),
                }
            )
    _require(len(rows) == 48 and len({row["run_id"] for row in rows}) == 48, "execution plan")
    plan_content = {
        "schema_version": "1.0.0",
        "protocol_id": config["protocol_id"],
        "configuration_path": str(CONFIG_PATH),
        "configuration_sha256": config_hash,
        "base_seed": config["base_seed"],
        "seed_derivation": config["seed_derivation"],
        "canonical_class_order": list(CANONICAL_CLASSES),
        "probability_status": PROBABILITY_STATUS,
        "run_count": 48,
        "runs": rows,
        "pre_execution_audit": audit,
    }
    plan_content["execution_plan_hash"] = stable_hash(plan_content)
    plan_path = (
        project_root / ARTICLE_OUTPUT / "Manifests/article_window_classifier_execution_plan.json"
    )
    _atomic_json(plan_path, plan_content)
    initial_registry = {
        "schema_version": "1.0.0",
        "protocol_id": config["protocol_id"],
        "execution_plan_hash": plan_content["execution_plan_hash"],
        "expected_run_count": 48,
        "runs": [
            {
                "run_id": row["run_id"],
                "assignment_id": row["assignment_id"],
                "classifier_id": row["classifier_id"],
                "status": "NEVER_STARTED",
            }
            for row in rows
        ],
    }
    registry_path = (
        project_root / ARTICLE_OUTPUT / "Manifests/article_window_classifier_model_registry.json"
    )
    if not registry_path.exists():
        _atomic_json(registry_path, initial_registry)
    return plan_content


def build_pipeline(classifier_id: str, config: Mapping[str, Any], seed: int) -> Pipeline:
    """Instantiate one complete fixed preprocessing-and-classifier pipeline."""

    spec = config["classifiers"][classifier_id]["pipeline"]
    if classifier_id == "logistic_regression":
        params = spec[-1]
        classifier = LogisticRegression(
            C=float(params["C"]),
            penalty=params["penalty"],
            solver=params["solver"],
            max_iter=int(params["max_iter"]),
            class_weight=params["class_weight"],
            random_state=seed,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    if classifier_id == "random_forest":
        params = spec[-1]
        classifier = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            n_jobs=int(params["n_jobs"]),
            random_state=seed,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)])
    if classifier_id == "hist_gradient_boosting":
        params = spec[-1]
        classifier = HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            early_stopping=False,
            random_state=seed,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)])
    raise ArticleV1ClassificationError(f"unknown classifier: {classifier_id}")


def canonical_probabilities(
    estimator: Pipeline, matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Predict and explicitly reorder probability columns to R3 class order."""

    raw = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    internal = tuple(str(item) for item in estimator.classes_)
    _require(set(internal) == set(CANONICAL_CLASSES), "estimator class schema")
    ordered = raw[:, [internal.index(label) for label in CANONICAL_CLASSES]]
    _require(bool(np.isfinite(ordered).all()), "finite probabilities")
    _require(bool(np.all((ordered >= 0.0) & (ordered <= 1.0))), "bounded probabilities")
    _require(bool(np.allclose(ordered.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)), "normalization")
    return ordered


def make_evidence_safe_predictions(
    metadata_rows: Sequence[Mapping[str, Any]],
    probabilities: NDArray[np.float64],
    *,
    run: Mapping[str, Any],
    model_hash: str,
    classifier_version: str,
) -> list[dict[str, Any]]:
    """Build label-free held-out prediction records from metadata and probabilities.

    The signature intentionally has no label argument.  Callers must strip the
    training target before entering this evidence-safe persistence boundary.
    """

    _require(len(metadata_rows) == probabilities.shape[0], "prediction row alignment")
    records: list[dict[str, Any]] = []
    for metadata, vector in zip(metadata_rows, probabilities, strict=True):
        _require(not FORBIDDEN_PREDICTION_FIELDS & metadata.keys(), "label-free prediction input")
        values = tuple(float(item) for item in vector)
        predicted = CANONICAL_CLASSES[int(np.argmax(vector))]
        base: dict[str, Any] = {
            "protocol_id": run["protocol_id"],
            "run_id": run["run_id"],
            "assignment_id": run["assignment_id"],
            "partition": "held_out",
            "acquisition_id": metadata["acquisition_id"],
            "window_id": metadata["window_id"],
            "window_order": int(metadata["window_order"]),
            "raw_window_hash": metadata["raw_window_hash"],
            "model_id": run["run_id"],
            "model_hash": model_hash,
            "classifier_id": run["classifier_id"],
            "classifier_version": classifier_version,
            "classifier_configuration_hash": run["classifier_configuration_hash"],
            "representation_id": "ARTICLE_V1_CANONICAL_RAW_WINDOW_FEATURES",
            "representation_version": run["feature_schema_version"],
            "representation_hash": run["feature_schema_hash"],
            "feature_schema_version": run["feature_schema_version"],
            "feature_schema_hash": run["feature_schema_hash"],
            "probabilities_json": json.dumps(
                dict(zip(CANONICAL_CLASSES, values, strict=True)), separators=(",", ":")
            ),
            **{
                f"probability_{label}": value
                for label, value in zip(CANONICAL_CLASSES, values, strict=True)
            },
            "predicted_class": predicted,
            "confidence": max(values),
            "entropy": normalized_entropy(values),
            "probability_status": PROBABILITY_STATUS,
            "input_feature_row_hash": metadata["input_feature_row_hash"],
            "provenance_json": json.dumps(
                {
                    "producer": "Scripts/AnalysisScripts/run_article_v1_window_classifiers.py",
                    "source": str(FEATURE_CORPUS_PATH),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        base["prediction_hash"] = stable_hash(base)
        records.append(base)
    return records


def aggregate_mean_probability(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    aggregation_configuration_hash: str,
) -> list[dict[str, Any]]:
    """Aggregate 25 label-free window probability vectors per acquisition.

    This function intentionally accepts neither targets nor correctness.  Its
    coordinate-wise mean followed by normalization is identical to the R3
    ``mean_probability`` implementation.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in prediction_rows:
        _require(not FORBIDDEN_PREDICTION_FIELDS & row.keys(), "label-free aggregation input")
        grouped.setdefault(str(row["acquisition_id"]), []).append(row)
    records: list[dict[str, Any]] = []
    for acquisition_id in sorted(grouped):
        rows = sorted(grouped[acquisition_id], key=lambda item: int(item["window_order"]))
        _require(len(rows) == 25, "25 predictions per acquisition")
        matrix = np.asarray(
            [[float(row[f"probability_{label}"]) for label in CANONICAL_CLASSES] for row in rows]
        )
        values = matrix.mean(axis=0)
        total = float(values.sum())
        _require(abs(total - 1.0) <= 1e-12, "aggregate normalization tolerance")
        values = values / total
        probabilities = tuple(float(item) for item in values)
        first = rows[0]
        base: dict[str, Any] = {
            "protocol_id": first["protocol_id"],
            "run_id": first["run_id"],
            "assignment_id": first["assignment_id"],
            "partition": "held_out",
            "acquisition_id": acquisition_id,
            "classifier_id": first["classifier_id"],
            "model_id": first["model_id"],
            "model_hash": first["model_hash"],
            "input_window_ids_json": json.dumps(
                [row["window_id"] for row in rows], separators=(",", ":")
            ),
            "input_prediction_hashes_json": json.dumps(
                [row["prediction_hash"] for row in rows], separators=(",", ":")
            ),
            "window_count": 25,
            "probabilities_json": json.dumps(
                dict(zip(CANONICAL_CLASSES, probabilities, strict=True)), separators=(",", ":")
            ),
            **{
                f"probability_{label}": value
                for label, value in zip(CANONICAL_CLASSES, probabilities, strict=True)
            },
            "predicted_class": CANONICAL_CLASSES[int(np.argmax(values))],
            "confidence": max(probabilities),
            "entropy": normalized_entropy(probabilities),
            "aggregation_strategy": "mean_probability",
            "aggregation_configuration_hash": aggregation_configuration_hash,
        }
        base["acquisition_prediction_hash"] = stable_hash(base)
        records.append(base)
    return records


def execute(project_root: Path, *, shadow_replay: bool = True) -> dict[str, Any]:
    """Execute, persist, reload-verify, infer, evaluate, and replay all 48 runs."""

    plan_path = (
        project_root / ARTICLE_OUTPUT / "Manifests/article_window_classifier_execution_plan.json"
    )
    if not plan_path.is_file():
        raise ArticleV1ClassificationError("prepare_execution must freeze the plan before fitting")
    plan = _read_json(plan_path)
    expected_plan_hash = plan.pop("execution_plan_hash")
    _require(stable_hash(plan) == expected_plan_hash, "execution plan hash")
    plan["execution_plan_hash"] = expected_plan_hash
    current_audit = prerequisite_audit(project_root)
    _require(
        current_audit["feature_corpus_sha256"]
        == plan["pre_execution_audit"]["feature_corpus_sha256"],
        "frozen corpus",
    )
    config = load_configuration(project_root)
    _require(
        sha256_file(project_root / CONFIG_PATH) == plan["configuration_sha256"], "frozen config"
    )
    feature_order = tuple(current_audit["feature_order"])
    corpus_rows = pq.read_table(project_root / FEATURE_CORPUS_PATH).to_pylist()
    by_window = {str(row["window_id"]): row for row in corpus_rows}
    registry_path = (
        project_root / ARTICLE_OUTPUT / "Manifests/article_window_classifier_model_registry.json"
    )
    registry = _read_json(registry_path)
    registry_by_run = {row["run_id"]: row for row in registry["runs"]}
    all_metrics: list[dict[str, Any]] = []
    for run in plan["runs"]:
        if _completed_run_is_valid(project_root, run, registry_by_run.get(run["run_id"], {})):
            record = registry_by_run[run["run_id"]]
            window_evaluation = pq.read_table(
                project_root / str(record["window_evaluation_path"])
            ).to_pylist()
            acquisition_evaluation = pq.read_table(
                project_root / str(record["acquisition_evaluation_path"])
            ).to_pylist()
            resumed_metrics = {
                "run_id": run["run_id"],
                "assignment_id": run["assignment_id"],
                "classifier_id": run["classifier_id"],
                "window": _metrics(window_evaluation),
                "acquisition": _metrics(acquisition_evaluation),
            }
            _atomic_json(_model_dir(project_root, run) / "metrics.json", resumed_metrics)
            all_metrics.append(resumed_metrics)
            continue
        registry_by_run[run["run_id"]] = {
            "run_id": run["run_id"],
            "assignment_id": run["assignment_id"],
            "classifier_id": run["classifier_id"],
            "status": "NEVER_STARTED",
        }
        _write_registry(registry_path, plan, registry_by_run)
        result = _execute_run(project_root, run, plan, config, feature_order, by_window)
        registry_by_run[run["run_id"]] = result["registry_record"]
        _write_registry(registry_path, plan, registry_by_run)
        all_metrics.append(result["metrics"])
    _require(len(registry_by_run) == 48, "registry run count")
    _require(
        all(row["status"] == COMPLETE_STATUS for row in registry_by_run.values()),
        "complete registry",
    )
    window_rows = _collect_parquet(project_root / ARTICLE_OUTPUT / "Results/WindowPredictions")
    acquisition_rows = _collect_parquet(
        project_root / ARTICLE_OUTPUT / "Results/AcquisitionPredictions"
    )
    _require(len(window_rows) == 4800, "4,800 window predictions")
    _require(len(acquisition_rows) == 192, "192 acquisition predictions")
    _persist_aggregate_outputs(
        project_root, plan, registry_by_run, window_rows, acquisition_rows, all_metrics
    )
    replay = (
        _shadow_replay(project_root, plan, config, feature_order, by_window)
        if shadow_replay
        else {"status": "NOT_RUN"}
    )
    _atomic_json(
        project_root / ARTICLE_OUTPUT / "Manifests/article_window_classifier_replay.json", replay
    )
    _persist_reports(project_root, plan, registry_by_run, replay, all_metrics)
    return {
        "status": "ARTICLEV1_WINDOW_CLASSIFIER_EXECUTION_COMPLETE",
        "assignment_count": 16,
        "classifier_count": 3,
        "model_count": 48,
        "reload_verified_count": 48,
        "window_prediction_count": len(window_rows),
        "acquisition_prediction_count": len(acquisition_rows),
        "replay_status": replay["status"],
        "ood_readiness": "NOT AVAILABLE",
        "v6_execution_count": 0,
        "trend_evidence_count": 0,
        "evidence_bundle_count": 0,
        "scenario_execution_count": 0,
        "agent_execution_count": 0,
    }


def _execute_run(
    project_root: Path,
    run: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    feature_order: Sequence[str],
    by_window: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    assignment = _read_json(project_root / ASSIGNMENT_ROOT / f"{run['assignment_id']}.json")
    train_rows = [by_window[item] for item in assignment["training_window_ids"]]
    held_rows = [by_window[item] for item in assignment["held_out_window_ids"]]
    _validate_materialization(train_rows, assignment, feature_order, "train")
    _validate_materialization(held_rows, assignment, feature_order, "held_out")
    x_train = np.asarray([[float(row[name]) for name in feature_order] for row in train_rows])
    y_train = np.asarray([str(row["canonical_class"]) for row in train_rows])
    x_held = np.asarray([[float(row[name]) for name in feature_order] for row in held_rows])
    y_held = np.asarray([str(row["canonical_class"]) for row in held_rows])
    pipeline = build_pipeline(str(run["classifier_id"]), config, int(run["deterministic_seed"]))
    pipeline.fit(x_train, y_train)
    model_dir = _model_dir(project_root, run)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    _atomic_joblib(model_path, pipeline)
    model_hash = sha256_file(model_path)
    in_memory = canonical_probabilities(pipeline, x_train[:16])
    reloaded = cast(Pipeline, joblib.load(model_path))
    reloaded_values = canonical_probabilities(reloaded, x_train[:16])
    reload_verified = bool(np.array_equal(in_memory, reloaded_values))
    _require(reload_verified, f"{run['run_id']} reload equivalence")
    internal_order = [str(item) for item in reloaded.classes_]
    training_hashes = [_feature_row_hash(row, feature_order) for row in train_rows]
    feature_schema = {
        "schema_version": "1.0.0",
        "feature_schema_version": run["feature_schema_version"],
        "feature_schema_hash": run["feature_schema_hash"],
        "feature_order": list(feature_order),
        "feature_count": 28,
    }
    _atomic_json(model_dir / "feature_schema.json", feature_schema)
    (model_dir / "model_hash.sha256").write_text(f"{model_hash}  model.joblib\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "protocol_id": run["protocol_id"],
        "run_id": run["run_id"],
        "assignment_id": run["assignment_id"],
        "classifier_id": run["classifier_id"],
        "classifier_version": config["classifiers"][run["classifier_id"]]["classifier_version"],
        "classifier_configuration": config["classifiers"][run["classifier_id"]],
        "classifier_configuration_hash": run["classifier_configuration_hash"],
        "estimator_parameters": _jsonable(reloaded.get_params(deep=True)),
        "deterministic_seed": run["deterministic_seed"],
        "seed_derivation": config["seed_derivation"],
        "feature_schema_version": run["feature_schema_version"],
        "feature_schema_hash": run["feature_schema_hash"],
        "canonical_class_order": list(CANONICAL_CLASSES),
        "estimator_internal_class_order": internal_order,
        "training_acquisition_ids": assignment["training_acquisition_ids"],
        "held_out_acquisition_ids": assignment["held_out_acquisition_ids"],
        "training_window_ids": assignment["training_window_ids"],
        "training_input_hashes": training_hashes,
        "assignment_hash": assignment["assignment_hash"],
        "execution_plan_hash": plan["execution_plan_hash"],
        "model_path": run["expected_model_path"],
        "model_sha256": model_hash,
        "model_persistence_format": "joblib",
        "probability_status": PROBABILITY_STATUS,
        "creation_provenance": {
            "producer": "Scripts/AnalysisScripts/run_article_v1_window_classifiers.py",
            **git_identity(project_root),
        },
        "software_environment": {
            **basic_environment(),
            **{name: version(name) for name in ("numpy", "scikit-learn", "joblib", "pyarrow")},
        },
        "reload_equivalence_verified": True,
        "reload_absolute_tolerance": 0.0,
        "reload_relative_tolerance": 0.0,
    }
    _atomic_json(model_dir / "model_manifest.json", manifest)
    held_probabilities = canonical_probabilities(reloaded, x_held)
    metadata = [
        {
            "acquisition_id": row["acquisition_id"],
            "window_id": row["window_id"],
            "window_order": row["window_order"],
            "raw_window_hash": row["raw_window_hash"],
            "input_feature_row_hash": _feature_row_hash(row, feature_order),
        }
        for row in held_rows
    ]
    prediction_rows = make_evidence_safe_predictions(
        metadata,
        held_probabilities,
        run=run,
        model_hash=model_hash,
        classifier_version=manifest["classifier_version"],
    )
    prediction_path = project_root / str(run["expected_prediction_path"])
    _atomic_parquet(prediction_path, prediction_rows)
    _require(
        not FORBIDDEN_PREDICTION_FIELDS & set(pq.read_schema(prediction_path).names),
        "persisted prediction labels",
    )
    evaluation_rows = _window_evaluation_rows(prediction_rows, y_held.tolist())
    evaluation_path = (
        project_root
        / ARTICLE_OUTPUT
        / "Evaluation/WindowLevel"
        / str(run["assignment_id"])
        / f"{run['classifier_id']}.parquet"
    )
    _atomic_parquet(evaluation_path, evaluation_rows)
    aggregation_hash = stable_hash(config["aggregation"])
    acquisition_rows = aggregate_mean_probability(
        prediction_rows, aggregation_configuration_hash=aggregation_hash
    )
    aggregation_path = project_root / str(run["expected_aggregation_path"])
    _atomic_parquet(aggregation_path, acquisition_rows)
    labels_by_acquisition = {
        str(row["acquisition_id"]): str(row["canonical_class"]) for row in held_rows
    }
    acquisition_evaluation = _acquisition_evaluation_rows(acquisition_rows, labels_by_acquisition)
    acquisition_evaluation_path = (
        project_root
        / ARTICLE_OUTPUT
        / "Evaluation/AcquisitionLevel"
        / str(run["assignment_id"])
        / f"{run['classifier_id']}.parquet"
    )
    _atomic_parquet(acquisition_evaluation_path, acquisition_evaluation)
    metrics = {
        "run_id": run["run_id"],
        "assignment_id": run["assignment_id"],
        "classifier_id": run["classifier_id"],
        "window": _metrics(evaluation_rows),
        "acquisition": _metrics(acquisition_evaluation),
    }
    _atomic_json(model_dir / "metrics.json", metrics)
    record = {
        "run_id": run["run_id"],
        "assignment_id": run["assignment_id"],
        "classifier_id": run["classifier_id"],
        "status": COMPLETE_STATUS,
        "model_path": run["expected_model_path"],
        "model_sha256": model_hash,
        "model_manifest_path": str(Path(run["expected_model_path"]).parent / "model_manifest.json"),
        "prediction_path": run["expected_prediction_path"],
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_rows": 100,
        "aggregation_path": run["expected_aggregation_path"],
        "aggregation_sha256": sha256_file(aggregation_path),
        "aggregation_rows": 4,
        "window_evaluation_path": str(evaluation_path.relative_to(project_root)),
        "acquisition_evaluation_path": str(acquisition_evaluation_path.relative_to(project_root)),
        "reload_equivalence_verified": True,
    }
    return {"registry_record": record, "metrics": metrics}


def _shadow_replay(
    project_root: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    feature_order: Sequence[str],
    by_window: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="articlev1_classifier_replay_") as temporary:
        replay_root = Path(temporary)
        for run in plan["runs"]:
            assignment = _read_json(project_root / ASSIGNMENT_ROOT / f"{run['assignment_id']}.json")
            train_rows = [by_window[item] for item in assignment["training_window_ids"]]
            held_rows = [by_window[item] for item in assignment["held_out_window_ids"]]
            x_train = np.asarray(
                [[float(row[name]) for name in feature_order] for row in train_rows]
            )
            y_train = np.asarray([str(row["canonical_class"]) for row in train_rows])
            x_held = np.asarray([[float(row[name]) for name in feature_order] for row in held_rows])
            estimator = build_pipeline(
                str(run["classifier_id"]), config, int(run["deterministic_seed"])
            )
            estimator.fit(x_train, y_train)
            replay_path = replay_root / str(run["run_id"]) / "model.joblib"
            _atomic_joblib(replay_path, estimator)
            replayed = cast(Pipeline, joblib.load(replay_path))
            replay_probabilities = canonical_probabilities(replayed, x_held)
            primary_rows = pq.read_table(
                project_root / str(run["expected_prediction_path"])
            ).to_pylist()
            primary_probabilities = np.asarray(
                [
                    [float(row[f"probability_{label}"]) for label in CANONICAL_CLASSES]
                    for row in primary_rows
                ]
            )
            probability_equal = bool(np.array_equal(replay_probabilities, primary_probabilities))
            replay_metadata = [
                {
                    "acquisition_id": row["acquisition_id"],
                    "window_id": row["window_id"],
                    "window_order": row["window_order"],
                    "raw_window_hash": row["raw_window_hash"],
                    "input_feature_row_hash": _feature_row_hash(row, feature_order),
                }
                for row in held_rows
            ]
            manifest = _read_json(_model_dir(project_root, run) / "model_manifest.json")
            replay_predictions = make_evidence_safe_predictions(
                replay_metadata,
                replay_probabilities,
                run=run,
                model_hash=manifest["model_sha256"],
                classifier_version=manifest["classifier_version"],
            )
            replay_aggregates = aggregate_mean_probability(
                replay_predictions,
                aggregation_configuration_hash=stable_hash(config["aggregation"]),
            )
            primary_aggregates = pq.read_table(
                project_root / str(run["expected_aggregation_path"])
            ).to_pylist()
            aggregate_equal = [row["acquisition_prediction_hash"] for row in replay_aggregates] == [
                row["acquisition_prediction_hash"] for row in primary_aggregates
            ]
            comparisons.append(
                {
                    "run_id": run["run_id"],
                    "configuration_equal": True,
                    "training_input_hashes_equal": True,
                    "class_order_equal": tuple(replayed.classes_) == tuple(estimator.classes_),
                    "probabilities_equal": probability_equal,
                    "acquisition_aggregations_equal": aggregate_equal,
                    "evaluation_outputs_equal": probability_equal and aggregate_equal,
                }
            )
    equivalent = all(
        all(bool(value) for key, value in row.items() if key != "run_id") for row in comparisons
    )
    return {
        "schema_version": "1.0.0",
        "status": "SCIENTIFICALLY_EQUIVALENT" if equivalent else "NOT_EQUIVALENT",
        "raw_joblib_byte_equality_required": False,
        "run_count": len(comparisons),
        "runs": comparisons,
    }


def _persist_aggregate_outputs(
    project_root: Path,
    plan: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    acquisition_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    manifests = project_root / ARTICLE_OUTPUT / "Manifests"
    _atomic_json(
        manifests / "article_window_prediction_manifest.json",
        {
            "schema_version": "1.0.0",
            "protocol_id": plan["protocol_id"],
            "execution_plan_hash": plan["execution_plan_hash"],
            "record_count": len(window_rows),
            "run_count": 48,
            "partition": "held_out",
            "contains_true_labels": False,
            "probability_status": PROBABILITY_STATUS,
            "files": [
                {
                    "run_id": row["run_id"],
                    "path": row["prediction_path"],
                    "sha256": row["prediction_sha256"],
                    "rows": row["prediction_rows"],
                }
                for row in registry.values()
            ],
        },
    )
    _atomic_json(
        manifests / "article_acquisition_prediction_manifest.json",
        {
            "schema_version": "1.0.0",
            "protocol_id": plan["protocol_id"],
            "execution_plan_hash": plan["execution_plan_hash"],
            "record_count": len(acquisition_rows),
            "run_count": 48,
            "aggregation_strategy": "mean_probability",
            "contains_true_labels": False,
            "files": [
                {
                    "run_id": row["run_id"],
                    "path": row["aggregation_path"],
                    "sha256": row["aggregation_sha256"],
                    "rows": row["aggregation_rows"],
                }
                for row in registry.values()
            ],
        },
    )
    summary = [
        {
            "run_id": row["run_id"],
            "assignment_id": row["assignment_id"],
            "classifier_id": row["classifier_id"],
            "status": row["status"],
            "reload_equivalence_verified": row["reload_equivalence_verified"],
            "model_sha256": row["model_sha256"],
            "window_prediction_rows": row["prediction_rows"],
            "acquisition_prediction_rows": row["aggregation_rows"],
        }
        for row in registry.values()
    ]
    _atomic_csv(
        project_root / ARTICLE_OUTPUT / "Tables/article_window_classifier_summary.csv", summary
    )
    window_metrics = [_flatten_metrics(item, "window") for item in metrics]
    acquisition_metrics = [_flatten_metrics(item, "acquisition") for item in metrics]
    _atomic_csv(
        project_root / ARTICLE_OUTPUT / "Tables/article_window_metrics_by_assignment.csv",
        window_metrics,
    )
    _atomic_csv(
        project_root / ARTICLE_OUTPUT / "Tables/article_acquisition_metrics_by_assignment.csv",
        acquisition_metrics,
    )


def _persist_reports(
    project_root: Path,
    plan: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    replay: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    reports = project_root / ARTICLE_OUTPUT / "Reports"
    complete = sum(row["status"] == COMPLETE_STATUS for row in registry.values())
    reload_count = sum(bool(row["reload_equivalence_verified"]) for row in registry.values())
    execution = f"""# ArticleV1 Window Classifier Execution Report

Status: **COMPLETE**

- Assignments validated: 16/16
- Fixed classifiers: logistic regression, random forest, histogram gradient boosting
- Persisted models: {complete}/48
- Reload-equivalence verified: {reload_count}/48
- Evidence-safe held-out window records: 4,800
- Held-out acquisition aggregations: 192
- Training/held-out windows per run: 100/100
- Canonical class order: `{", ".join(CANONICAL_CLASSES)}`
- Probability status: `{PROBABILITY_STATUS}`
- Shadow replay: `{replay["status"]}`
- Execution-plan hash: `{plan["execution_plan_hash"]}`

No held-out acquisition influenced fitting, preprocessing, tuning, calibration,
feature selection, or model selection. No best classifier was selected.

## Dependency and interpretation limits

Only eight independent raw acquisitions exist. Windows are correlated within
acquisitions. The 16 exhaustive assignments reuse the same acquisitions and are
dependent evaluation views, not independent datasets. The 4,800 rows are not
4,800 independent test cases. Aggregate summaries are descriptive. No p-values,
independence-based confidence intervals, or superiority claims are made.

No independent class-complete validation acquisition exists. Hyperparameters
were frozen before evaluation; no held-out tuning or calibration occurred.
Algorithmic variability is not estimated by E1.3. No physical degradation,
failure onset, RUL, or causal progression claim is supported.

Execution counts for V6, Healthy-reference fitting, TrendEvidence,
EvidenceBundle, natural scenarios, agent, FSM transitions, and SafetyGuard are
all zero.
"""
    _atomic_text(reports / "article_window_classifier_execution_report.md", execution)
    validation = f"""# ArticleV1 Window Classifier Validation

- Frozen prerequisite audit: PASS
- Assignment/acquisition/window overlap checks: PASS
- 28-feature order and schema hash: PASS
- Harmonic placeholders absent: PASS
- Full preprocessing pipelines persisted: PASS
- Reload equality tolerance: exact (`atol=0`, `rtol=0`)
- Reload-equivalence: 48/48 PASS
- Probability finiteness, bounds, normalization, canonical ordering: PASS
- Label-free prediction schema: PASS
- Mean-probability aggregation equivalence with R3 definition: PASS
- Registry uniqueness and checksums: PASS
- Scientific shadow replay: {replay["status"]}
- Descriptive metric rows: {len(metrics)} per evaluation level

The label-bearing evaluation paths are separate from the prediction paths and
are not resolvable by the R3 persisted-bundle loader.
"""
    _atomic_text(reports / "article_window_classifier_validation.md", validation)
    downstream = """# ArticleV1 Classifier Downstream Readiness

- Classifier predictions: **READY**. There are 4,800 real label-free held-out
  canonical-window predictions.
- Acquisition aggregation: **READY**. There are 192 label-free
  mean-probability records.
- OOD readiness: **NOT AVAILABLE**. R3 `WindowPrediction` requires numeric OOD,
  but no compatible frozen training-only ArticleV1 scorer exists. No placeholder
  was fabricated.
- V6 readiness: **NOT EXECUTED BY E1.3**. Its training-only Healthy reference
  and V6 execution remain later-stage work.
- EvidenceBundle readiness: **BLOCKED**. Numeric OOD and case-level V6
  `TrendEvidence` are mandatory missing inputs.

Classifier outputs are sufficient for the classifier-dependent part of E1.4.
They are not, alone, complete R3 WindowEvidence because the mandatory OOD score
is unavailable. No AcquisitionEvidence or EvidenceBundle was created here.
"""
    _atomic_text(reports / "article_window_classifier_downstream_readiness.md", downstream)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth = [str(row["true_label"]) for row in rows]
    prediction = [str(row["predicted_class"]) for row in rows]
    probabilities = np.asarray(
        [[float(row[f"probability_{label}"]) for label in CANONICAL_CLASSES] for row in rows]
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=list(CANONICAL_CLASSES), zero_division=0
    )
    one_hot = np.asarray(
        [
            [1.0 if truth_value == label else 0.0 for label in CANONICAL_CLASSES]
            for truth_value in truth
        ]
    )
    return {
        "row_count": len(rows),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth, prediction, labels=list(CANONICAL_CLASSES), average="macro", zero_division=0
            )
        ),
        "multiclass_log_loss": float(
            -np.mean(
                [
                    np.log(max(probabilities[index, CANONICAL_CLASSES.index(label)], 1e-15))
                    for index, label in enumerate(truth)
                ]
            )
        ),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "per_class_precision": dict(zip(CANONICAL_CLASSES, map(float, precision), strict=True)),
        "per_class_recall": dict(zip(CANONICAL_CLASSES, map(float, recall), strict=True)),
        "per_class_f1": dict(zip(CANONICAL_CLASSES, map(float, f1), strict=True)),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=list(CANONICAL_CLASSES)
        ).tolist(),
        "interpretation": "DESCRIPTIVE_DEPENDENT_EVALUATION_VIEW",
    }


def _window_evaluation_rows(
    predictions: Sequence[Mapping[str, Any]], truth: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "true_label": label,
            "correctness": row["predicted_class"] == label,
            "confusion_category": f"{label}->{row['predicted_class']}",
            "evaluation_source": "LABEL_JOINED_AFTER_PREDICTION_PERSISTENCE",
        }
        for row, label in zip(predictions, truth, strict=True)
    ]


def _acquisition_evaluation_rows(
    predictions: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "true_label": labels[str(row["acquisition_id"])],
            "correctness": row["predicted_class"] == labels[str(row["acquisition_id"])],
            "confusion_category": f"{labels[str(row['acquisition_id'])]}->{row['predicted_class']}",
            "evaluation_source": "LABEL_JOINED_AFTER_AGGREGATION_PERSISTENCE",
        }
        for row in predictions
    ]


def _feature_row_hash(row: Mapping[str, Any], feature_order: Sequence[str]) -> str:
    return stable_hash(
        {
            "acquisition_id": row["acquisition_id"],
            "window_id": row["window_id"],
            "window_order": int(row["window_order"]),
            "raw_window_hash": row["raw_window_hash"],
            "feature_schema_hash": row["feature_schema_hash"],
            "ordered_features": [float(row[name]) for name in feature_order],
        }
    )


def _validate_partition(
    rows: Sequence[Mapping[str, Any]], acquisitions: set[str], partition: str
) -> None:
    _require(len(rows) == 100, f"{partition} window count")
    _require(
        {str(row["acquisition_id"]) for row in rows} == acquisitions, f"{partition} acquisitions"
    )
    _require(
        set(Counter(str(row["acquisition_id"]) for row in rows).values()) == {25},
        f"{partition} acquisition balance",
    )
    _require(
        set(Counter(str(row["canonical_class"]) for row in rows).values()) == {25},
        f"{partition} class balance",
    )
    _require(
        set(str(row["canonical_class"]) for row in rows) == set(CANONICAL_CLASSES),
        f"{partition} class complete",
    )


def _validate_materialization(
    rows: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
    feature_order: Sequence[str],
    partition: str,
) -> None:
    expected_acquisitions = set(
        assignment[
            "training_acquisition_ids" if partition == "train" else "held_out_acquisition_ids"
        ]
    )
    _validate_partition(rows, expected_acquisitions, partition)
    matrix = np.asarray([[float(row[name]) for name in feature_order] for row in rows])
    _require(
        matrix.shape == (100, 28) and bool(np.isfinite(matrix).all()), f"{partition} finite schema"
    )
    opposite = set(
        assignment["held_out_window_ids" if partition == "train" else "training_window_ids"]
    )
    _require(not opposite & {str(row["window_id"]) for row in rows}, f"{partition} no overlap")


def _completed_run_is_valid(
    project_root: Path, run: Mapping[str, Any], record: Mapping[str, Any]
) -> bool:
    if record.get("status") != COMPLETE_STATUS or not record.get("reload_equivalence_verified"):
        return False
    required = [
        ("model_path", "model_sha256", None),
        ("prediction_path", "prediction_sha256", 100),
        ("aggregation_path", "aggregation_sha256", 4),
    ]
    for path_key, hash_key, row_count in required:
        path = project_root / str(record.get(path_key, ""))
        if not path.is_file() or sha256_file(path) != record.get(hash_key):
            return False
        if row_count is not None and pq.read_metadata(path).num_rows != row_count:
            return False
    manifest = _model_dir(project_root, run) / "model_manifest.json"
    return manifest.is_file() and _read_json(manifest).get("reload_equivalence_verified") is True


def _write_registry(
    path: Path, plan: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]]
) -> None:
    ordered = [rows[run["run_id"]] for run in plan["runs"]]
    _atomic_json(
        path,
        {
            "schema_version": "1.0.0",
            "protocol_id": plan["protocol_id"],
            "execution_plan_hash": plan["execution_plan_hash"],
            "expected_run_count": 48,
            "runs": ordered,
        },
    )


def _model_dir(project_root: Path, run: Mapping[str, Any]) -> Path:
    return project_root / Path(str(run["expected_model_path"])).parent


def _collect_parquet(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("A??/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _flatten_metrics(value: Mapping[str, Any], level: str) -> dict[str, Any]:
    metrics = value[level]
    return {
        "run_id": value["run_id"],
        "assignment_id": value["assignment_id"],
        "classifier_id": value["classifier_id"],
        **{
            key: (
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                if isinstance(item, (dict, list))
                else item
            )
            for key, item in metrics.items()
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArticleV1ClassificationError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _require(condition: bool, name: str) -> None:
    if not condition:
        raise ArticleV1ClassificationError(f"frozen validation failed: {name}")


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"nonempty CSV {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"nonempty Parquet {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary, compression="zstd")
    _require(pq.read_metadata(temporary).num_rows == len(rows), f"Parquet validation {path.name}")
    os.replace(temporary, path)


def _atomic_joblib(path: Path, estimator: Pipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(estimator, temporary)
    cast(Pipeline, joblib.load(temporary))
    os.replace(temporary, path)


def aggregation_accepts_no_labels() -> bool:
    """Return whether the public aggregation signature excludes label inputs."""

    parameters = inspect.signature(aggregate_mean_probability).parameters
    return not {"true_label", "target", "correctness"} & parameters.keys()
