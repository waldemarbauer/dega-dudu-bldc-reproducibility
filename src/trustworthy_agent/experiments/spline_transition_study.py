"""Frozen split protocol for SPLINE_TRANSITION_STUDY_V1."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from trustworthy_agent.config.resolver import resolved_config_hash
from trustworthy_agent.data.splits import SplitProvenance, create_leakage_safe_split
from trustworthy_agent.exceptions import DataQualityError
from trustworthy_agent.experiments.pipeline import (
    ExperimentResult,
    TabularDataset,
    load_dudu_bldc_analysis_dataset,
)
from trustworthy_agent.provenance.hashing import sha256_file

PROTOCOL_ID = "SPLINE_TRANSITION_STUDY_V1"
DEFAULT_CONFIG_PATH = Path("configs/experiments/spline_transition_study_v1.yaml")


def load_protocol_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen split protocol configuration.

    Purpose:
        Read the safe YAML contract for `SPLINE_TRANSITION_STUDY_V1`.
    Parameters:
        path: Project-relative or absolute configuration path.
    Return value:
        JSON-compatible configuration mapping.
    Raised exceptions:
        ValueError when the file is not the expected protocol mapping.
    Scientific assumptions:
        This loader does not execute arbitrary Python from YAML.
    Side effects:
        Reads the configuration file only.
    Reproducibility implications:
        The returned mapping is hashed before split artifacts are persisted.
    """

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Protocol config must be a mapping: {path}")
    config = dict(loaded)
    if config.get("experiment_id") != PROTOCOL_ID:
        raise ValueError(f"Unsupported experiment protocol: {config.get('experiment_id')}")
    split = config.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("Protocol config requires a split mapping.")
    if split.get("fallback_without_verified_groups") != "fail_closed":
        raise ValueError("SPLINE_TRANSITION_STUDY_V1 requires fail-closed group splitting.")
    leakage = config.get("leakage_policy")
    if not isinstance(leakage, Mapping) or leakage.get("forbid_test_fit") is not True:
        raise ValueError("Protocol config must explicitly forbid fitting on the test split.")
    if leakage.get("classifier_training_enabled_in_this_task") is not False:
        raise ValueError("This task freezes splits only; classifier training must remain disabled.")
    _validate_v5_selection_contract(config)
    return config


def _validate_v5_selection_contract(config: Mapping[str, Any]) -> None:
    """Validate that fit-quality and diagnostic V5 selection are not conflated."""

    representations = _mapping(config.get("representations"), "representations")
    fit_selected = _mapping(representations.get("V5_fit_selected"), "V5_fit_selected")
    if fit_selected.get("selection_metric") != "validation_reconstruction_rmse":
        raise ValueError("V5_fit_selected must use validation_reconstruction_rmse.")
    if fit_selected.get("diagnostic_superiority_claim") != "none":
        raise ValueError("V5_fit_selected must not claim diagnostic superiority.")

    diagnostic = _mapping(
        representations.get("V5_diagnostic_selected"),
        "V5_diagnostic_selected",
    )
    expected_candidates = {
        "V1_smoothing_spline_derivatives",
        "V2_b_spline_coefficients",
        "V3_p_spline",
        "V4_healthy_relative_spline",
    }
    if set(diagnostic.get("candidate_spline_representations", ())) != expected_candidates:
        raise ValueError("V5_diagnostic_selected must include V1, V2, V3, and V4 candidates.")
    if diagnostic.get("primary_metric") != "validation_macro_f1":
        raise ValueError("V5_diagnostic_selected primary metric must be validation_macro_f1.")
    if diagnostic.get("secondary_metric") != "validation_balanced_accuracy":
        raise ValueError(
            "V5_diagnostic_selected secondary metric must be validation_balanced_accuracy."
        )
    if diagnostic.get("test_set_use_for_selection") != "forbidden":
        raise ValueError("V5_diagnostic_selected must forbid test-set selection evidence.")
    status = diagnostic.get("status")
    has_validation_results = bool(diagnostic.get("validation_classifier_results_artifact"))
    if status == "pending_classifier_evaluation":
        if diagnostic.get("selected_representation") is not None:
            raise ValueError(
                "V5_diagnostic_selected cannot resolve selected_representation while pending."
            )
        return
    if not has_validation_results:
        raise ValueError(
            "V5_diagnostic_selected cannot be resolved before validation classifier results exist."
        )


def create_frozen_split_artifacts(
    dataset: TabularDataset,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the frozen group-aware split evidence without fitting models.

    Purpose:
        Produce the exact train/validation/test group and row membership for
        `SPLINE_TRANSITION_STUDY_V1`.
    Parameters:
        dataset: Analysis-ready rows with canonical class labels and grouping
            candidates discovered from data.
        config: Resolved protocol configuration.
    Return value:
        Persistable split artifact payload.
    Raised exceptions:
        DataQualityError when the required grouping field is missing or invalid.
    Scientific assumptions:
        Group identity is an experimental leakage-control boundary, not a
        physical degradation trajectory.
    Side effects:
        None.
    Reproducibility implications:
        Records seed, split hash, dataset hash, row identifiers, groups, and
        class distributions.
    """

    split_config = _mapping(config.get("split"), "split")
    required_groups = tuple(
        str(value) for value in split_config.get("required_grouping_fields", ())
    )
    group_columns = _validated_group_columns(dataset, required_groups)
    seed = int(split_config.get("seed", 20260712))
    split = create_leakage_safe_split(
        dataset.rows,
        label_column=dataset.label_column,
        candidate_group_columns=group_columns,
        feature_columns=_numeric_feature_columns(dataset, group_columns),
        seed=seed,
        split_id=str(split_config.get("split_id", "spline_transition_study_v1")),
        train_size=float(split_config.get("train_fraction", 0.60)),
        validation_size=float(split_config.get("validation_fraction", 0.20)),
    )
    if split.algorithm != "deterministic_group_aware_holdout":
        raise DataQualityError("SPLINE_TRANSITION_STUDY_V1 requires verified group-aware split.")
    partitions = _partition_payload(dataset, split, group_columns)
    payload = {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "protocol_version": str(config.get("protocol_version", "1.0.0")),
        "config_hash": resolved_config_hash(config),
        "dataset_id": _mapping(config.get("dataset"), "dataset").get("dataset_id"),
        "dataset_version": _mapping(config.get("dataset"), "dataset").get("dataset_version"),
        "dataset_hash": dataset.dataset_hash(),
        "split": split.to_json_dict(),
        "group_columns": list(group_columns),
        "train_groups": partitions["train"]["groups"],
        "validation_groups": partitions["validation"]["groups"],
        "test_groups": partitions["test"]["groups"],
        "train_row_identifiers": partitions["train"]["row_identifiers"],
        "validation_row_identifiers": partitions["validation"]["row_identifiers"],
        "test_row_identifiers": partitions["test"]["row_identifiers"],
        "class_distributions": {
            "train": partitions["train"]["class_distribution"],
            "validation": partitions["validation"]["class_distribution"],
            "test": partitions["test"]["class_distribution"],
        },
        "group_disjointness": _group_disjointness(partitions),
        "leakage_policy": dict(_mapping(config.get("leakage_policy"), "leakage_policy")),
        "planned_comparison": dict(
            _mapping(config.get("planned_comparison"), "planned_comparison")
        ),
    }
    payload["artifact_hash"] = _stable_json_hash(payload)
    return payload


def persist_frozen_split(
    project_root: Path,
    *,
    config_path: Path | None = None,
    dataset: TabularDataset | None = None,
) -> ExperimentResult:
    """Persist the SPLINE_TRANSITION_STUDY_V1 frozen split artifacts.

    Purpose:
        Materialize the protocol's split JSON and manifest without training any
        classifier or fitting any learned preprocessing.
    Parameters:
        project_root: Repository root used for project-relative paths.
        config_path: Optional protocol config path; defaults to the canonical
            experiment config.
        dataset: Optional in-memory dataset for tests.
    Return value:
        ExperimentResult summarizing the persisted split.
    Raised exceptions:
        File, YAML, and data-quality exceptions propagate to fail closed.
    Scientific assumptions:
        Only split creation is in scope; model comparison remains future work.
    Side effects:
        Writes the required split JSON and manifest under project outputs.
    Reproducibility implications:
        Writes project-relative artifact paths and stable hashes.
    """

    active_config_path = config_path or project_root / DEFAULT_CONFIG_PATH
    if not active_config_path.is_absolute():
        active_config_path = project_root / active_config_path
    config = load_protocol_config(active_config_path)
    active_dataset = dataset or load_dudu_bldc_analysis_dataset(project_root)
    artifact = create_frozen_split_artifacts(active_dataset, config)
    outputs = _mapping(config.get("outputs"), "outputs")
    split_path = project_root / str(outputs["split_json"])
    manifest_path = project_root / str(outputs["split_manifest"])
    split_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "protocol_version": artifact["protocol_version"],
        "config_hash": artifact["config_hash"],
        "dataset_hash": artifact["dataset_hash"],
        "split_id": artifact["split"]["split_id"],
        "split_hash": artifact["split"]["split_hash"],
        "seed": artifact["split"]["seed"],
        "algorithm": artifact["split"]["algorithm"],
        "group_columns": artifact["group_columns"],
        "group_disjointness": artifact["group_disjointness"],
        "class_distributions": artifact["class_distributions"],
        "split_artifact": str(split_path.relative_to(project_root)),
        "split_artifact_hash": sha256_file(split_path),
        "classifier_training_performed": False,
    }
    manifest["artifact_hash"] = _stable_json_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExperimentResult(
        experiment_id=f"{PROTOCOL_ID}_split",
        status="COMPLETED",
        metrics={
            "row_count": len(active_dataset.rows),
            "train_group_count": len(artifact["train_groups"]),
            "validation_group_count": len(artifact["validation_groups"]),
            "test_group_count": len(artifact["test_groups"]),
            "group_disjointness": artifact["group_disjointness"]["all_disjoint"],
        },
        artifacts={
            "split_json": str(split_path.relative_to(project_root)),
            "split_manifest": str(manifest_path.relative_to(project_root)),
        },
        provenance={
            "dataset_hash": artifact["dataset_hash"],
            "split_hash": artifact["split"]["split_hash"],
            "seed": artifact["split"]["seed"],
            "config_hash": artifact["config_hash"],
            "classifier_training_performed": False,
        },
    )


def _validated_group_columns(
    dataset: TabularDataset,
    required_groups: Sequence[str],
) -> tuple[str, ...]:
    available = set(dataset.rows[0]) if dataset.rows else set()
    candidates = tuple(group for group in required_groups if group in available)
    if not candidates:
        raise DataQualityError(
            "Required grouping field is unavailable; refusing non-group-aware split."
        )
    for column in candidates:
        values = [str(row.get(column, "")).strip() for row in dataset.rows]
        if any(not value for value in values) or len(set(values)) <= 1:
            raise DataQualityError(f"Grouping field {column!r} is not valid for group split.")
    return candidates


def _partition_payload(
    dataset: TabularDataset,
    split: SplitProvenance,
    group_columns: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    return {
        "train": _partition(dataset, split.train_indices, group_columns),
        "validation": _partition(dataset, split.validation_indices, group_columns),
        "test": _partition(dataset, split.test_indices, group_columns),
    }


def _partition(
    dataset: TabularDataset,
    indices: Sequence[int],
    group_columns: tuple[str, ...],
) -> dict[str, Any]:
    row_identifiers = [
        _row_identifier(dataset.rows[index], index, group_columns) for index in indices
    ]
    groups = sorted({identifier["group_key"] for identifier in row_identifiers})
    classes = Counter(str(dataset.rows[index].get(dataset.label_column, "")) for index in indices)
    return {
        "indices": list(indices),
        "groups": groups,
        "row_identifiers": row_identifiers,
        "class_distribution": dict(sorted(classes.items())),
    }


def _row_identifier(
    row: Mapping[str, Any],
    row_index: int,
    group_columns: tuple[str, ...],
) -> dict[str, Any]:
    group_values = {column: str(row.get(column, "")) for column in group_columns}
    return {
        "row_index": row_index,
        "case_id": str(row.get("case_id", row_index)),
        "window_id": str(row.get("window_id", row.get("Experiment ID", row_index))),
        "source_file": str(row.get("source_file", "")),
        "class_label": str(row.get("Class", row.get("label", ""))),
        "group_values": group_values,
        "group_key": "|".join(group_values[column] for column in group_columns),
    }


def _group_disjointness(partitions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    train = set(cast(Sequence[str], partitions["train"]["groups"]))
    validation = set(cast(Sequence[str], partitions["validation"]["groups"]))
    test = set(cast(Sequence[str], partitions["test"]["groups"]))
    return {
        "train_validation_overlap": sorted(train & validation),
        "train_test_overlap": sorted(train & test),
        "validation_test_overlap": sorted(validation & test),
        "all_disjoint": not (train & validation or train & test or validation & test),
    }


def _numeric_feature_columns(
    dataset: TabularDataset,
    group_columns: tuple[str, ...],
) -> tuple[str, ...]:
    excluded = {dataset.label_column, *dataset.candidate_group_columns, *group_columns}
    excluded.update({"case_id", "window_id", "source_file"})
    columns: list[str] = []
    for column in sorted(dataset.rows[0] if dataset.rows else ()):
        if column in excluded:
            continue
        values = [_try_float(row.get(column)) for row in dataset.rows]
        if any(value is not None for value in values):
            columns.append(column)
    return tuple(columns)


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    import hashlib

    return hashlib.sha256(encoded).hexdigest()
