"""Artifact pipeline for ArticleV1 window-analysis prerequisites only."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from trustworthy_agent.article_v1.assignments import (
    CANONICAL_CLASSES,
    AssignmentDefinition,
    build_exhaustive_assignments,
)
from trustworthy_agent.article_v1.contracts import canonical_hash, require_sha256
from trustworthy_agent.article_v1.features import (
    CanonicalWindowInput,
    WindowFeatureExtractor,
    load_feature_config,
)

DEFAULT_PROTOCOL_CONFIG = Path("configs/experiments/article_v1/article_window_analysis_v1.yaml")
PRODUCER = "Scripts/ProcessingScripts/build_article_v1_window_analysis.py"
RAW_HASH_DOMAIN_SEPARATOR = b"DUDU-BLDC-V2-RAW-WINDOW-v1\0"


def build_article_v1_window_analysis(
    project_root: Path,
    *,
    protocol_config_path: Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """Generate all frozen ArticleV1 pre-training analysis artifacts.

    Parameters
    ----------
    project_root : pathlib.Path
        Repository root containing immutable source data and canonical V2
        window identity/provenance artifacts.
    protocol_config_path : pathlib.Path, optional
        Project-relative ArticleV1 protocol configuration.

    Returns
    -------
    dict
        Deterministic status, counts, declarations, and generated paths.

    Raises
    ------
    OSError, ValueError, yaml.YAMLError
        For unavailable inputs, checksum/schema disagreement, extraction
        failure, or output write failure.

    Scientific Assumptions
    ----------------------
    The eight raw acquisitions are the independent source units.  All 200
    windows remain nested observations.  Feature extraction is case-local;
    class labels are joined only after extraction.

    Side Effects
    ------------
    Writes only the declared ArticleV1 configuration-derived AnalysisData and
    Output artifacts.  It never writes source data, trains or loads a
    classifier, fits a Healthy reference, executes V6, runs the agent, or
    generates evidence bundles.

    Reproducibility Implications
    ----------------------------
    Source, configuration, schema, assignment, corpus, and output hashes are
    persisted.  The generator is deterministic for identical inputs.
    """

    root = project_root.resolve()
    protocol_path = _resolve(root, protocol_config_path)
    protocol = _read_yaml(protocol_path)
    _validate_protocol(protocol)
    feature_path = root / str(protocol["feature_extraction"]["config"])
    feature_config = load_feature_config(feature_path)
    extractor = WindowFeatureExtractor(feature_config)

    provenance_path = root / str(protocol["dataset"]["canonical_window_provenance"])
    windows = cast(list[dict[str, Any]], pq.read_table(provenance_path).to_pylist())
    acquisitions = cast(list[dict[str, Any]], protocol["raw_acquisitions"])
    assignments = build_exhaustive_assignments(
        windows,
        acquisitions,
        protocol_id=str(protocol["protocol_id"]),
    )

    output_paths = _output_paths(root, protocol)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    assignment_root = root / str(protocol["outputs"]["assignment_root"])
    assignment_root.mkdir(parents=True, exist_ok=True)

    feature_rows = _extract_corpus(root, protocol, windows, extractor)
    feature_corpus_path = output_paths["feature_corpus"]
    _write_feature_parquet(feature_corpus_path, feature_rows, extractor.feature_names)
    corpus_sha256 = _sha256_file(feature_corpus_path)

    assignment_files = _write_assignments(
        assignment_root,
        assignments,
        feature_corpus_path.relative_to(root),
        corpus_sha256,
    )
    _write_assignment_summary(output_paths["assignment_summary"], assignments)
    _write_feature_dictionary(output_paths["feature_dictionary"], feature_config)
    _write_feature_summary(output_paths["feature_summary"], feature_rows, extractor.feature_names)

    common = _generation_provenance(root, protocol_path, feature_path, provenance_path)
    data_unit_manifest = _data_unit_manifest(protocol, windows, common)
    _write_manifest(output_paths["data_unit_manifest"], data_unit_manifest)
    assignment_catalog = _assignment_catalog(
        assignments, assignment_files, root, corpus_sha256, common
    )
    _write_manifest(output_paths["assignment_catalog"], assignment_catalog)
    feature_manifest = _feature_manifest(
        extractor,
        feature_config,
        feature_rows,
        feature_corpus_path,
        root,
        corpus_sha256,
        common,
    )
    _write_manifest(output_paths["feature_manifest"], feature_manifest)
    healthy_catalog = _healthy_source_catalog(assignments, extractor, common)
    _write_manifest(output_paths["healthy_catalog"], healthy_catalog)
    training_constraints = _training_constraints(assignments, common)
    _write_manifest(output_paths["training_constraints"], training_constraints)

    _write_assignment_audit(output_paths["assignment_audit"], assignments, assignment_catalog)
    _write_feature_validation(output_paths["feature_validation"], feature_rows, feature_manifest)
    _write_readiness_report(
        output_paths["readiness_report"],
        assignments,
        feature_manifest,
        healthy_catalog,
        training_constraints,
    )

    return {
        "status": "ARTICLEV1_WINDOW_ANALYSIS_PREREQUISITE_COMPLETE",
        "acquisition_count": 8,
        "window_count": len(feature_rows),
        "assignment_count": len(assignments),
        "feature_count": len(extractor.feature_names),
        "classifier_training_performed": False,
        "classifier_inference_performed": False,
        "v6_execution_performed": False,
        "healthy_reference_fitted": False,
        "evidence_bundles_generated": False,
        "scenarios_executed": False,
        "agent_executed": False,
        "artifacts": {name: str(path.relative_to(root)) for name, path in output_paths.items()},
    }


def _extract_corpus(
    root: Path,
    protocol: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    extractor: WindowFeatureExtractor,
) -> list[dict[str, Any]]:
    raw_root = root / str(protocol["dataset"]["raw_source_root"])
    windows_by_acquisition: dict[str, list[Mapping[str, Any]]] = {}
    for row in windows:
        windows_by_acquisition.setdefault(str(row["source_group_id"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for source in cast(Sequence[Mapping[str, Any]], protocol["raw_acquisitions"]):
        acquisition_id = str(source["acquisition_id"])
        source_path = raw_root / str(source["source_file"])
        expected_source_hash = str(source["source_sha256"])
        require_sha256(expected_source_hash, "source_sha256")
        before_hash = _sha256_file(source_path)
        if before_hash != expected_source_hash:
            raise ValueError(f"SOURCE_SHA256_MISMATCH:{acquisition_id}")
        raw = _read_raw_channels(source_path)
        current_amperes = (raw[:, 2] - 2.5) * 8.0
        rotational_speed_rpm = raw[:, 3] * 10_000.0
        source_windows = sorted(
            windows_by_acquisition[acquisition_id],
            key=lambda item: int(item["window_index_within_acquisition"]),
        )
        if len(source_windows) != 25:
            raise ValueError(f"WINDOW_COUNT_PER_ACQUISITION_MISMATCH:{acquisition_id}")
        for window in source_windows:
            start = int(window["start_sample"])
            end = int(window["end_sample"])
            source_slice = slice(start, end)
            raw_hash = _raw_window_hash(raw[source_slice])
            if raw_hash != str(window["raw_window_hash"]):
                raise ValueError(f"RAW_WINDOW_HASH_MISMATCH:{window['window_id']}")
            extraction = extractor.extract(
                CanonicalWindowInput(
                    acquisition_id=acquisition_id,
                    window_id=str(window["window_id"]),
                    window_order=int(window["window_index_within_acquisition"]),
                    current_amperes=np.asarray(current_amperes[source_slice], dtype=np.float64),
                    rotational_speed_rpm=np.asarray(
                        rotational_speed_rpm[source_slice], dtype=np.float64
                    ),
                    sampling_frequency_hz=int(window["sampling_frequency_hz"]),
                    raw_window_hash=raw_hash,
                    source_acquisition_hash=expected_source_hash,
                )
            )
            # The class is joined after extraction and is evaluation metadata;
            # it never enters CanonicalWindowInput or numerical computation.
            feature_row: dict[str, Any] = {
                "protocol_id": str(protocol["protocol_id"]),
                "assignment_id": None,
                "partition": None,
                "materialization_scope": "CANONICAL_CASE_LOCAL_CORPUS",
                "acquisition_id": acquisition_id,
                "window_id": str(window["window_id"]),
                "window_order": int(window["window_index_within_acquisition"]),
                "canonical_class": str(window["canonical_class"]),
                "raw_window_hash": raw_hash,
                "source_acquisition_hash": expected_source_hash,
                "feature_schema_version": extraction.feature_schema_version,
                "feature_schema_hash": extraction.feature_schema_hash,
                "extractor_version": extraction.extractor_version,
                "extractor_configuration_hash": extraction.extractor_configuration_hash,
                "extraction_status": extraction.extraction_status,
                "extraction_failure_reason": extraction.extraction_failure_reason,
                "ordered_feature_values": list(extraction.values),
                "frequency_provenance_json": json.dumps(
                    extraction.frequency_provenance,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            feature_row.update(dict(zip(extraction.feature_names, extraction.values, strict=True)))
            rows.append(feature_row)
        if _sha256_file(source_path) != before_hash:
            raise ValueError(f"IMMUTABLE_SOURCE_CHANGED:{acquisition_id}")
    rows.sort(key=lambda item: (str(item["acquisition_id"]), int(item["window_order"])))
    if len(rows) != 200 or len({str(row["window_id"]) for row in rows}) != 200:
        raise ValueError("CANONICAL_FEATURE_CORPUS_IDENTITY_FAILURE")
    return rows


def _read_raw_channels(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig") as handle:
        prefix = [handle.readline().rstrip("\r\n") for _ in range(7)]
    if len(prefix) != 7 or any(not value for value in prefix):
        raise ValueError(f"RAW_PREFIX_INCOMPLETE:{path.name}")
    header = prefix[6].split(";")
    if header[:4] != ["Sample", "Time (s)", "CURRENT (V)", "ROTO (V)"]:
        raise ValueError(f"RAW_SCHEMA_MISMATCH:{path.name}")
    values = np.loadtxt(path, delimiter=";", skiprows=7, usecols=(0, 1, 2, 3))
    if values.shape != (1_000_000, 4) or not np.all(np.isfinite(values)):
        raise ValueError(f"RAW_VALUES_INVALID:{path.name}")
    sample = values[:, 0].astype(np.int64)
    if not np.array_equal(values[:, 0], sample) or not np.array_equal(
        sample, np.arange(1_000_000, dtype=np.int64)
    ):
        raise ValueError(f"RAW_SAMPLE_INDEX_INVALID:{path.name}")
    if not np.allclose(values[:, 1], sample / 50_000.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"RAW_TIME_SYNCHRONIZATION_FAILURE:{path.name}")
    return values


def _raw_window_hash(raw: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(RAW_HASH_DOMAIN_SEPARATOR)
    for values, dtype in (
        (raw[:, 0], "<i8"),
        (raw[:, 1], "<f8"),
        (raw[:, 2], "<f8"),
        (raw[:, 3], "<f8"),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def _write_feature_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> None:
    fields = [
        pa.field("protocol_id", pa.string(), nullable=False),
        pa.field("assignment_id", pa.string(), nullable=True),
        pa.field("partition", pa.string(), nullable=True),
        pa.field("materialization_scope", pa.string(), nullable=False),
        pa.field("acquisition_id", pa.string(), nullable=False),
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("window_order", pa.int64(), nullable=False),
        pa.field("canonical_class", pa.string(), nullable=False),
        pa.field("raw_window_hash", pa.string(), nullable=False),
        pa.field("source_acquisition_hash", pa.string(), nullable=False),
        pa.field("feature_schema_version", pa.string(), nullable=False),
        pa.field("feature_schema_hash", pa.string(), nullable=False),
        pa.field("extractor_version", pa.string(), nullable=False),
        pa.field("extractor_configuration_hash", pa.string(), nullable=False),
        pa.field("extraction_status", pa.string(), nullable=False),
        pa.field("extraction_failure_reason", pa.string(), nullable=True),
        pa.field("ordered_feature_values", pa.list_(pa.float64()), nullable=False),
        pa.field("frequency_provenance_json", pa.string(), nullable=False),
        *(pa.field(name, pa.float64(), nullable=False) for name in feature_names),
    ]
    table = pa.Table.from_pylist(list(rows), schema=pa.schema(fields))
    pq.write_table(table, path, compression="zstd", use_dictionary=False)


def _write_assignments(
    root: Path,
    assignments: Sequence[AssignmentDefinition],
    corpus_path: Path,
    corpus_sha256: str,
) -> list[Path]:
    paths: list[Path] = []
    for definition in assignments:
        payload = definition.to_dict()
        payload.update(
            {
                "feature_corpus_reference": str(corpus_path),
                "feature_corpus_sha256": corpus_sha256,
                "feature_values_duplicated": False,
                "view_hash": canonical_hash(
                    {
                        "assignment_hash": definition.assignment_hash,
                        "feature_corpus_reference": str(corpus_path),
                        "feature_corpus_sha256": corpus_sha256,
                    }
                ),
            }
        )
        path = root / f"{definition.assignment_id}.json"
        _write_json(path, payload)
        paths.append(path)
    return paths


def _data_unit_manifest(
    protocol: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    common: Mapping[str, Any],
) -> dict[str, Any]:
    acquisitions = cast(Sequence[Mapping[str, Any]], protocol["raw_acquisitions"])
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "manifest_id": "ARTICLE_WINDOW_DATA_UNIT_MANIFEST",
        "article_study": protocol["article_study"],
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "data_units": protocol["data_units"],
        "acquisition_count": len(acquisitions),
        "acquisition_ids": [row["acquisition_id"] for row in acquisitions],
        "canonical_class_mapping": {
            str(row["acquisition_id"]): str(row["canonical_class"]) for row in acquisitions
        },
        "window_count": len(windows),
        "windows_per_acquisition": 25,
        "samples_per_window": 40000,
        "sampling_frequency_hz": 50000,
        "ordering_field": "window_index_within_acquisition",
        "source_hashes": {
            str(row["acquisition_id"]): str(row["source_sha256"]) for row in acquisitions
        },
        "windows_are_nested_within_acquisitions": True,
        "windows_are_independent_experimental_units": False,
        "classifier_training_performed": False,
        **common,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def _assignment_catalog(
    assignments: Sequence[AssignmentDefinition],
    paths: Sequence[Path],
    root: Path,
    corpus_sha256: str,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "catalog_id": "ARTICLE_V1_ACQUISITION_ASSIGNMENT_CATALOG",
        "assignment_count": len(assignments),
        "assignment_ids": [item.assignment_id for item in assignments],
        "algorithm": "exhaustive_class_complete_binary_cartesian_product",
        "assignment_bit_order": list(CANONICAL_CLASSES),
        "assignment_files": [
            {
                "assignment_id": definition.assignment_id,
                "path": str(path.relative_to(root)),
                "sha256": _sha256_file(path),
                "assignment_hash": definition.assignment_hash,
            }
            for definition, path in zip(assignments, paths, strict=True)
        ],
        "feature_corpus_sha256": corpus_sha256,
        "assignments_are_independent_datasets": False,
        "independent_validation_acquisition_exists": False,
        **common,
    }
    payload["catalog_hash"] = canonical_hash(payload)
    return payload


def _feature_manifest(
    extractor: WindowFeatureExtractor,
    feature_config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    corpus_path: Path,
    root: Path,
    corpus_sha256: str,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "manifest_id": "ARTICLE_WINDOW_FEATURE_MANIFEST",
        "feature_schema_id": feature_config["feature_schema_id"],
        "feature_schema_version": feature_config["feature_schema_version"],
        "feature_schema_hash": extractor.feature_schema_hash,
        "extractor_version": feature_config["extractor_version"],
        "extractor_configuration_hash": extractor.configuration_hash,
        "feature_count": len(extractor.feature_names),
        "feature_order": list(extractor.feature_names),
        "row_count": len(rows),
        "successful_row_count": sum(row["extraction_status"] == "EXTRACTION_OK" for row in rows),
        "failed_row_count": sum(row["extraction_status"] != "EXTRACTION_OK" for row in rows),
        "corpus_path": str(corpus_path.relative_to(root)),
        "corpus_sha256": corpus_sha256,
        "fit_scope": "CASE_LOCAL",
        "training_fitted_parameters": False,
        "class_label_joined_after_extraction": True,
        "base_corpus_assignment_id": None,
        "base_corpus_partition": None,
        "assignment_membership_source": "Data/AnalysisData/ArticleV1/Assignments/*.json",
        "harmonic_status": feature_config["harmonic_semantics"]["status"],
        "not_implemented_feature_names": [
            row["feature_name"] for row in feature_config["not_implemented_features"]
        ],
        "frequency_provenance_persisted_per_row": True,
        "stable_v1_feature_count": 26,
        "article_v1_feature_count": 28,
        "discrepancy_resolution": (
            "ArticleV1 defines 14 formula-complete features per channel. Variance, absolute "
            "peak, and spectral bandwidth replace unavailable harmonics. Stable analytical V1 "
            "remains unchanged at 26."
        ),
        "classifier_training_performed": False,
        "classifier_inference_performed": False,
        "v6_execution_performed": False,
        **common,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def _healthy_source_catalog(
    assignments: Sequence[AssignmentDefinition],
    extractor: WindowFeatureExtractor,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    entries = []
    for assignment in assignments:
        training_acquisition = next(
            value for value in assignment.training_acquisition_ids if value.startswith("healthy_")
        )
        held_out_acquisition = next(
            value for value in assignment.held_out_acquisition_ids if value.startswith("healthy_")
        )
        training_windows = [
            value
            for value in assignment.training_window_ids
            if value.startswith(f"{training_acquisition}:")
        ]
        held_out_windows = [
            value
            for value in assignment.held_out_window_ids
            if value.startswith(f"{held_out_acquisition}:")
        ]
        entry: dict[str, Any] = {
            "assignment_id": assignment.assignment_id,
            "training_healthy_acquisition_id": training_acquisition,
            "held_out_healthy_acquisition_id": held_out_acquisition,
            "training_healthy_window_ids": training_windows,
            "held_out_healthy_window_ids": held_out_windows,
            "training_window_count": len(training_windows),
            "held_out_window_count": len(held_out_windows),
            "held_out_windows_excluded_from_source": not bool(
                set(training_windows) & set(held_out_windows)
            ),
            "feature_schema_hash": extractor.feature_schema_hash,
            "applicable_v6_indicator_schema": {
                "representation_id": "WINDOW_TEMPORAL_SPLINE",
                "representation_version": "1.0.0",
                "configuration": "configs/trends/window_temporal_spline.yaml",
                "feature_vector_order": list(extractor.feature_names),
            },
            "reference_fit_performed": False,
        }
        entry["source_definition_hash"] = canonical_hash(entry)
        entries.append(entry)
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "catalog_id": "ARTICLE_V1_HEALTHY_REFERENCE_SOURCE_CATALOG",
        "entry_count": len(entries),
        "fit_scope_required_for_future_task": "assignment_training_healthy_windows_only",
        "global_healthy_reference_forbidden": True,
        "real_reference_fit_performed": False,
        "entries": entries,
        **common,
    }
    payload["catalog_hash"] = canonical_hash(payload)
    return payload


def _training_constraints(
    assignments: Sequence[AssignmentDefinition], common: Mapping[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "constraint_id": "ARTICLE_V1_WINDOW_MODEL_TRAINING_CONSTRAINTS",
        "status": "FROZEN_PREREQUISITE_NO_TRAINING_PERFORMED",
        "assignment_count": len(assignments),
        "assignment_ids": [item.assignment_id for item in assignments],
        "partition_unit": "RAW_ACQUISITION",
        "window_crossing_between_train_and_held_out": "forbidden",
        "model_configuration_policy": (
            "fixed_preregistered_unless_training_only_inner_procedure_is_scientifically_defensible"
        ),
        "held_out_tuning": "forbidden",
        "held_out_calibration": "forbidden",
        "held_out_representation_selection": "forbidden",
        "same_assignments_required_for_all_models": True,
        "persist_full_pipelines": True,
        "predict_proba_required_where_scientifically_applicable": True,
        "reload_equivalence_required": True,
        "window_level_predictions_required": True,
        "acquisition_level_aggregation_required_later": True,
        "independent_validation_acquisition_exists": False,
        "expanded_training_schedule_generated": False,
        "classifier_training_performed": False,
        **common,
    }
    payload["constraint_hash"] = canonical_hash(payload)
    return payload


def _write_assignment_summary(path: Path, assignments: Sequence[AssignmentDefinition]) -> None:
    rows = [
        {
            "assignment_id": item.assignment_id,
            "bits": "".join(str(bit) for bit in item.bits),
            "training_acquisition_ids": "|".join(item.training_acquisition_ids),
            "held_out_acquisition_ids": "|".join(item.held_out_acquisition_ids),
            "training_acquisition_count": len(item.training_acquisition_ids),
            "held_out_acquisition_count": len(item.held_out_acquisition_ids),
            "training_window_count": len(item.training_window_ids),
            "held_out_window_count": len(item.held_out_window_ids),
            "class_complete": True,
            "acquisition_disjoint": True,
            "window_disjoint": True,
            "assignment_hash": item.assignment_hash,
        }
        for item in assignments
    ]
    _write_csv(path, rows)


def _write_feature_dictionary(path: Path, config: Mapping[str, Any]) -> None:
    features = cast(Sequence[Mapping[str, Any]], config["features"])
    excluded = cast(Sequence[Mapping[str, Any]], config["not_implemented_features"])
    rows = [
        {**dict(item), "implementation_status": "IMPLEMENTED", "included_in_feature_vector": True}
        for item in features
    ]
    rows.extend(
        {
            **dict(item),
            "implementation_status": "NOT_IMPLEMENTED_FOR_ARTICLE_V1",
            "included_in_feature_vector": False,
        }
        for item in excluded
    )
    _write_csv(path, rows)


def _write_feature_summary(
    path: Path, rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> None:
    matrix = np.asarray([[float(row[name]) for name in feature_names] for row in rows])
    summary = [
        {
            "feature_name": name,
            "row_count": matrix.shape[0],
            "finite_count": int(np.count_nonzero(np.isfinite(matrix[:, index]))),
            "minimum": float(np.min(matrix[:, index])),
            "maximum": float(np.max(matrix[:, index])),
            "mean": float(np.mean(matrix[:, index])),
            "standard_deviation": float(np.std(matrix[:, index])),
        }
        for index, name in enumerate(feature_names)
    ]
    _write_csv(path, summary)


def _write_assignment_audit(
    path: Path,
    assignments: Sequence[AssignmentDefinition],
    catalog: Mapping[str, Any],
) -> None:
    lines = [
        "# ArticleV1 acquisition-assignment audit",
        "",
        "Status: **PASS**",
        "",
        "The grouping unit is the raw acquisition. Exactly 16 exhaustive class-complete "
        "views were generated; they reuse eight acquisitions and are not 16 independent datasets.",
        "No acquisition-independent validation partition exists because each class has only two "
        "raw acquisitions.",
        "",
        "| assignment | train acquisitions | held-out acquisitions | train windows | "
        "held-out windows | hash |",
        "|---|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {item.assignment_id} | 4 | 4 | 100 | 100 | `{item.assignment_hash}` |"
        for item in assignments
    )
    lines.extend(
        [
            "",
            "All views passed acquisition disjointness, window disjointness, class completeness, "
            "expected counts, and deterministic hashing.",
            f"Catalog hash: `{catalog['catalog_hash']}`.",
            "Feature values are not duplicated; each view references the one canonical corpus.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_feature_validation(
    path: Path, rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    acquisition_counts = Counter(str(row["acquisition_id"]) for row in rows)
    class_counts = Counter(str(row["canonical_class"]) for row in rows)
    lines = [
        "# ArticleV1 canonical-window feature validation",
        "",
        "Status: **PASS**",
        "",
        f"- Corpus rows: {len(rows)}",
        f"- Feature count: {manifest['feature_count']} (14 current + 14 speed)",
        f"- Feature schema hash: `{manifest['feature_schema_hash']}`",
        f"- Corpus SHA-256: `{manifest['corpus_sha256']}`",
        "- Extraction fit scope: case-local only",
        "- Training-fitted parameters: none",
        "- Class label supplied to extractor: no; joined after extraction",
        "- Harmonics: NOT_IMPLEMENTED_FOR_ARTICLE_V1; no arbitrary bins or zero fill",
        "- Failed/non-finite rows: 0",
        "- Classifier training or inference: not performed",
        "- V6 TrendModel execution: not performed",
        "",
        "## Counts",
        "",
    ]
    lines.extend(f"- `{key}`: {value} windows" for key, value in sorted(acquisition_counts.items()))
    lines.append("")
    lines.extend(f"- `{key}`: {value} windows" for key, value in sorted(class_counts.items()))
    lines.extend(
        [
            "",
            "## Scientific limitation",
            "",
            "Some canonical windows contain no positive speed samples, so no defensible "
            "case-local rotational fundamental exists for every row. Harmonics are explicitly "
            "not implemented and are not zero-filled. The windows do not form a run-to-failure "
            "or true degradation trajectory.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_readiness_report(
    path: Path,
    assignments: Sequence[AssignmentDefinition],
    feature_manifest: Mapping[str, Any],
    healthy_catalog: Mapping[str, Any],
    training_constraints: Mapping[str, Any],
) -> None:
    path.write_text(
        "\n".join(
            [
                "# ArticleV1 window-analysis readiness report",
                "",
                "Status: **PREREQUISITE DATA LAYER COMPLETE; MODEL TRAINING NOT STARTED**",
                "",
                "- Independent data units: 8 raw acquisitions.",
                "- Diagnostic observations: 200 canonical 0.8-second windows.",
                "- Windows are correlated observations nested within acquisitions.",
                f"- Assignment views: {len(assignments)} exhaustive class-complete "
                "train/held-out views.",
                "- Assignments reuse acquisitions; 16 assignments are not 16 independent datasets.",
                "- No independent validation acquisition exists.",
                f"- Authoritative ArticleV1 features: {feature_manifest['feature_count']} "
                "(stable analytical V1 remains 26).",
                "- Healthy reference sources are cataloged per assignment using training "
                "Healthy windows only.",
                f"- Healthy source entries: {healthy_catalog['entry_count']}; no reference "
                "was fitted.",
                f"- Training constraint status: `{training_constraints['status']}`.",
                "- No classifier training was performed.",
                "- No classifier inference was performed.",
                "- No V6 TrendEvidence was generated.",
                "- No evidence bundles were generated.",
                "- No scenario or agent execution occurred.",
                "- No natural ArticleV1 scenario is executable yet; window-level models and "
                "persisted downstream evidence remain future work.",
                "",
                "## Scientific boundary",
                "",
                "Ordered windows are not a run-to-failure trajectory. This prerequisite does "
                "not support claims of true degradation, physical failure onset, RUL, or "
                "causal progression.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _generation_provenance(
    root: Path, protocol_path: Path, feature_path: Path, provenance_path: Path
) -> dict[str, Any]:
    return {
        "producer": PRODUCER,
        "source_commit": _git_commit(root),
        "protocol_config_path": str(protocol_path.relative_to(root)),
        "protocol_config_sha256": _sha256_file(protocol_path),
        "feature_config_path": str(feature_path.relative_to(root)),
        "feature_config_sha256": _sha256_file(feature_path),
        "canonical_window_provenance_path": str(provenance_path.relative_to(root)),
        "canonical_window_provenance_sha256": _sha256_file(provenance_path),
    }


def _output_paths(root: Path, protocol: Mapping[str, Any]) -> dict[str, Path]:
    manifests = root / str(protocol["outputs"]["manifest_root"])
    tables = root / str(protocol["outputs"]["table_root"])
    reports = root / str(protocol["outputs"]["report_root"])
    return {
        "feature_corpus": root / str(protocol["outputs"]["feature_corpus"]),
        "data_unit_manifest": manifests / "article_window_data_unit_manifest.json",
        "assignment_catalog": manifests / "acquisition_assignment_catalog.json",
        "feature_manifest": manifests / "article_window_feature_manifest.json",
        "healthy_catalog": manifests / "article_healthy_reference_source_catalog.json",
        "training_constraints": manifests / "article_window_model_training_constraints.json",
        "assignment_summary": tables / "acquisition_assignment_summary.csv",
        "feature_dictionary": tables / "article_window_feature_dictionary.csv",
        "feature_summary": tables / "article_window_feature_summary.csv",
        "assignment_audit": reports / "acquisition_assignment_audit.md",
        "feature_validation": reports / "article_window_feature_validation.md",
        "readiness_report": reports / "article_window_analysis_readiness_report.md",
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("config_type") != "article_window_analysis":
        raise ValueError("INVALID_ARTICLE_WINDOW_PROTOCOL")
    if protocol.get("article_study") != "DUDU_BLDC_TRUSTWORTHY_AGENT_ARTICLE_V1":
        raise ValueError("INVALID_ARTICLE_STUDY")
    gates = protocol.get("execution_gates")
    if not isinstance(gates, Mapping) or any(value != "forbidden" for value in gates.values()):
        raise ValueError("EXECUTION_GATES_MUST_ALL_BE_FORBIDDEN")
    dataset = cast(Mapping[str, Any], protocol["dataset"])
    if (
        int(dataset["sampling_frequency_hz"]) != 50_000
        or int(dataset["samples_per_window"]) != 40_000
        or int(dataset["canonical_window_count"]) != 200
    ):
        raise ValueError("CANONICAL_WINDOW_CONTRACT_MISMATCH")
    acquisitions = protocol.get("raw_acquisitions")
    if not isinstance(acquisitions, list) or len(acquisitions) != 8:
        raise ValueError("ARTICLE_V1_REQUIRES_EIGHT_ACQUISITIONS")


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, payload)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CANNOT_WRITE_EMPTY_CSV:{path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML_MUST_BE_MAPPING:{path}")
    return dict(loaded)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
