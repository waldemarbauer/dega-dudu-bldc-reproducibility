"""Leakage-safe split protocol structures and helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from trustworthy_agent.exceptions import DataQualityError


@dataclass(frozen=True)
class SplitManifest:
    """Persistable split identity without selecting any data."""

    split_id: str
    algorithm: str
    seed: int | None
    split_hash: str


@dataclass(frozen=True)
class DuplicateAnalysis:
    """Duplicate and near-duplicate evidence required for fallback splitting.

    Purpose:
        Record leakage-risk evidence before any non-group-aware split is used.
    Parameters:
        exact_duplicate_rows: Count of repeated feature rows.
        near_duplicate_pairs: Count of row pairs whose numeric feature distance
            is below the configured tolerance.
        tolerance: Absolute numeric tolerance used for near-duplicate detection.
        feature_columns: Columns included in duplicate analysis.
    Return value:
        Immutable duplicate analysis record.
    Raised exceptions:
        None during construction.
    Scientific assumptions:
        Near-duplicate detection is an engineering leakage-risk screen, not a
        claim about physical equivalence.
    Side effects:
        None.
    Reproducibility implications:
        The tolerance and feature columns are persisted with split provenance.
    """

    exact_duplicate_rows: int
    near_duplicate_pairs: int
    tolerance: float
    feature_columns: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable duplicate evidence."""

        return {
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "tolerance": self.tolerance,
            "feature_columns": list(self.feature_columns),
        }


@dataclass(frozen=True)
class SplitProvenance:
    """Exact persisted train/validation/test split evidence.

    Purpose:
        Store deterministic indices and leakage-relevant split metadata.
    Parameters:
        split_id: Stable split identifier.
        algorithm: Split algorithm used.
        seed: Explicit RNG seed; no hidden global RNG is permitted.
        train_indices, validation_indices, test_indices: Exact row indices.
        group_columns: Verified grouping columns used for group-aware splitting.
        grouping_limitation: Explicit limitation when no valid groups exist.
        duplicate_analysis: Duplicate-risk evidence for fallback splits.
        split_hash: Stable hash over all split-defining fields.
    Return value:
        Immutable split provenance.
    Raised exceptions:
        None during construction.
    Scientific assumptions:
        Row indices refer to the immutable analysis table supplied to the splitter.
    Side effects:
        None.
    Reproducibility implications:
        Exact indices and hash make split reuse auditable.
    """

    split_id: str
    algorithm: str
    seed: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    group_columns: tuple[str, ...]
    grouping_limitation: str | None
    duplicate_analysis: DuplicateAnalysis
    split_hash: str

    def to_manifest(self) -> SplitManifest:
        """Return the compact split manifest used by older callers."""

        return SplitManifest(
            split_id=self.split_id,
            algorithm=self.algorithm,
            seed=self.seed,
            split_hash=self.split_hash,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable split provenance."""

        return {
            "split_id": self.split_id,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "test_indices": list(self.test_indices),
            "group_columns": list(self.group_columns),
            "grouping_limitation": self.grouping_limitation,
            "duplicate_analysis": self.duplicate_analysis.to_json_dict(),
            "split_hash": self.split_hash,
        }


def create_leakage_safe_split(
    rows: tuple[Mapping[str, Any], ...],
    *,
    label_column: str,
    candidate_group_columns: tuple[str, ...] = (),
    feature_columns: tuple[str, ...] = (),
    seed: int,
    split_id: str = "holdout_v1",
    train_size: float = 0.6,
    validation_size: float = 0.2,
    near_duplicate_tolerance: float = 1.0e-12,
) -> SplitProvenance:
    """Create a deterministic split that prefers verified group identifiers.

    Purpose:
        Implement the protocol split hierarchy: use group-aware splitting when
        valid identifiers exist; otherwise run duplicate analysis and record the
        stratified fallback limitation.
    Parameters:
        rows: Analysis rows to split.
        label_column: Canonical class-label column.
        candidate_group_columns: Columns discovered from actual data/schema.
        feature_columns: Optional feature columns for duplicate-risk analysis.
        seed: Explicit RNG seed.
        split_id: Stable split identity.
        train_size: Desired training fraction.
        validation_size: Desired validation fraction.
        near_duplicate_tolerance: Tolerance for numeric near-duplicate analysis.
    Return value:
        SplitProvenance with exact indices and split hash.
    Raised exceptions:
        DataQualityError if there are too few rows or labels are missing.
    Scientific assumptions:
        Fallback stratification is a documented limitation when grouping
        identifiers cannot be verified; grouping identifiers are never invented.
    Side effects:
        None.
    Reproducibility implications:
        All randomness is driven by the supplied seed and recorded.
    """

    if len(rows) < 3:
        raise DataQualityError("At least three rows are required for train/validation/test split.")
    labels = _labels(rows, label_column)
    if any(label == "" for label in labels):
        raise DataQualityError(f"Missing labels in required column {label_column!r}.")

    valid_groups = _valid_group_columns(rows, candidate_group_columns)
    duplicate_analysis = analyze_duplicates(
        rows,
        feature_columns=feature_columns,
        tolerance=near_duplicate_tolerance,
    )
    if valid_groups:
        train, validation, test = _group_split(
            rows, valid_groups, seed, train_size, validation_size
        )
        algorithm = "deterministic_group_aware_holdout"
        limitation = None
    else:
        train, validation, test = _stratified_fallback_split(
            labels, seed, train_size, validation_size
        )
        algorithm = "stratified_holdout_without_verified_groups"
        limitation = (
            "No valid grouping identifiers were verified from actual data; duplicate and "
            "near-duplicate leakage-risk analysis was performed before stratified fallback."
        )
    payload: dict[str, Any] = {
        "split_id": split_id,
        "algorithm": algorithm,
        "seed": seed,
        "train_indices": train,
        "validation_indices": validation,
        "test_indices": test,
        "group_columns": valid_groups,
        "grouping_limitation": limitation,
        "duplicate_analysis": duplicate_analysis.to_json_dict(),
    }
    return SplitProvenance(
        split_id=split_id,
        algorithm=algorithm,
        seed=seed,
        train_indices=tuple(train),
        validation_indices=tuple(validation),
        test_indices=tuple(test),
        group_columns=valid_groups,
        grouping_limitation=limitation,
        duplicate_analysis=duplicate_analysis,
        split_hash=_stable_hash(payload),
    )


def analyze_duplicates(
    rows: tuple[Mapping[str, Any], ...],
    *,
    feature_columns: tuple[str, ...],
    tolerance: float,
) -> DuplicateAnalysis:
    """Analyze exact and numeric near-duplicate rows for leakage risk."""

    columns = feature_columns or _numeric_feature_columns(rows, excluded=())
    row_keys: dict[tuple[str, ...], int] = {}
    vectors: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for row in rows:
        key = tuple(_canonical_cell(row.get(column)) for column in columns)
        row_keys[key] = row_keys.get(key, 0) + 1
        vectors.append(
            np.asarray([_finite_float(row.get(column)) for column in columns], dtype=float)
        )
    exact_duplicates = sum(count - 1 for count in row_keys.values() if count > 1)
    near_duplicate_pairs = 0
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            if left.size and bool(np.all(np.abs(left - right) <= tolerance)):
                near_duplicate_pairs += 1
    return DuplicateAnalysis(
        exact_duplicate_rows=exact_duplicates,
        near_duplicate_pairs=near_duplicate_pairs,
        tolerance=tolerance,
        feature_columns=columns,
    )


def _labels(rows: tuple[Mapping[str, Any], ...], label_column: str) -> list[str]:
    return [str(row.get(label_column, "")).strip() for row in rows]


def _valid_group_columns(
    rows: tuple[Mapping[str, Any], ...], candidate_group_columns: tuple[str, ...]
) -> tuple[str, ...]:
    valid: list[str] = []
    for column in candidate_group_columns:
        values = [row.get(column) for row in rows]
        if all(value is not None and str(value).strip() != "" for value in values):
            unique_count = len({str(value) for value in values})
            if 1 < unique_count < len(rows):
                valid.append(column)
    return tuple(valid)


def _group_split(
    rows: tuple[Mapping[str, Any], ...],
    group_columns: tuple[str, ...],
    seed: int,
    train_size: float,
    validation_size: float,
) -> tuple[list[int], list[int], list[int]]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(rows):
        key = tuple(str(row[column]) for column in group_columns)
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(seed)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    train_cut = max(1, int(round(len(group_keys) * train_size)))
    validation_cut = max(
        train_cut + 1, int(round(len(group_keys) * (train_size + validation_size)))
    )
    validation_cut = min(validation_cut, len(group_keys) - 1)
    train_groups = set(group_keys[:train_cut])
    validation_groups = set(group_keys[train_cut:validation_cut])
    test_groups = set(group_keys[validation_cut:])
    if not test_groups:
        test_groups = {group_keys[-1]}
        train_groups.discard(group_keys[-1])
    return (
        sorted(index for key in train_groups for index in groups[key]),
        sorted(index for key in validation_groups for index in groups[key]),
        sorted(index for key in test_groups for index in groups[key]),
    )


def _stratified_fallback_split(
    labels: list[str], seed: int, train_size: float, validation_size: float
) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(len(labels)))
    stratify: list[str] | None = (
        labels if min(labels.count(label) for label in set(labels)) >= 3 else None
    )
    train, remaining, _train_labels, remaining_labels = train_test_split(
        indices,
        labels,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    remaining_fraction = 1.0 - train_size
    validation_fraction = validation_size / remaining_fraction
    remaining_stratify = (
        remaining_labels
        if min(remaining_labels.count(label) for label in set(remaining_labels)) >= 2
        else None
    )
    validation, test = train_test_split(
        remaining,
        train_size=validation_fraction,
        random_state=seed + 1,
        shuffle=True,
        stratify=remaining_stratify,
    )
    return sorted(train), sorted(validation), sorted(test)


def _numeric_feature_columns(
    rows: tuple[Mapping[str, Any], ...], excluded: tuple[str, ...]
) -> tuple[str, ...]:
    if not rows:
        return ()
    excluded_set = set(excluded)
    columns = sorted(set(rows[0]) - excluded_set)
    numeric: list[str] = []
    for column in columns:
        values = [_finite_float(row.get(column)) for row in rows]
        if all(np.isfinite(value) for value in values):
            numeric.append(column)
    return tuple(numeric)


def _finite_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if bool(np.isfinite(number)) else float("nan")


def _canonical_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
