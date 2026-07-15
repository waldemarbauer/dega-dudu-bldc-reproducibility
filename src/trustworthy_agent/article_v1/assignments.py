"""Exhaustive acquisition-level assignment definitions for ArticleV1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trustworthy_agent.article_v1.contracts import canonical_hash

CANONICAL_CLASSES = ("Healthy", "Mech_Damage", "Elec_Damage", "Mech_Elec_Damage")
KNOWN_PARTITIONS = ("train", "held_out")
ASSIGNMENT_IDS = tuple(f"A{index:02d}" for index in range(16))


@dataclass(frozen=True)
class AssignmentDefinition:
    """Describe one class-complete acquisition-disjoint train/held-out view."""

    assignment_id: str
    protocol_id: str
    bits: tuple[int, int, int, int]
    training_acquisition_ids: tuple[str, ...]
    held_out_acquisition_ids: tuple[str, ...]
    training_window_ids: tuple[str, ...]
    held_out_window_ids: tuple[str, ...]
    training_class_distribution: dict[str, int]
    held_out_class_distribution: dict[str, int]
    acquisition_distribution: dict[str, dict[str, int]]
    assignment_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return the compact persisted assignment representation."""

        return {
            "schema_version": "1.0.0",
            "assignment_id": self.assignment_id,
            "protocol_id": self.protocol_id,
            "bits": list(self.bits),
            "partition_vocabulary": list(KNOWN_PARTITIONS),
            "training_acquisition_ids": list(self.training_acquisition_ids),
            "held_out_acquisition_ids": list(self.held_out_acquisition_ids),
            "training_window_ids": list(self.training_window_ids),
            "held_out_window_ids": list(self.held_out_window_ids),
            "training_class_distribution": self.training_class_distribution,
            "held_out_class_distribution": self.held_out_class_distribution,
            "acquisition_distribution": self.acquisition_distribution,
            "assignment_hash": self.assignment_hash,
        }


def build_exhaustive_assignments(
    windows: Sequence[Mapping[str, Any]],
    acquisitions: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str,
) -> tuple[AssignmentDefinition, ...]:
    """Build all 16 class-complete binary acquisition assignments.

    Parameters
    ----------
    windows : sequence of mappings
        Canonical window rows containing ``window_id``, ``source_group_id``,
        ``canonical_class``, and acquisition-local ordering.
    acquisitions : sequence of mappings
        Exactly two ordered source acquisitions for each canonical class.
    protocol_id : str
        Frozen ArticleV1 protocol identity.

    Returns
    -------
    tuple of AssignmentDefinition
        A00--A15, where each view has four train and four held-out
        acquisitions and 100 windows in each partition.

    Raises
    ------
    ValueError
        If class balance, identity, nesting, or expected counts are violated.

    Scientific Assumptions
    ----------------------
    Assignments exhaustively reuse eight acquisitions; they are not sixteen
    independent datasets and provide no independent validation acquisition.
    """

    _validate_source_contract(windows, acquisitions)
    by_class: dict[str, list[str]] = {label: [] for label in CANONICAL_CLASSES}
    for source in acquisitions:
        by_class[str(source["canonical_class"])].append(str(source["acquisition_id"]))
    for values in by_class.values():
        values.sort(key=lambda value: _run_number(value, acquisitions))
    windows_by_acquisition: dict[str, list[str]] = {
        str(source["acquisition_id"]): [] for source in acquisitions
    }
    window_class = {str(row["window_id"]): str(row["canonical_class"]) for row in windows}
    for row in sorted(
        windows,
        key=lambda item: (
            str(item["source_group_id"]),
            int(item["window_index_within_acquisition"]),
        ),
    ):
        windows_by_acquisition[str(row["source_group_id"])].append(str(row["window_id"]))

    definitions: list[AssignmentDefinition] = []
    for index, assignment_id in enumerate(ASSIGNMENT_IDS):
        bits = tuple((index >> bit_index) & 1 for bit_index in range(4))
        training = tuple(
            by_class[label][bit] for label, bit in zip(CANONICAL_CLASSES, bits, strict=True)
        )
        held_out = tuple(
            by_class[label][1 - bit] for label, bit in zip(CANONICAL_CLASSES, bits, strict=True)
        )
        training_windows = tuple(
            window_id
            for acquisition in training
            for window_id in windows_by_acquisition[acquisition]
        )
        held_out_windows = tuple(
            window_id
            for acquisition in held_out
            for window_id in windows_by_acquisition[acquisition]
        )
        training_classes = dict(Counter(window_class[value] for value in training_windows))
        held_out_classes = dict(Counter(window_class[value] for value in held_out_windows))
        acquisition_distribution = {
            label: {"train": 1, "held_out": 1} for label in CANONICAL_CLASSES
        }
        payload = {
            "schema_version": "1.0.0",
            "assignment_id": assignment_id,
            "protocol_id": protocol_id,
            "bits": list(bits),
            "partition_vocabulary": list(KNOWN_PARTITIONS),
            "training_acquisition_ids": list(training),
            "held_out_acquisition_ids": list(held_out),
            "training_window_ids": list(training_windows),
            "held_out_window_ids": list(held_out_windows),
            "training_class_distribution": training_classes,
            "held_out_class_distribution": held_out_classes,
            "acquisition_distribution": acquisition_distribution,
        }
        definition = AssignmentDefinition(
            assignment_id=assignment_id,
            protocol_id=protocol_id,
            bits=bits,  # type: ignore[arg-type]
            training_acquisition_ids=training,
            held_out_acquisition_ids=held_out,
            training_window_ids=training_windows,
            held_out_window_ids=held_out_windows,
            training_class_distribution=training_classes,
            held_out_class_distribution=held_out_classes,
            acquisition_distribution=acquisition_distribution,
            assignment_hash=canonical_hash(payload),
        )
        _validate_definition(definition)
        definitions.append(definition)
    return tuple(definitions)


def validate_assignment_rows(
    definitions: Sequence[AssignmentDefinition],
    assignment_id: str,
    partition: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate a materialized assignment view without fitting any transform.

    Rejects unknown assignment/partition, duplicate windows, mixed acquisition
    partition membership, and windows not registered for the selected view.
    """

    by_id = {definition.assignment_id: definition for definition in definitions}
    if assignment_id not in by_id:
        raise ValueError("UNKNOWN_ASSIGNMENT")
    if partition not in KNOWN_PARTITIONS:
        raise ValueError("UNKNOWN_PARTITION")
    definition = by_id[assignment_id]
    allowed_acquisitions = set(
        definition.training_acquisition_ids
        if partition == "train"
        else definition.held_out_acquisition_ids
    )
    allowed_windows = set(
        definition.training_window_ids if partition == "train" else definition.held_out_window_ids
    )
    window_ids = [str(row["window_id"]) for row in rows]
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("DUPLICATE_WINDOW")
    if any(str(row["acquisition_id"]) not in allowed_acquisitions for row in rows):
        raise ValueError("MIXED_ACQUISITION_OR_PARTITION")
    if any(window_id not in allowed_windows for window_id in window_ids):
        raise ValueError("WINDOW_NOT_IN_ASSIGNMENT_PARTITION")


def _run_number(value: str, acquisitions: Sequence[Mapping[str, Any]]) -> int:
    return int(next(row["run_number"] for row in acquisitions if row["acquisition_id"] == value))


def _validate_source_contract(
    windows: Sequence[Mapping[str, Any]], acquisitions: Sequence[Mapping[str, Any]]
) -> None:
    if len(acquisitions) != 8 or len(windows) != 200:
        raise ValueError("ARTICLE_V1_SOURCE_COUNT_MISMATCH")
    ids = [str(row["acquisition_id"]) for row in acquisitions]
    if len(set(ids)) != 8:
        raise ValueError("DUPLICATE_ACQUISITION")
    class_counts = Counter(str(row["canonical_class"]) for row in acquisitions)
    if class_counts != Counter({label: 2 for label in CANONICAL_CLASSES}):
        raise ValueError("ACQUISITION_CLASS_BALANCE_MISMATCH")
    window_ids = [str(row["window_id"]) for row in windows]
    if len(set(window_ids)) != 200:
        raise ValueError("DUPLICATE_WINDOW")
    window_counts = Counter(str(row["source_group_id"]) for row in windows)
    if window_counts != Counter({value: 25 for value in ids}):
        raise ValueError("WINDOW_ACQUISITION_BALANCE_MISMATCH")
    class_by_id = {str(row["acquisition_id"]): str(row["canonical_class"]) for row in acquisitions}
    if any(
        str(row["source_group_id"]) not in class_by_id
        or class_by_id[str(row["source_group_id"])] != str(row["canonical_class"])
        for row in windows
    ):
        raise ValueError("WINDOW_CROSSES_ACQUISITION_BOUNDARY")


def _validate_definition(definition: AssignmentDefinition) -> None:
    if len(definition.training_acquisition_ids) != 4:
        raise ValueError("TRAIN_ACQUISITION_COUNT_MISMATCH")
    if len(definition.held_out_acquisition_ids) != 4:
        raise ValueError("HELD_OUT_ACQUISITION_COUNT_MISMATCH")
    if set(definition.training_acquisition_ids) & set(definition.held_out_acquisition_ids):
        raise ValueError("ACQUISITION_PARTITION_OVERLAP")
    if len(definition.training_window_ids) != 100 or len(definition.held_out_window_ids) != 100:
        raise ValueError("WINDOW_PARTITION_COUNT_MISMATCH")
    if set(definition.training_window_ids) & set(definition.held_out_window_ids):
        raise ValueError("WINDOW_PARTITION_OVERLAP")
    expected = {label: 25 for label in CANONICAL_CLASSES}
    if definition.training_class_distribution != expected:
        raise ValueError("TRAIN_CLASS_COMPLETENESS_FAILURE")
    if definition.held_out_class_distribution != expected:
        raise ValueError("HELD_OUT_CLASS_COMPLETENESS_FAILURE")
