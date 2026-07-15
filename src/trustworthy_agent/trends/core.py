"""Shared validation, evidence construction, and persistence for V6 models."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
from numpy.typing import NDArray

from trustworthy_agent.evidence import TrendEvidence, TrendFitStatus
from trustworthy_agent.trends.contracts import (
    HealthyReference,
    OrderedWindow,
    TrendMode,
    TrendModelFailure,
    WindowPartition,
    canonical_hash,
)


@dataclass(frozen=True)
class FitComputation:
    """Hold fitted values and derivatives returned by one spline algorithm."""

    fitted: NDArray[np.float64]
    first_derivative: NDArray[np.float64]
    second_derivative: NDArray[np.float64]
    effective_degrees_of_freedom: float


class BaseTrendModel(ABC):
    """Enforce the common V6 lifecycle and scientific safety boundaries.

    Parameters
    ----------
    config : Mapping of str to Any
        Resolved deterministic configuration. Required common fields are
        ``mode``, ``min_windows``, ``degree``, and ``feature_reduction``.

    Raises
    ------
    TrendModelFailure
        If configuration is invalid. The exception contains structured failed
        ``TrendEvidence`` with reason, metadata, and provenance.

    Scientific Assumptions
    ----------------------
    Every fit is local to ordered windows from one acquisition. The normalized
    ordering axis carries no degradation-time or remaining-life semantics.
    """

    model_name = "BaseTrendModel"
    model_version = "1.0.0"
    requires_healthy_reference = False

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._configuration_hash = canonical_hash(
            {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "parameters": self._config,
            }
        )
        self._windows: tuple[OrderedWindow, ...] = ()
        self._healthy_reference: HealthyReference | None = None
        self._validate_config()

    def fit(
        self,
        windows: Sequence[OrderedWindow],
        *,
        healthy_reference: HealthyReference | None = None,
    ) -> Self:
        """Validate and retain ordered windows from exactly one acquisition.

        Parameters
        ----------
        windows : Sequence of OrderedWindow
            Canonically ordered windows sharing acquisition and assignment.
        healthy_reference : HealthyReference or None, optional
            Split-local reference. When supplied it must originate exclusively
            from the active assignment's training partition.

        Returns
        -------
        model : BaseTrendModel
            The same fitted instance for fluent lifecycle use.

        Raises
        ------
        TrendModelFailure
            For mixed acquisitions, unordered or duplicate windows, non-finite
            values, inconsistent positions, or inadmissible Healthy provenance.

        Side Effects
        ------------
        Retains an immutable tuple of validated windows and reference in memory.
        Source windows and external artifacts are not modified.
        """

        supplied = tuple(windows)
        fit_boundary = (
            max((window.window_order for window in supplied), default=None)
            if self._mode is TrendMode.ONLINE
            else None
        )
        validated = self._validate_windows(supplied, observed_through=fit_boundary)
        self._validate_healthy_reference(healthy_reference, validated)
        self._windows = validated
        self._healthy_reference = healthy_reference
        return self

    def transform(
        self,
        windows: Sequence[OrderedWindow] | None = None,
        *,
        observed_through: int | None = None,
    ) -> TrendEvidence:
        """Compute trend evidence from complete or already-observed windows.

        Parameters
        ----------
        windows : Sequence of OrderedWindow or None, optional
            Explicit sequence to transform. ``None`` reuses the fitted history.
        observed_through : int or None, optional
            Highest observed ``window_order`` in ONLINE mode. It is forbidden
            in OFFLINE mode and mandatory in ONLINE mode.

        Returns
        -------
        evidence : TrendEvidence
            Successful metrics or explicit ``INSUFFICIENT_HISTORY`` or
            ``NUMERICAL_FAILURE`` evidence.

        Raises
        ------
        TrendModelFailure
            For invalid lifecycle, structure, future windows, non-finite input,
            or Healthy reference provenance.

        Notes
        -----
        ONLINE transformation never filters future rows silently; their
        presence rejects the operation.
        """

        active = self._windows if windows is None else tuple(windows)
        if not active:
            raise self._exception(
                TrendFitStatus.INSUFFICIENT_HISTORY,
                "MODEL_NOT_FITTED",
                active,
                metadata={"minimum_window_count": self._min_windows},
            )
        validated = self._validate_windows(active, observed_through=observed_through)
        self._validate_healthy_reference(self._healthy_reference, validated)
        if len(validated) < self._min_windows:
            return self._failure_evidence(
                TrendFitStatus.INSUFFICIENT_HISTORY,
                "INSUFFICIENT_HISTORY",
                validated,
                metadata={"minimum_window_count": self._min_windows},
            )

        x = np.linspace(0.0, 1.0, len(validated), dtype=float)
        signal = self._diagnostic_signal(validated)
        try:
            computation = self._fit_arrays(x, signal)
            return self._success_evidence(validated, signal, computation)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            return self._failure_evidence(
                TrendFitStatus.NUMERICAL_FAILURE,
                "NUMERICAL_FAILURE",
                validated,
                metadata={"exception_type": type(exc).__name__, "message": str(exc)},
            )

    def update(
        self,
        window: OrderedWindow,
        *,
        observed_through: int | None = None,
    ) -> TrendEvidence:
        """Append one already-observed online window and recompute evidence.

        Parameters
        ----------
        window : OrderedWindow
            Next window in strictly increasing canonical order.
        observed_through : int or None, optional
            Highest order currently observable. Required in ONLINE mode.

        Returns
        -------
        evidence : TrendEvidence
            Evidence over all retained history through the new window.

        Raises
        ------
        TrendModelFailure
            If used in OFFLINE mode or if the new window is duplicate,
            unordered, mixed-acquisition, or beyond the observation boundary.

        Side Effects
        ------------
        Replaces the in-memory immutable history only after validation passes.
        """

        if self._mode is not TrendMode.ONLINE:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "UPDATE_REQUIRES_ONLINE_MODE",
                (*self._windows, window),
            )
        candidate = (*self._windows, window)
        validated = self._validate_windows(candidate, observed_through=observed_through)
        self._validate_healthy_reference(self._healthy_reference, validated)
        self._windows = validated
        return self.transform(observed_through=observed_through)

    def save(self, path: Path) -> Path:
        """Persist configuration, history, and reference as deterministic JSON.

        Parameters
        ----------
        path : pathlib.Path
            Destination JSON path. Parent directories are created explicitly.

        Returns
        -------
        saved_path : pathlib.Path
            The supplied path after a successful deterministic JSON write.

        Raises
        ------
        OSError
            If the destination cannot be created or written.

        Security Notes
        --------------
        Persistence uses JSON only; executable pickle deserialization is not
        used.
        """

        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "configuration": self._config,
            "configuration_hash": self._configuration_hash,
            "windows": [window.to_dict() for window in self._windows],
            "healthy_reference": (
                None if self._healthy_reference is None else self._healthy_reference.to_dict()
            ),
        }
        payload["artifact_hash"] = canonical_hash(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore a model from deterministic JSON and revalidate all content.

        Parameters
        ----------
        path : pathlib.Path
            JSON artifact produced by :meth:`save` for the same model class.

        Returns
        -------
        model : BaseTrendModel
            Reconstructed model with validated configuration and history.

        Raises
        ------
        OSError, json.JSONDecodeError
            If persistence cannot be read or parsed.
        TrendModelFailure
            If class identity, hashes, configuration, windows, or Healthy
            provenance fail validation.
        """

        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        supplied_hash = str(payload.pop("artifact_hash", ""))
        configuration = cast(dict[str, Any], payload["configuration"])
        model = cls(configuration)
        if payload.get("model_name") != cls.model_name or payload.get("model_version") != (
            cls.model_version
        ):
            raise model._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "SERIALIZED_MODEL_IDENTITY_MISMATCH",
                (),
            )
        if supplied_hash != canonical_hash(payload):
            raise model._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "SERIALIZED_ARTIFACT_HASH_MISMATCH",
                (),
            )
        if payload.get("configuration_hash") != model._configuration_hash:
            raise model._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "SERIALIZED_CONFIGURATION_HASH_MISMATCH",
                (),
            )
        windows = tuple(
            OrderedWindow.from_dict(cast(dict[str, Any], item)) for item in payload["windows"]
        )
        reference_data = payload.get("healthy_reference")
        reference = (
            None
            if reference_data is None
            else HealthyReference.from_dict(cast(dict[str, Any], reference_data))
        )
        if windows:
            model.fit(windows, healthy_reference=reference)
        elif reference is not None:
            raise model._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "REFERENCE_WITHOUT_WINDOWS",
                (),
            )
        return model

    def schema(self) -> dict[str, Any]:
        """Return closed schemas for OrderedWindow and TrendEvidence.

        Returns
        -------
        schema : dict of str to Any
            Versioned JSON-Schema-compatible contract mapping.
        """

        window_properties: dict[str, Any] = {
            "acquisition_id": {"type": "string", "minLength": 1},
            "window_id": {"type": "string", "minLength": 1},
            "window_order": {"type": "integer", "minimum": 0},
            "assignment_id": {"type": "string", "minLength": 1},
            "partition": {"enum": [partition.value for partition in WindowPartition]},
            "features": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "number"},
            },
            "timestamp": {"type": ["string", "null"]},
            "ordinal_position": {"type": ["number", "null"]},
        }
        return {
            "schema_version": "1.0",
            "ordered_window": {
                "type": "object",
                "additionalProperties": False,
                "required": list(window_properties),
                "properties": window_properties,
                "anyOf": [
                    {"properties": {"timestamp": {"type": "string"}}},
                    {"properties": {"ordinal_position": {"type": "number"}}},
                ],
            },
            "trend_evidence": TrendEvidence.schema(),
        }

    def get_metadata(self) -> dict[str, Any]:
        """Return stable model identity, mode, fit, and reference metadata.

        Returns
        -------
        metadata : dict of str to Any
            JSON-compatible lifecycle metadata without volatile timestamps or
            machine paths.
        """

        return {
            "representation_family": "V6",
            "representation_id": TrendEvidence.REPRESENTATION_ID,
            "representation_version": TrendEvidence.REPRESENTATION_VERSION,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "mode": self._mode.value,
            "configuration_hash": self._configuration_hash,
            "window_count": len(self._windows),
            "acquisition_id": self._windows[0].acquisition_id if self._windows else None,
            "assignment_id": self._windows[0].assignment_id if self._windows else None,
            "healthy_reference_id": (
                None if self._healthy_reference is None else self._healthy_reference.reference_id
            ),
            "ordering_semantics": "normalized_ordered_window_position",
            "scientific_claim_scope": "ordered_diagnostic_evolution_within_one_acquisition",
        }

    @abstractmethod
    def _fit_arrays(
        self,
        x: NDArray[np.float64],
        signal: NDArray[np.float64],
    ) -> FitComputation:
        """Fit one model and return values and derivatives on observed positions."""

    def _diagnostic_signal(self, windows: Sequence[OrderedWindow]) -> NDArray[np.float64]:
        matrix = np.asarray([window.features for window in windows], dtype=float)
        if self._feature_reduction == "mean":
            return cast(NDArray[np.float64], np.mean(matrix, axis=1))
        if self._feature_reduction == "l2_norm":
            return cast(NDArray[np.float64], np.linalg.norm(matrix, axis=1))
        raise self._exception(
            TrendFitStatus.INVALID_CONFIGURATION,
            "UNSUPPORTED_FEATURE_REDUCTION",
            windows,
        )

    def _success_evidence(
        self,
        windows: Sequence[OrderedWindow],
        signal: NDArray[np.float64],
        computation: FitComputation,
    ) -> TrendEvidence:
        arrays = (
            computation.fitted,
            computation.first_derivative,
            computation.second_derivative,
        )
        if any(array.shape != signal.shape for array in arrays) or any(
            not np.all(np.isfinite(array)) for array in arrays
        ):
            raise ValueError("Spline computation returned invalid array shape or values.")
        curvature = np.abs(computation.second_derivative) / np.power(
            1.0 + np.square(computation.first_derivative), 1.5
        )
        residual = signal - computation.fitted
        fit_rmse = float(np.sqrt(np.mean(np.square(residual))))
        x = np.linspace(0.0, 1.0, len(windows), dtype=float)
        roughness = float(np.trapezoid(np.square(computation.second_derivative), x))
        residual_dof = max(float(len(windows)) - computation.effective_degrees_of_freedom, 1.0)
        uncertainty = float(np.sqrt(np.sum(np.square(residual)) / residual_dof))
        return TrendEvidence(
            trend_value=float(computation.fitted[-1]),
            first_derivative=float(computation.first_derivative[-1]),
            second_derivative=float(computation.second_derivative[-1]),
            curvature=float(curvature[-1]),
            maximum_curvature=float(np.max(curvature)),
            healthy_distance=self._healthy_distance(windows[-1]),
            fit_rmse=fit_rmse,
            roughness=roughness,
            effective_degrees_of_freedom=float(computation.effective_degrees_of_freedom),
            uncertainty=uncertainty,
            window_count=len(windows),
            fit_status=TrendFitStatus.FIT_OK,
            representation_id=TrendEvidence.REPRESENTATION_ID,
            representation_version=TrendEvidence.REPRESENTATION_VERSION,
            configuration_hash=self._configuration_hash,
            input_window_hashes=tuple(window.stable_hash() for window in windows),
            metadata={
                "model_name": self.model_name,
                "model_version": self.model_version,
                "mode": self._mode.value,
                "feature_reduction": self._feature_reduction,
                "derivative_axis": "normalized_ordered_window_position",
            },
            provenance=self._provenance(windows),
        )

    def _failure_evidence(
        self,
        status: TrendFitStatus,
        reason: str,
        windows: Sequence[OrderedWindow],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrendEvidence:
        active_mode = getattr(self, "_mode", None)
        mode_value = (
            active_mode.value if isinstance(active_mode, TrendMode) else self._config.get("mode")
        )
        return TrendEvidence(
            trend_value=None,
            first_derivative=None,
            second_derivative=None,
            curvature=None,
            maximum_curvature=None,
            healthy_distance=None,
            fit_rmse=None,
            roughness=None,
            effective_degrees_of_freedom=None,
            uncertainty=None,
            window_count=len(windows),
            fit_status=status,
            representation_id=TrendEvidence.REPRESENTATION_ID,
            representation_version=TrendEvidence.REPRESENTATION_VERSION,
            configuration_hash=self._configuration_hash,
            input_window_hashes=tuple(window.stable_hash() for window in windows),
            reason=reason,
            metadata={
                "model_name": self.model_name,
                "model_version": self.model_version,
                "mode": mode_value,
                **dict(metadata or {}),
            },
            provenance=self._provenance(windows),
        )

    def _exception(
        self,
        status: TrendFitStatus,
        reason: str,
        windows: Sequence[OrderedWindow],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrendModelFailure:
        return TrendModelFailure(self._failure_evidence(status, reason, windows, metadata=metadata))

    def _validate_config(self) -> None:
        try:
            self._mode = TrendMode(str(self._config["mode"]))
            self._min_windows = int(self._config["min_windows"])
            self._degree = int(self._config["degree"])
            self._feature_reduction = str(self._config["feature_reduction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "INVALID_COMMON_CONFIGURATION",
                (),
                metadata={"message": str(exc)},
            ) from exc
        if self._min_windows < 3:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "MIN_WINDOWS_MUST_BE_AT_LEAST_THREE",
                (),
            )
        if self._degree < 1 or self._degree > 3:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "DEGREE_OUT_OF_RANGE",
                (),
            )
        if self._min_windows <= self._degree:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "MIN_WINDOWS_MUST_EXCEED_DEGREE",
                (),
            )
        if self._feature_reduction not in {"mean", "l2_norm"}:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "UNSUPPORTED_FEATURE_REDUCTION",
                (),
            )

    def _validate_windows(
        self,
        windows: Sequence[OrderedWindow],
        *,
        observed_through: int | None,
    ) -> tuple[OrderedWindow, ...]:
        active = tuple(windows)
        if not active:
            return active
        if len({window.acquisition_id for window in active}) != 1:
            raise self._exception(
                TrendFitStatus.MIXED_ACQUISITIONS,
                "MIXED_ACQUISITIONS",
                active,
            )
        if len({window.assignment_id for window in active}) != 1:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "MIXED_ASSIGNMENTS",
                active,
            )
        partitions = {window.partition for window in active}
        if any(
            window.partition not in {partition.value for partition in WindowPartition}
            for window in active
        ):
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "UNKNOWN_WINDOW_PARTITION",
                active,
            )
        if len(partitions) != 1:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "MIXED_WINDOW_PARTITIONS",
                active,
            )
        window_ids = [window.window_id for window in active]
        orders = [window.window_order for window in active]
        if len(set(window_ids)) != len(window_ids):
            raise self._exception(
                TrendFitStatus.UNORDERED_WINDOWS,
                "DUPLICATE_WINDOW_ID",
                active,
            )
        if len(set(orders)) != len(orders):
            raise self._exception(
                TrendFitStatus.UNORDERED_WINDOWS,
                "DUPLICATE_WINDOW_ORDER",
                active,
            )
        if any(
            current >= following for current, following in zip(orders, orders[1:], strict=False)
        ):
            raise self._exception(
                TrendFitStatus.UNORDERED_WINDOWS,
                "UNORDERED_WINDOWS",
                active,
            )
        feature_lengths = {len(window.features) for window in active}
        if feature_lengths == {0} or len(feature_lengths) != 1:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "INCONSISTENT_FEATURE_VECTOR_SHAPE",
                active,
            )
        if any(not math.isfinite(float(value)) for window in active for value in window.features):
            raise self._exception(
                TrendFitStatus.NONFINITE_INPUT,
                "NONFINITE_INPUT",
                active,
            )
        self._validate_positions(active)
        if self._mode is TrendMode.ONLINE:
            if observed_through is None:
                raise self._exception(
                    TrendFitStatus.INVALID_CONFIGURATION,
                    "ONLINE_OBSERVATION_BOUNDARY_REQUIRED",
                    active,
                )
            if any(window.window_order > observed_through for window in active):
                raise self._exception(
                    TrendFitStatus.UNORDERED_WINDOWS,
                    "FUTURE_WINDOW_FORBIDDEN",
                    active,
                    metadata={"observed_through": observed_through},
                )
        elif observed_through is not None:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "OFFLINE_MODE_FORBIDS_OBSERVATION_BOUNDARY",
                active,
            )
        return active

    def _validate_positions(self, windows: Sequence[OrderedWindow]) -> None:
        if all(window.ordinal_position is not None for window in windows):
            positions = [cast(float, window.ordinal_position) for window in windows]
            if any(not math.isfinite(value) for value in positions) or any(
                current >= following
                for current, following in zip(positions, positions[1:], strict=False)
            ):
                raise self._exception(
                    TrendFitStatus.UNORDERED_WINDOWS,
                    "UNORDERED_ORDINAL_POSITIONS",
                    windows,
                )
            return
        if all(window.timestamp is not None for window in windows):
            try:
                timestamps = [
                    datetime.fromisoformat(cast(str, window.timestamp)) for window in windows
                ]
                if any(
                    current >= following
                    for current, following in zip(timestamps, timestamps[1:], strict=False)
                ):
                    raise ValueError("timestamps must be strictly increasing")
            except (TypeError, ValueError) as exc:
                raise self._exception(
                    TrendFitStatus.UNORDERED_WINDOWS,
                    "INVALID_OR_UNORDERED_TIMESTAMPS",
                    windows,
                    metadata={"message": str(exc)},
                ) from exc
            return
        raise self._exception(
            TrendFitStatus.INVALID_CONFIGURATION,
            "CONSISTENT_TIMESTAMP_OR_ORDINAL_POSITION_REQUIRED",
            windows,
        )

    def _validate_healthy_reference(
        self,
        reference: HealthyReference | None,
        windows: Sequence[OrderedWindow],
    ) -> None:
        if reference is None:
            if self.requires_healthy_reference:
                raise self._exception(
                    TrendFitStatus.UNKNOWN_HEALTHY_REFERENCE,
                    "HEALTHY_REFERENCE_REQUIRED",
                    windows,
                )
            return
        if reference.source_partitions != (WindowPartition.TRAIN.value,):
            raise self._exception(
                TrendFitStatus.UNKNOWN_HEALTHY_REFERENCE,
                "HEALTHY_REFERENCE_MUST_BE_TRAINING_ONLY",
                windows,
                metadata={"source_partitions": list(reference.source_partitions)},
            )
        if not reference.assignment_id or (
            windows and reference.assignment_id != windows[0].assignment_id
        ):
            raise self._exception(
                TrendFitStatus.UNKNOWN_HEALTHY_REFERENCE,
                "HEALTHY_REFERENCE_ASSIGNMENT_MISMATCH",
                windows,
            )
        expected_size = len(windows[0].features) if windows else len(reference.feature_mean)
        if (
            len(reference.feature_mean) != expected_size
            or len(reference.feature_scale) != expected_size
            or not reference.source_window_hashes
        ):
            raise self._exception(
                TrendFitStatus.UNKNOWN_HEALTHY_REFERENCE,
                "HEALTHY_REFERENCE_PROVENANCE_OR_SHAPE_INVALID",
                windows,
            )
        if any(not math.isfinite(value) for value in reference.feature_mean) or any(
            not math.isfinite(value) or value <= 0.0 for value in reference.feature_scale
        ):
            raise self._exception(
                TrendFitStatus.UNKNOWN_HEALTHY_REFERENCE,
                "HEALTHY_REFERENCE_VALUES_INVALID",
                windows,
            )
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in reference.source_window_hashes
        ):
            raise self._exception(
                TrendFitStatus.UNKNOWN_HEALTHY_REFERENCE,
                "HEALTHY_REFERENCE_SOURCE_HASH_INVALID",
                windows,
            )

    def _healthy_distance(self, window: OrderedWindow) -> float | None:
        if self._healthy_reference is None:
            return None
        mean = np.asarray(self._healthy_reference.feature_mean, dtype=float)
        scale = np.asarray(self._healthy_reference.feature_scale, dtype=float)
        values = np.asarray(window.features, dtype=float)
        return float(np.sqrt(np.mean(np.square((values - mean) / scale))))

    def _provenance(self, windows: Sequence[OrderedWindow]) -> dict[str, Any]:
        return {
            "acquisition_id": windows[0].acquisition_id if windows else None,
            "acquisition_ids": sorted({window.acquisition_id for window in windows}),
            "assignment_id": windows[0].assignment_id if windows else None,
            "assignment_ids": sorted({window.assignment_id for window in windows}),
            "partitions": sorted({window.partition for window in windows}),
            "window_orders": [window.window_order for window in windows],
            "input_semantics": "ordered_diagnostic_windows_within_one_acquisition",
            "temporal_claim": "not_a_real_degradation_trajectory",
            "healthy_reference_id": (
                None if self._healthy_reference is None else self._healthy_reference.reference_id
            ),
            "healthy_reference_hash": (
                None if self._healthy_reference is None else self._healthy_reference.stable_hash()
            ),
            "healthy_reference_source_partitions": (
                []
                if self._healthy_reference is None
                else list(self._healthy_reference.source_partitions)
            ),
        }
