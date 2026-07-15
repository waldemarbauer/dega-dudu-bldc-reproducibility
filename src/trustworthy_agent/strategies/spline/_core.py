"""Core S3 spline helpers with explicit leakage and failure semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline  # type: ignore[import-untyped]

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.state_result import StateResult
from trustworthy_agent.exceptions import StrategyExecutionError
from trustworthy_agent.strategies._common import (
    as_mapping,
    context_value,
    ordered_series_from_context,
    provenance,
    require_number,
    require_strategy_identity,
    require_training_only,
)

FIT_OK = "FIT_OK"
INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
NONFINITE_INPUT = "NONFINITE_INPUT"
NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


def fit_training_healthy_reference(
    healthy_series: list[list[float]],
    *,
    reference_id: str = "healthy_reference",
) -> dict[str, Any]:
    """Fit a Healthy reference from training-only Healthy ordered series.

    Purpose:
        Provide a small reference artifact for S3 distance features without
        allowing test-set information into the reference.
    Parameters:
        healthy_series: Training-fold Healthy ordered series, one sequence per
            row.
        reference_id: Stable reference artifact label.
    Return value:
        JSON-compatible Healthy reference artifact.
    Raised exceptions:
        StrategyExecutionError for empty or non-finite training data.
    Scientific assumptions:
        This artifact is a training-fold summary, not a universal physical
        health model.
    Side effects:
        None.
    Reproducibility implications:
        Records `fit_scope=training_only` and deterministic summary arrays.
    """

    values = np.asarray(healthy_series, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise StrategyExecutionError("Healthy reference requires at least one training series.")
    if not np.all(np.isfinite(values)):
        raise StrategyExecutionError("Healthy reference training data contains non-finite values.")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std <= 0.0, 1.0, std)
    return {
        "reference_id": reference_id,
        "fit_scope": "training_only",
        "mean_series": mean.tolist(),
        "std_series": std.tolist(),
        "sample_count": int(values.shape[0]),
    }


class BaseSplineStrategy:
    """Shared implementation for S3 spline strategies."""

    strategy_name = "base_spline"
    strategy_version = "1.0.0"

    def __init__(self) -> None:
        self._config: Mapping[str, Any] = {}

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Validate S3 config and leakage-sensitive fit scopes."""

        require_strategy_identity(config, self.strategy_name, self.strategy_version)
        healthy_reference = as_mapping(config.get("healthy_reference"), "healthy_reference")
        if healthy_reference.get("enabled", False):
            require_training_only(healthy_reference.get("fit_scope"), "healthy_reference.fit_scope")
        ordering = as_mapping(config.get("ordering"), "ordering")
        constructed = as_mapping(ordering.get("constructed_order"), "ordering.constructed_order")
        if constructed.get("enabled", False):
            require_training_only(
                constructed.get("ordering_fitted_on"),
                "ordering.constructed_order.ordering_fitted_on",
            )
        self._config = dict(config)

    def execute(self, context: AgentContext) -> StateResult:
        """Execute S3 and return spline facts without selecting a next state."""

        series = ordered_series_from_context(context)
        fit_config = as_mapping(self._config.get("fit"), "fit")
        min_points = int(require_number(fit_config.get("min_points", 4), "fit.min_points"))
        if series is None:
            return self._failure(INSUFFICIENT_POINTS, "ORDERED_SERIES_UNAVAILABLE")
        if series.ndim != 1 or series.size < min_points:
            return self._failure(INSUFFICIENT_POINTS, "INSUFFICIENT_ORDERED_POINTS")
        if not np.all(np.isfinite(series)):
            return self._failure(NONFINITE_INPUT, "NONFINITE_ORDERED_SERIES")
        return self._fit_ok(context, cast(NDArray[np.float64], series.astype(float)))

    def _fit_ok(self, context: AgentContext, series: NDArray[np.float64]) -> StateResult:
        raise NotImplementedError

    def _result_from_arrays(
        self,
        context: AgentContext,
        *,
        smoothed: NDArray[np.float64],
        slope: NDArray[np.float64],
        second_derivative: NDArray[np.float64],
        fit_quality: float,
    ) -> StateResult:
        curvature = np.abs(second_derivative) / np.power(1.0 + np.square(slope), 1.5)
        distance, area = healthy_distance_and_area(context, smoothed)
        first_threshold_crossing = threshold_crossing(self._config, smoothed)
        spline_features = {
            "smoothed_value": float(smoothed[-1]),
            "slope": float(slope[-1]),
            "second_derivative": float(second_derivative[-1]),
            "curvature": float(curvature[-1]),
            "max_curvature": float(np.max(curvature)),
            "distance_from_healthy": distance,
            "area_deviation_from_healthy": area,
            "first_threshold_crossing": first_threshold_crossing,
            "fit_quality": fit_quality,
            "fit_status": FIT_OK,
            "ordering_semantics": "ordered_operating_windows",
            "temporal_claim": "not_real_degradation_time",
        }
        return StateResult(
            facts={"spline_features": spline_features, **spline_features},
            reason_codes=("S3_FIT_OK",),
            provenance=provenance(
                self.strategy_name,
                self.strategy_version,
                self._config,
                {
                    "fit_status": FIT_OK,
                    "healthy_reference_fit_scope": healthy_reference_fit_scope(context),
                    "ordering_semantics": "ordered_operating_windows",
                },
            ),
        )

    def _failure(self, fit_status: str, reason_code: str) -> StateResult:
        spline_features = {
            "smoothed_value": None,
            "slope": None,
            "second_derivative": None,
            "curvature": None,
            "max_curvature": None,
            "distance_from_healthy": None,
            "area_deviation_from_healthy": None,
            "first_threshold_crossing": None,
            "fit_quality": None,
            "fit_status": fit_status,
        }
        return StateResult(
            facts={"spline_features": spline_features, **spline_features},
            reason_codes=(reason_code,),
            provenance=provenance(
                self.strategy_name,
                self.strategy_version,
                self._config,
                {"fit_status": fit_status},
            ),
        )


class SmoothingSplineMixin(BaseSplineStrategy):
    """Smoothing spline feature extraction using SciPy `UnivariateSpline`."""

    def _fit_ok(self, context: AgentContext, series: NDArray[np.float64]) -> StateResult:
        x = np.arange(series.size, dtype=float)
        try:
            smoothing = float(series.size)
            spline = UnivariateSpline(x, series, s=smoothing, k=min(3, series.size - 1))
            smoothed = cast(NDArray[np.float64], spline(x))
            slope = cast(NDArray[np.float64], spline.derivative(1)(x))
            second = cast(NDArray[np.float64], spline.derivative(2)(x))
        except Exception as exc:
            if "Singular" in str(exc):
                return self._failure("SINGULAR_FIT", "SINGULAR_FIT")
            return self._failure(NUMERICAL_FAILURE, "SPLINE_NUMERICAL_FAILURE")
        residual = series - smoothed
        denominator = float(np.var(series))
        fit_quality = (
            1.0 if denominator == 0.0 else max(0.0, 1.0 - float(np.var(residual)) / denominator)
        )
        return self._result_from_arrays(
            context,
            smoothed=smoothed,
            slope=slope,
            second_derivative=second,
            fit_quality=fit_quality,
        )


class IdentitySplineMixin(BaseSplineStrategy):
    """Identity/no-spline baseline preserving the input ordered series."""

    def _fit_ok(self, context: AgentContext, series: NDArray[np.float64]) -> StateResult:
        slope = cast(NDArray[np.float64], np.gradient(series))
        second = cast(NDArray[np.float64], np.gradient(slope))
        return self._result_from_arrays(
            context,
            smoothed=series,
            slope=slope,
            second_derivative=second,
            fit_quality=1.0,
        )


def healthy_distance_and_area(
    context: AgentContext,
    series: NDArray[np.float64],
) -> tuple[float | None, float | None]:
    reference = context_value(context, "healthy_reference")
    if reference is None:
        return None, None
    if not isinstance(reference, Mapping):
        raise StrategyExecutionError("Healthy reference must be a mapping.")
    if reference.get("fit_scope") != "training_only":
        raise StrategyExecutionError("Healthy reference must be fit on training data only.")
    mean = np.asarray(reference.get("mean_series"), dtype=float)
    std = np.asarray(reference.get("std_series"), dtype=float)
    if mean.shape != series.shape or std.shape != series.shape:
        raise StrategyExecutionError("Healthy reference shape must match ordered series.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise StrategyExecutionError("Healthy reference contains non-finite values.")
    safe_std = np.where(std <= 0.0, 1.0, std)
    standardized = (series - mean) / safe_std
    distance = float(np.sqrt(np.mean(np.square(standardized))))
    area = float(np.trapezoid(np.abs(series - mean)))
    return distance, area


def threshold_crossing(config: Mapping[str, Any], series: NDArray[np.float64]) -> int | None:
    threshold_config = config.get("threshold_crossing")
    if threshold_config is None:
        return None
    threshold = require_number(
        as_mapping(threshold_config, "threshold_crossing").get("value"), "threshold"
    )
    crossings = np.flatnonzero(series >= threshold)
    if crossings.size == 0:
        return None
    return int(crossings[0])


def healthy_reference_fit_scope(context: AgentContext) -> str | None:
    reference = context_value(context, "healthy_reference")
    if not isinstance(reference, Mapping):
        return None
    fit_scope = reference.get("fit_scope")
    if isinstance(fit_scope, str):
        return fit_scope
    return None
