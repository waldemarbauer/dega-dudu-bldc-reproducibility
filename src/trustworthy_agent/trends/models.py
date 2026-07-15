"""Deterministic rolling spline implementations for V6 trend evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, UnivariateSpline  # type: ignore[import-untyped]

from trustworthy_agent.evidence import TrendFitStatus
from trustworthy_agent.trends.contracts import OrderedWindow
from trustworthy_agent.trends.core import BaseTrendModel, FitComputation


class RollingSmoothingSpline(BaseTrendModel):
    """Fit a smoothing spline to one acquisition's ordered diagnostic index.

    Parameters
    ----------
    config : Mapping of str to Any
        Common V6 configuration plus non-negative ``smoothing_factor``.

    Notes
    -----
    The smoothing condition is ``smoothing_factor * window_count``. The
    independent axis is normalized ordered-window position, not elapsed time.

    References
    ----------
    SciPy Developers. scipy.interpolate.UnivariateSpline, SciPy 1.17 API
    documentation. doi: none.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    """

    model_name = "RollingSmoothingSpline"

    def _validate_config(self) -> None:
        super()._validate_config()
        try:
            self._smoothing_factor = float(self._config["smoothing_factor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "INVALID_SMOOTHING_FACTOR",
                (),
                metadata={"message": str(exc)},
            ) from exc
        if not np.isfinite(self._smoothing_factor) or self._smoothing_factor < 0.0:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "INVALID_SMOOTHING_FACTOR",
                (),
            )

    def _fit_arrays(
        self,
        x: NDArray[np.float64],
        signal: NDArray[np.float64],
    ) -> FitComputation:
        degree = min(self._degree, len(signal) - 1)
        spline = UnivariateSpline(
            x,
            signal,
            k=degree,
            s=self._smoothing_factor * len(signal),
            check_finite=True,
        )
        fitted = cast(NDArray[np.float64], np.asarray(spline(x), dtype=float))
        first = cast(NDArray[np.float64], np.asarray(spline.derivative(1)(x), dtype=float))
        second = (
            cast(NDArray[np.float64], np.asarray(spline.derivative(2)(x), dtype=float))
            if degree >= 2
            else np.zeros_like(signal)
        )
        return FitComputation(
            fitted=fitted,
            first_derivative=first,
            second_derivative=second,
            effective_degrees_of_freedom=float(len(spline.get_coeffs())),
        )


class RollingBSpline(BaseTrendModel):
    """Fit an unpenalized least-squares B-spline on ordered-window position.

    Parameters
    ----------
    config : Mapping of str to Any
        Common V6 configuration plus integer ``basis_count`` not smaller than
        ``degree + 1``.

    Notes
    -----
    Coefficients are solved by deterministic least squares. Knot positions are
    evenly spaced on normalized ordering and are never learned from test data.

    References
    ----------
    SciPy Developers. scipy.interpolate.BSpline, SciPy 1.17 API documentation.
    doi: none. https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
    """

    model_name = "RollingBSpline"

    def _validate_config(self) -> None:
        super()._validate_config()
        try:
            self._basis_count = int(self._config["basis_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "INVALID_BASIS_COUNT",
                (),
                metadata={"message": str(exc)},
            ) from exc
        if self._basis_count < self._degree + 1:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "BASIS_COUNT_MUST_EXCEED_DEGREE",
                (),
            )

    def _fit_arrays(
        self,
        x: NDArray[np.float64],
        signal: NDArray[np.float64],
    ) -> FitComputation:
        knots, design = _bspline_design(x, self._degree, self._basis_count)
        coefficients_raw, _, rank, _ = np.linalg.lstsq(design, signal, rcond=None)
        coefficients = cast(NDArray[np.float64], np.asarray(coefficients_raw, dtype=float))
        return _evaluate_bspline(
            x,
            knots,
            coefficients,
            self._degree,
            effective_degrees_of_freedom=float(rank),
        )


class RollingPSpline(RollingBSpline):
    """Fit a penalized B-spline with finite-difference coefficient roughness.

    Parameters
    ----------
    config : Mapping of str to Any
        B-spline configuration plus positive ``penalty_lambda`` and
        ``difference_order`` between one and ``degree``.

    Notes
    -----
    Coefficients minimize ``||y - Bc||^2 + lambda * ||D c||^2``. This is an
    experimental deterministic smoother for diagnostic evolution, not a
    validated degradation or prognosis model.

    References
    ----------
    Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines
    and penalties. Statistical Science, 11(2), 89-121.
    doi: 10.1214/ss/1038425655
    """

    model_name = "RollingPSpline"

    def _validate_config(self) -> None:
        super()._validate_config()
        try:
            self._penalty_lambda = float(self._config["penalty_lambda"])
            self._difference_order = int(self._config["difference_order"])
        except (KeyError, TypeError, ValueError) as exc:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "INVALID_P_SPLINE_CONFIGURATION",
                (),
                metadata={"message": str(exc)},
            ) from exc
        if not np.isfinite(self._penalty_lambda) or self._penalty_lambda <= 0.0:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "PENALTY_LAMBDA_MUST_BE_POSITIVE",
                (),
            )
        if self._difference_order < 1 or self._difference_order > self._degree:
            raise self._exception(
                TrendFitStatus.INVALID_CONFIGURATION,
                "DIFFERENCE_ORDER_OUT_OF_RANGE",
                (),
            )

    def _fit_arrays(
        self,
        x: NDArray[np.float64],
        signal: NDArray[np.float64],
    ) -> FitComputation:
        knots, design = _bspline_design(x, self._degree, self._basis_count)
        coefficient_count = design.shape[1]
        difference = np.diff(
            np.eye(coefficient_count, dtype=float),
            n=self._difference_order,
            axis=0,
        )
        system = design.T @ design + self._penalty_lambda * (difference.T @ difference)
        coefficients = cast(
            NDArray[np.float64], np.asarray(np.linalg.solve(system, design.T @ signal), dtype=float)
        )
        # The trace of the smoother matrix makes complexity comparable across
        # penalty strengths without treating coefficient count as effective fit.
        smoother = design @ np.linalg.solve(system, design.T)
        effective_degrees_of_freedom = float(np.trace(smoother))
        return _evaluate_bspline(
            x,
            knots,
            coefficients,
            self._degree,
            effective_degrees_of_freedom=effective_degrees_of_freedom,
        )


class RollingHealthyRelativeSpline(RollingSmoothingSpline):
    """Smooth training-reference distances across observed acquisition windows.

    Parameters
    ----------
    config : Mapping of str to Any
        Smoothing-spline configuration. ``feature_reduction`` remains recorded
        for schema consistency but the modeled signal is standardized distance
        from the validated Healthy reference.

    Raises
    ------
    TrendModelFailure
        If the reference is absent, not training-only, belongs to another
        assignment, or has unknown/malformed provenance.

    Scientific Assumptions
    ----------------------
    Healthy distance is a training-split statistical contrast. Increasing
    distance does not itself establish damage progression or failure onset.
    """

    model_name = "RollingHealthyRelativeSpline"
    requires_healthy_reference = True

    def _diagnostic_signal(self, windows: Sequence[OrderedWindow]) -> NDArray[np.float64]:
        distances = [self._healthy_distance(window) for window in windows]
        if any(value is None for value in distances):
            raise ValueError("Healthy-relative signal requires a validated reference.")
        return np.asarray(distances, dtype=float)


def _bspline_design(
    x: NDArray[np.float64],
    degree: int,
    requested_basis_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    basis_count = max(degree + 1, min(requested_basis_count, len(x)))
    interior_count = basis_count - degree - 1
    interior = (
        np.linspace(0.0, 1.0, interior_count + 2, dtype=float)[1:-1]
        if interior_count
        else np.asarray([], dtype=float)
    )
    knots = np.concatenate(
        (
            np.repeat(x[0], degree + 1),
            interior,
            np.repeat(x[-1], degree + 1),
        )
    )
    design = np.asarray(BSpline.design_matrix(x, knots, degree).toarray(), dtype=float)
    return knots, design


def _evaluate_bspline(
    x: NDArray[np.float64],
    knots: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    degree: int,
    *,
    effective_degrees_of_freedom: float,
) -> FitComputation:
    spline = BSpline(knots, coefficients, degree, extrapolate=False)
    fitted = cast(NDArray[np.float64], np.asarray(spline(x), dtype=float))
    first = cast(NDArray[np.float64], np.asarray(spline.derivative(1)(x), dtype=float))
    second = (
        cast(NDArray[np.float64], np.asarray(spline.derivative(2)(x), dtype=float))
        if degree >= 2
        else np.zeros_like(fitted)
    )
    return FitComputation(
        fitted=fitted,
        first_derivative=first,
        second_derivative=second,
        effective_degrees_of_freedom=effective_degrees_of_freedom,
    )
