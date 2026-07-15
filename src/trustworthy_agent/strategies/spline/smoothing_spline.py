"""S3 smoothing spline strategy."""

from __future__ import annotations

from trustworthy_agent.strategies.spline._core import SmoothingSplineMixin


class SmoothingSplineStrategy(SmoothingSplineMixin):
    """Extract configured spline indicators from ordered diagnostic windows.

    Purpose:
        Implement the mandatory S3 `smoothing_spline` strategy behind the common
        state-strategy contract.
    Parameters:
        None during construction; YAML configuration is validated through
        `validate_config`.
    Return value:
        Strategy instance.
    Raised exceptions:
        ConfigurationError for invalid configuration and StrategyExecutionError
        for leakage-sensitive invalid references.
    Scientific assumptions:
        Ordered windows are treated as ordered operating windows, not proven
        real degradation trajectories.
    Side effects:
        None.
    Reproducibility implications:
        Emits strategy identity, config hash, fit status, and reference scope.
    """

    strategy_name = "smoothing_spline"
    strategy_version = "1.0.0"
