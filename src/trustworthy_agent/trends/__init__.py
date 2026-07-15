"""Optional V6 temporal trend representation over ordered acquisition windows."""

from trustworthy_agent.trends.config import create_trend_model, load_trend_family_config
from trustworthy_agent.trends.contracts import (
    HealthyReference,
    OrderedWindow,
    TrendMode,
    TrendModel,
    TrendModelFailure,
    WindowPartition,
)
from trustworthy_agent.trends.models import (
    RollingBSpline,
    RollingHealthyRelativeSpline,
    RollingPSpline,
    RollingSmoothingSpline,
)

__all__ = [
    "HealthyReference",
    "OrderedWindow",
    "RollingBSpline",
    "RollingHealthyRelativeSpline",
    "RollingPSpline",
    "RollingSmoothingSpline",
    "TrendMode",
    "TrendModel",
    "TrendModelFailure",
    "WindowPartition",
    "create_trend_model",
    "load_trend_family_config",
]
