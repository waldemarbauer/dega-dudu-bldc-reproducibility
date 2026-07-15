"""Shared helpers for replaceable S3-S7 strategy implementations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.exceptions import ConfigurationError, StrategyExecutionError

CANONICAL_CLASSES: tuple[str, ...] = (
    "Healthy",
    "Mech_Damage",
    "Elec_Damage",
    "Mech_Elec_Damage",
)

EVIDENCE_TAGS: tuple[str, ...] = (
    "EMPIRICALLY_MEASURED",
    "LITERATURE_SUPPORTED",
    "DOMAIN_HYPOTHESIS",
)


def require_strategy_identity(
    config: Mapping[str, Any],
    expected_name: str,
    expected_version: str,
) -> Mapping[str, Any]:
    """Validate and return the active strategy block from configuration.

    Purpose:
        Ensure a strategy executes only under the YAML contract that names its
        stable identity.
    Parameters:
        config: Parsed strategy configuration.
        expected_name: Strategy name implemented by the class.
        expected_version: Strategy version implemented by the class.
    Return value:
        Strategy identity mapping.
    Raised exceptions:
        ConfigurationError for missing or mismatched identity.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Prevents accidental execution under ambiguous strategy identity.
    """

    strategy = as_mapping(config.get("strategy"), "strategy")
    name = require_text(strategy.get("name"), "strategy.name")
    version = require_text(strategy.get("version"), "strategy.version")
    if name != expected_name or version != expected_version:
        raise ConfigurationError(
            f"Expected strategy {expected_name}@{expected_version}; got {name}@{version}."
        )
    return strategy


def context_value(context: AgentContext, key: str) -> Any:
    """Read a value from top-level context fields or derived facts.

    Purpose:
        Let replaceable strategies consume outputs from earlier strategies
        without depending on engine internals.
    Parameters:
        context: Current immutable agent context.
        key: Field/fact name.
    Return value:
        Stored value, or `None` when explicitly unavailable.
    Raised exceptions:
        None.
    Scientific assumptions:
        `None` means unknown/unavailable, not a numeric default.
    Side effects:
        None.
    Reproducibility implications:
        Keeps strategy chaining deterministic and explicit.
    """

    if hasattr(context, key):
        value = getattr(context, key)
        if value is not None and value != () and value != {}:
            return value
    return context.derived_facts.get(key)


def provenance(
    strategy_name: str,
    strategy_version: str,
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create common strategy provenance evidence.

    Purpose:
        Attach strategy identity and deterministic configuration hash to every
        derived field.
    Parameters:
        strategy_name: Stable strategy name.
        strategy_version: Stable strategy version.
        config: Parsed strategy configuration.
        extra: Optional extra provenance fields.
    Return value:
        JSON-compatible provenance mapping.
    Raised exceptions:
        TypeError if configuration is not JSON-serializable.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Supports replay and audit attribution for derived fields.
    """

    payload: dict[str, Any] = {
        "strategy_name": strategy_name,
        "strategy_version": strategy_version,
        "strategy_params_hash": stable_hash(config),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def stable_hash(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Hash JSON-compatible strategy evidence deterministically."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def feature_vector_from_context(
    context: AgentContext,
    feature_names: Sequence[str] | None = None,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Extract a deterministic numeric feature vector from context evidence."""

    feature_mapping: dict[str, Any] = {}
    for candidate in (
        context.raw_features,
        context.classical_features,
        context.spline_features,
        cast(Mapping[str, Any] | None, context.derived_facts.get("spline_features")),
        cast(Mapping[str, Any] | None, context.derived_facts.get("classical_features")),
    ):
        if isinstance(candidate, Mapping):
            feature_mapping.update(dict(candidate))
    if feature_names is None:
        names = tuple(sorted(name for name, value in feature_mapping.items() if is_number(value)))
    else:
        names = tuple(feature_names)
    if not names:
        raise StrategyExecutionError("No numeric features are available for strategy execution.")
    values = [float(feature_mapping[name]) for name in names]
    vector = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(vector)):
        raise StrategyExecutionError("Feature vector contains non-finite values.")
    return vector, names


def ordered_series_from_context(context: AgentContext) -> NDArray[np.float64] | None:
    """Read an ordered numeric series without inventing temporal semantics."""

    for key in ("ordered_series", "diagnostic_ordered_series", "series"):
        value = context.derived_facts.get(key)
        if value is None and isinstance(context.raw_features, Mapping):
            value = context.raw_features.get(key)
        if value is None and isinstance(context.classical_features, Mapping):
            value = context.classical_features.get(key)
        if value is not None:
            array = np.asarray(value, dtype=float)
            return cast(NDArray[np.float64], array)
    return None


def as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"`{field_name}` must be a mapping.")
    return cast(Mapping[str, Any], value)


def require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"`{field_name}` must be a non-empty string.")
    return value


def require_number(value: object, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise ConfigurationError(f"`{field_name}` must be numeric.")
    number = float(value)
    if not np.isfinite(number):
        raise ConfigurationError(f"`{field_name}` must be finite.")
    return number


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and bool(np.isfinite(float(value)))


def require_training_only(value: object, field_name: str) -> None:
    if value != "training_only":
        raise ConfigurationError(f"`{field_name}` must be training_only to prevent leakage.")


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
