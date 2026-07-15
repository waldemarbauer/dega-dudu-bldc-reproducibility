"""Validated declarative configuration and allowlisted V6 model factory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias, cast

import yaml  # type: ignore[import-untyped]

from trustworthy_agent.evidence import TrendEvidence, TrendFitStatus
from trustworthy_agent.trends.contracts import TrendMode, TrendModel, TrendModelFailure
from trustworthy_agent.trends.models import (
    RollingBSpline,
    RollingHealthyRelativeSpline,
    RollingPSpline,
    RollingSmoothingSpline,
)

TrendModelType: TypeAlias = type[
    RollingSmoothingSpline | RollingBSpline | RollingPSpline | RollingHealthyRelativeSpline
]

MODEL_TYPES: dict[str, TrendModelType] = {
    "RollingSmoothingSpline": RollingSmoothingSpline,
    "RollingBSpline": RollingBSpline,
    "RollingPSpline": RollingPSpline,
    "RollingHealthyRelativeSpline": RollingHealthyRelativeSpline,
}


def load_trend_family_config(path: Path) -> dict[str, Any]:
    """Load and validate the declarative WINDOW_TEMPORAL_SPLINE family config.

    Parameters
    ----------
    path : pathlib.Path
        YAML file containing the complete V6 family declaration.

    Returns
    -------
    config : dict of str to Any
        Parsed and structurally validated configuration.

    Raises
    ------
    OSError, yaml.YAMLError
        If the YAML cannot be read or parsed.
    TrendModelFailure
        If identity, supported models, defaults, or scientific semantics are
        absent or invalid.

    Security Notes
    --------------
    ``yaml.safe_load`` is used and implementation paths are informational only;
    runtime resolution uses the fixed ``MODEL_TYPES`` allowlist.
    """

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise _config_failure("TREND_CONFIG_MUST_BE_A_MAPPING")
    config = dict(loaded)
    if config.get("config_type") != "trend_representation":
        raise _config_failure("INVALID_TREND_CONFIG_TYPE")
    if config.get("representation_family") != "V6":
        raise _config_failure("REPRESENTATION_FAMILY_MUST_BE_V6")
    if config.get("representation_id") != TrendEvidence.REPRESENTATION_ID:
        raise _config_failure("INVALID_REPRESENTATION_ID")
    if config.get("representation_version") != TrendEvidence.REPRESENTATION_VERSION:
        raise _config_failure("INVALID_REPRESENTATION_VERSION")
    semantics = _mapping(config.get("scientific_semantics"), "scientific_semantics")
    if semantics.get("scope") != "ordered_diagnostic_evolution_within_one_acquisition":
        raise _config_failure("INVALID_SCIENTIFIC_SCOPE")
    forbidden = set(_string_sequence(semantics.get("forbidden_claims"), "forbidden_claims"))
    required_forbidden = {
        "true_degradation",
        "remaining_useful_life",
        "physical_failure_onset",
        "real_degradation_trajectory",
    }
    if not required_forbidden.issubset(forbidden):
        raise _config_failure("REQUIRED_SCIENTIFIC_CLAIM_GUARDS_MISSING")
    defaults = _mapping(config.get("defaults"), "defaults")
    try:
        TrendMode(str(defaults["mode"]))
    except (KeyError, ValueError) as exc:
        raise _config_failure("INVALID_DEFAULT_TREND_MODE") from exc
    models = _mapping(config.get("models"), "models")
    if set(models) != set(MODEL_TYPES):
        raise _config_failure("SUPPORTED_MODEL_SET_MISMATCH")
    for name, value in models.items():
        model_config = _mapping(value, f"models.{name}")
        if model_config.get("enabled") is not True:
            raise _config_failure(f"MODEL_NOT_ENABLED:{name}")
        _mapping(model_config.get("parameters"), f"models.{name}.parameters")
    return config


def create_trend_model(
    family_config: Mapping[str, Any],
    model_name: str,
    *,
    mode: TrendMode | str | None = None,
) -> TrendModel:
    """Instantiate one allowlisted trend model from resolved family settings.

    Parameters
    ----------
    family_config : Mapping of str to Any
        Validated family configuration returned by
        :func:`load_trend_family_config`.
    model_name : str
        One of the four supported stable implementation names.
    mode : TrendMode or str or None, optional
        Explicit OFFLINE/ONLINE override. ``None`` uses the declared default.

    Returns
    -------
    model : TrendModel
        New deterministic model satisfying the common protocol.

    Raises
    ------
    TrendModelFailure
        If the name or resolved configuration is invalid.

    Reproducibility Implications
    ----------------------------
    Runtime classes are selected from a fixed name-to-class map; YAML cannot
    execute arbitrary Python.
    """

    if model_name not in MODEL_TYPES:
        raise _config_failure(f"UNSUPPORTED_TREND_MODEL:{model_name}")
    defaults = dict(_mapping(family_config.get("defaults"), "defaults"))
    models = _mapping(family_config.get("models"), "models")
    declared = _mapping(models.get(model_name), f"models.{model_name}")
    parameters = dict(_mapping(declared.get("parameters"), "parameters"))
    resolved = {**defaults, **parameters}
    if mode is not None:
        try:
            resolved["mode"] = TrendMode(str(mode)).value
        except ValueError:
            if isinstance(mode, TrendMode):
                resolved["mode"] = mode.value
            else:
                raise _config_failure("INVALID_TREND_MODE_OVERRIDE") from None
    model_type = MODEL_TYPES[model_name]
    return cast(TrendModel, model_type(resolved))


def _config_failure(reason: str) -> TrendModelFailure:
    from trustworthy_agent.trends.contracts import canonical_hash

    return TrendModelFailure(
        TrendEvidence(
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
            window_count=0,
            fit_status=TrendFitStatus.INVALID_CONFIGURATION,
            representation_id=TrendEvidence.REPRESENTATION_ID,
            representation_version=TrendEvidence.REPRESENTATION_VERSION,
            configuration_hash=canonical_hash({"invalid_configuration": reason}),
            input_window_hashes=(),
            reason=reason,
            metadata={"component": "trend_configuration"},
            provenance={"configuration_status": "rejected_before_computation"},
        )
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _config_failure(f"{name.upper()}_MUST_BE_A_MAPPING")
    return value


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list):
        raise _config_failure(f"{name.upper()}_MUST_BE_A_LIST")
    return tuple(str(item) for item in value)
