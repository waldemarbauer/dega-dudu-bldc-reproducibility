"""Versioned case-local feature extraction for one ArticleV1 raw window."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

from trustworthy_agent.article_v1.contracts import canonical_hash, require_sha256

EXPECTED_FEATURE_COUNT = 28
EXPECTED_SAMPLE_COUNT = 40_000
EXPECTED_SAMPLING_FREQUENCY_HZ = 50_000
EXTRACTION_OK = "EXTRACTION_OK"


class FeatureExtractionError(ValueError):
    """Expose a machine-readable failure without returning fabricated values.

    Parameters
    ----------
    reason_code : str
        Stable failure reason suitable for manifests and tests.
    detail : str, optional
        Human-readable context that does not need to be parsed downstream.

    Attributes
    ----------
    reason_code : str
        Stable machine-readable reason.

    Scientific Assumptions
    ----------------------
    An invalid window has no successful feature vector.  Callers may persist a
    failure row, but must not silently replace it with zero-valued features.
    """

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}:{detail}"
        super().__init__(message)


@dataclass(frozen=True)
class CanonicalWindowInput:
    """Bind two converted signals to canonical window provenance.

    Parameters
    ----------
    acquisition_id, window_id : str
        Stable source-acquisition and canonical-window identities.
    window_order : int
        Acquisition-local ordinal; it is not degradation time.
    current_amperes, rotational_speed_rpm : numpy.ndarray
        Synchronized converted channels for exactly one 40,000-sample window.
    sampling_frequency_hz : int
        Sampling frequency, fixed to 50 kHz by this schema version.
    raw_window_hash, source_acquisition_hash : str
        Canonical raw-slice and immutable source-file SHA-256 identities.

    Scientific Assumptions
    ----------------------
    Class labels, assignment IDs, and partition labels are intentionally absent
    so case-local extraction cannot use evaluation or split information.
    """

    acquisition_id: str
    window_id: str
    window_order: int
    current_amperes: NDArray[np.float64]
    rotational_speed_rpm: NDArray[np.float64]
    sampling_frequency_hz: int
    raw_window_hash: str
    source_acquisition_hash: str


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Return one finite ordered vector and its case-local provenance."""

    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    feature_schema_version: str
    feature_schema_hash: str
    extractor_version: str
    extractor_configuration_hash: str
    extraction_status: str
    extraction_failure_reason: str | None
    frequency_provenance: tuple[dict[str, float | int | str], ...]


def load_feature_config(path: Path) -> dict[str, Any]:
    """Load and structurally validate the authoritative ArticleV1 dictionary.

    Parameters
    ----------
    path : pathlib.Path
        Feature-schema YAML path.

    Returns
    -------
    dict
        Validated configuration with exactly 28 declared ordered features.

    Raises
    ------
    OSError, yaml.YAMLError
        If the file cannot be read or parsed.
    ValueError
        If identity, ordering, formulas, or harmonic semantics are incomplete.

    Security Notes
    --------------
    YAML is data only.  Implementation strings are checked as declarations and
    are never imported or executed.
    """

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("FEATURE_CONFIG_MUST_BE_MAPPING")
    config = dict(loaded)
    if config.get("config_type") != "canonical_window_feature_schema":
        raise ValueError("INVALID_FEATURE_CONFIG_TYPE")
    if config.get("feature_schema_id") != "ARTICLE_V1_CANONICAL_RAW_WINDOW_FEATURES":
        raise ValueError("INVALID_FEATURE_SCHEMA_ID")
    order = config.get("feature_order")
    features = config.get("features")
    if not isinstance(order, list) or not isinstance(features, list):
        raise ValueError("FEATURE_ORDER_AND_DICTIONARY_REQUIRED")
    names = [str(item) for item in order]
    if len(names) != EXPECTED_FEATURE_COUNT or len(set(names)) != len(names):
        raise ValueError("ARTICLE_V1_REQUIRES_28_UNIQUE_FEATURES")
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError("FEATURE_DICTIONARY_COUNT_MISMATCH")
    dictionary_names: list[str] = []
    required = {
        "feature_id",
        "feature_name",
        "channel",
        "formula",
        "unit",
        "domain",
        "implementation",
        "required_preprocessing",
        "frequency_reference",
        "interpretation",
        "interpretation_evidence_class",
        "schema_version",
    }
    for feature in features:
        if not isinstance(feature, Mapping) or not required.issubset(feature):
            raise ValueError("INCOMPLETE_FEATURE_DEFINITION")
        if not str(feature["formula"]).strip():
            raise ValueError("EMPTY_FEATURE_FORMULA")
        dictionary_names.append(str(feature["feature_name"]))
    if dictionary_names != names:
        raise ValueError("FEATURE_ORDER_MISMATCH")
    harmonic = config.get("harmonic_semantics")
    if not isinstance(harmonic, Mapping) or harmonic.get("status") not in {
        "IMPLEMENTED_EXPERIMENTAL",
        "NOT_IMPLEMENTED_FOR_ARTICLE_V1",
    }:
        raise ValueError("HARMONIC_SEMANTICS_NOT_FROZEN")
    if harmonic.get("status") == "NOT_IMPLEMENTED_FOR_ARTICLE_V1" and any(
        "harmonic" in name for name in names
    ):
        raise ValueError("NOT_IMPLEMENTED_HARMONIC_IN_FEATURE_ORDER")
    excluded = config.get("not_implemented_features")
    if not isinstance(excluded, list) or len(excluded) != 6:
        raise ValueError("NOT_IMPLEMENTED_HARMONIC_DICTIONARY_REQUIRED")
    for feature in excluded:
        if (
            not isinstance(feature, Mapping)
            or feature.get("implementation") != "NOT_IMPLEMENTED_FOR_ARTICLE_V1"
            or feature.get("included_in_feature_vector") is not False
            or not required.issubset(feature)
        ):
            raise ValueError("INVALID_NOT_IMPLEMENTED_FEATURE_DEFINITION")
    return config


def feature_schema_hash(config: Mapping[str, Any]) -> str:
    """Hash every field that can change feature meaning or ordering."""

    keys = (
        "schema_version",
        "feature_schema_id",
        "feature_schema_version",
        "input_contract",
        "fit_scope",
        "numerical_conventions",
        "harmonic_semantics",
        "not_implemented_features",
        "feature_order",
        "features",
    )
    return canonical_hash({key: config[key] for key in keys})


class WindowFeatureExtractor:
    """Compute the frozen 28-feature ArticleV1 vector from one raw window.

    Parameters
    ----------
    config : Mapping of str to Any
        Validated configuration returned by :func:`load_feature_config`.

    Side Effects
    ------------
    None.  The extractor neither fits population parameters nor mutates input.

    Reproducibility Implications
    ----------------------------
    Feature order and all numerical conventions are bound to configuration and
    schema hashes.  No RNG, label, assignment, or partition information exists
    in the public extraction input.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self.feature_names = tuple(str(item) for item in cast(list[Any], config["feature_order"]))
        if len(self.feature_names) != EXPECTED_FEATURE_COUNT:
            raise ValueError("ARTICLE_V1_REQUIRES_28_FEATURES")
        self.feature_schema_hash = feature_schema_hash(config)
        self.configuration_hash = canonical_hash(config)

    def extract(self, window: CanonicalWindowInput) -> FeatureExtractionResult:
        """Extract one finite ordered feature vector plus frequency provenance.

        Parameters
        ----------
        window : CanonicalWindowInput
            One canonical two-channel window.  It cannot carry a class label or
            train/held-out membership.

        Returns
        -------
        FeatureExtractionResult
            Exactly 28 finite values in the declared schema order.

        Raises
        ------
        FeatureExtractionError
            For malformed identity, shape, sampling metadata, non-finite input,
            or numerical failure.

        Scientific Assumptions
        ----------------------
        All operations are case-local. Harmonic features are not computed
        because the canonical corpus contains windows without a defensible
        case-local shaft frequency.
        """

        self._validate_window(window)
        current = np.asarray(window.current_amperes, dtype=np.float64)
        speed = np.asarray(window.rotational_speed_rpm, dtype=np.float64)
        values: dict[str, float] = {}
        frequency_provenance: list[dict[str, float | int | str]] = []
        for prefix, signal in (("current", current), ("speed", speed)):
            channel_values, channel_frequency = self._channel_features(
                prefix,
                signal,
                float(window.sampling_frequency_hz),
            )
            values.update(channel_values)
            frequency_provenance.extend(channel_frequency)
        try:
            ordered = tuple(float(values[name]) for name in self.feature_names)
        except KeyError as exc:
            raise FeatureExtractionError("FEATURE_ORDERING_MISMATCH", str(exc)) from exc
        if len(ordered) != EXPECTED_FEATURE_COUNT or not all(math.isfinite(v) for v in ordered):
            raise FeatureExtractionError("NONFINITE_FEATURE_OUTPUT")
        return FeatureExtractionResult(
            feature_names=self.feature_names,
            values=ordered,
            feature_schema_version=str(self._config["feature_schema_version"]),
            feature_schema_hash=self.feature_schema_hash,
            extractor_version=str(self._config["extractor_version"]),
            extractor_configuration_hash=self.configuration_hash,
            extraction_status=EXTRACTION_OK,
            extraction_failure_reason=None,
            frequency_provenance=tuple(frequency_provenance),
        )

    def _validate_window(self, window: CanonicalWindowInput) -> None:
        if not window.acquisition_id or not window.window_id or window.window_order < 0:
            raise FeatureExtractionError("INVALID_WINDOW_IDENTITY")
        try:
            require_sha256(window.raw_window_hash, "raw_window_hash")
            require_sha256(window.source_acquisition_hash, "source_acquisition_hash")
        except ValueError as exc:
            raise FeatureExtractionError("INVALID_WINDOW_HASH", str(exc)) from exc
        if window.sampling_frequency_hz != EXPECTED_SAMPLING_FREQUENCY_HZ:
            raise FeatureExtractionError("SAMPLING_FREQUENCY_MISMATCH")
        current = np.asarray(window.current_amperes)
        speed = np.asarray(window.rotational_speed_rpm)
        if current.ndim != 1 or speed.ndim != 1 or current.shape != speed.shape:
            raise FeatureExtractionError("CHANNEL_SHAPE_MISMATCH")
        if current.size != EXPECTED_SAMPLE_COUNT:
            raise FeatureExtractionError("WINDOW_SAMPLE_COUNT_MISMATCH")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(speed)):
            raise FeatureExtractionError("NONFINITE_INPUT")

    def _channel_features(
        self,
        prefix: str,
        signal: NDArray[np.float64],
        sampling_frequency_hz: float,
    ) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
        mean = float(np.mean(signal))
        centered = signal - mean
        standard_deviation = float(np.sqrt(np.mean(np.square(centered))))
        rms = float(np.sqrt(np.mean(np.square(signal))))
        if standard_deviation == 0.0:
            skewness = 0.0
            excess_kurtosis = 0.0
        else:
            standardized = centered / standard_deviation
            skewness = float(np.mean(standardized**3))
            excess_kurtosis = float(np.mean(standardized**4) - 3.0)
        crest_factor = 0.0 if rms == 0.0 else float(np.max(np.abs(signal)) / rms)
        frequencies, _amplitudes, powers, spectral_energy = _spectrum(
            centered, sampling_frequency_hz
        )
        total_power = float(np.sum(powers))
        spectral_centroid = (
            0.0 if total_power == 0.0 else float(np.sum(frequencies * powers) / total_power)
        )
        spectral_bandwidth = (
            0.0
            if total_power == 0.0
            else float(
                np.sqrt(np.sum(np.square(frequencies - spectral_centroid) * powers) / total_power)
            )
        )
        features = {
            f"{prefix}_mean": mean,
            f"{prefix}_standard_deviation": standard_deviation,
            f"{prefix}_variance": standard_deviation**2,
            f"{prefix}_rms": rms,
            f"{prefix}_maximum": float(np.max(signal)),
            f"{prefix}_minimum": float(np.min(signal)),
            f"{prefix}_peak_absolute": float(np.max(np.abs(signal))),
            f"{prefix}_peak_to_peak": float(np.ptp(signal)),
            f"{prefix}_skewness": skewness,
            f"{prefix}_excess_kurtosis": excess_kurtosis,
            f"{prefix}_crest_factor": crest_factor,
            f"{prefix}_spectral_energy": spectral_energy,
            f"{prefix}_spectral_centroid_hz": spectral_centroid,
            f"{prefix}_spectral_bandwidth_hz": spectral_bandwidth,
        }
        resolution = sampling_frequency_hz / signal.size
        nyquist = sampling_frequency_hz / 2.0
        provenance: list[dict[str, float | int | str]] = [
            {
                "channel": prefix,
                "frequency_bin_resolution_hz": resolution,
                "nyquist_frequency_hz": nyquist,
                "dominant_frequency_hz": float(frequencies[int(np.argmax(powers))]),
                "spectral_window": "periodic_hann",
                "harmonic_status": "NOT_IMPLEMENTED_FOR_ARTICLE_V1",
            }
        ]
        return features, provenance


def _spectrum(
    centered: NDArray[np.float64], sampling_frequency_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    sample_count = centered.size
    # A periodic Hann is deterministic and avoids duplicating the endpoint of
    # the sampled interval.  Coherent-gain and power normalizations serve
    # different quantities, so amplitude and energy are normalized separately.
    hann = np.hanning(sample_count + 1)[:-1]
    transformed = np.fft.rfft(centered * hann)
    frequencies = cast(
        NDArray[np.float64],
        np.asarray(np.fft.rfftfreq(sample_count, d=1.0 / sampling_frequency_hz), dtype=float),
    )
    amplitudes = cast(
        NDArray[np.float64],
        np.asarray(np.abs(transformed) / float(np.sum(hann)), dtype=float),
    )
    powers = cast(NDArray[np.float64], np.asarray(np.square(np.abs(transformed)), dtype=float))
    if sample_count > 1:
        amplitudes[1:-1] *= 2.0
        powers[1:-1] *= 2.0
    spectral_energy = float(np.sum(powers) / (sample_count * np.sum(np.square(hann))))
    return frequencies, amplitudes, powers, spectral_energy
