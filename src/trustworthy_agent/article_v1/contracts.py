"""Deterministic identities shared by ArticleV1 prerequisite artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_hash(value: Any) -> str:
    """Return a deterministic SHA-256 for finite JSON-compatible content.

    Parameters
    ----------
    value : Any
        JSON-compatible object whose map order must not affect its identity.

    Returns
    -------
    str
        Lowercase SHA-256 digest of compact, sorted UTF-8 JSON.

    Raises
    ------
    TypeError, ValueError
        If the value cannot be serialized or contains NaN/infinity.

    Reproducibility Implications
    ----------------------------
    Absolute paths and timestamps are excluded unless a caller explicitly
    places them in the supplied value.
    """

    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def require_sha256(value: str, field: str) -> None:
    """Reject a malformed lowercase SHA-256 identity.

    Parameters
    ----------
    value : str
        Candidate digest.
    field : str
        Field name included in a structured validation error.

    Raises
    ------
    ValueError
        If ``value`` is not exactly 64 lowercase hexadecimal characters.
    """

    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"INVALID_SHA256:{field}")


def require_finite(values: Sequence[float], field: str) -> None:
    """Reject non-finite numerical content without imputing a default.

    Parameters
    ----------
    values : Sequence of float
        Numerical values to validate.
    field : str
        Field name included in the failure reason.

    Raises
    ------
    ValueError
        If any value is NaN or infinity.

    Scientific Assumptions
    ----------------------
    Missing and invalid measurements remain explicit failures; zero is never
    used as an undeclared imputation value.
    """

    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"NONFINITE_INPUT:{field}")


def canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a mapping into a plain dictionary for canonical serialization."""

    return {str(key): item for key, item in value.items()}
