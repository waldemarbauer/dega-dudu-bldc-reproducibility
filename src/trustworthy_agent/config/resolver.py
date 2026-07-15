"""Configuration resolution skeleton."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def resolved_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a resolved configuration mapping deterministically.

    Purpose:
        Provide a non-scientific helper for config fingerprinting.
    Parameters:
        config: JSON-serializable resolved config mapping.
    Return value:
        SHA-256 hex digest.
    Raised exceptions:
        TypeError if config contains non-serializable values.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Excludes nondeterministic key ordering from config identity.
    """

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
