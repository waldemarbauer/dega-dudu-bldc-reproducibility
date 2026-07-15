"""Deterministic JSON serialization for audit hash inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def canonical_json_bytes(event: Mapping[str, Any]) -> bytes:
    """Serialize an event deterministically for audit hashing.

    Purpose:
        Provide sorted-key UTF-8 JSON serialization and exclude `event_hash`
        from its own hash input.
    Parameters:
        event: Audit event mapping.
    Return value:
        UTF-8 encoded canonical JSON bytes.
    Raised exceptions:
        TypeError if values are not JSON serializable.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Stable serialization supports replayable hash chains.
    """

    hash_input = {key: value for key, value in event.items() if key != "event_hash"}
    return json.dumps(hash_input, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
