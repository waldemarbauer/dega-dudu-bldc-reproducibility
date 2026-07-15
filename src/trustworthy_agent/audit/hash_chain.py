"""Hash-chain helpers for audit events."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from trustworthy_agent.audit.canonical_json import canonical_json_bytes


def event_hash(event: Mapping[str, Any]) -> str:
    """Compute the SHA-256 hash for a canonical audit event.

    Purpose:
        Support tamper-evident audit chains without blockchain.
    Parameters:
        event: Audit event mapping.
    Return value:
        Hex SHA-256 digest.
    Raised exceptions:
        TypeError if the event cannot be canonicalized.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Hashes are deterministic for semantically identical event mappings.
    """

    return hashlib.sha256(canonical_json_bytes(event)).hexdigest()
