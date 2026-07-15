"""Audit hash-chain verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trustworthy_agent.audit.hash_chain import event_hash
from trustworthy_agent.audit.logger import read_audit_events
from trustworthy_agent.exceptions import AuditWriteError


@dataclass(frozen=True)
class AuditVerificationResult:
    """Structured result from audit verification."""

    valid: bool
    event_count: int
    run_id: str | None
    failure_reason: str | None = None
    failure_event_index: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible verification summary."""

        return {
            "valid": self.valid,
            "event_count": self.event_count,
            "run_id": self.run_id,
            "failure_reason": self.failure_reason,
            "failure_event_index": self.failure_event_index,
        }


def verify_audit_chain(path: Path) -> AuditVerificationResult:
    """Verify tamper-evident hash-chain integrity for an audit JSONL file.

    Purpose:
        Detect tampered events, missing sequence entries, and modified previous
        hashes.
    Parameters:
        path: Audit JSONL path.
    Return value:
        Structured verification result.
    Raised exceptions:
        AuditWriteError when the audit file cannot be read or parsed.
    Scientific assumptions:
        None.
    Side effects:
        Reads the audit file only.
    Reproducibility implications:
        Provides deterministic audit-verification evidence.
    """

    try:
        events = read_audit_events(path)
    except (OSError, ValueError) as exc:
        raise AuditWriteError(f"Failed to read audit file {path}: {exc}") from exc
    if not events:
        return AuditVerificationResult(False, 0, None, "EMPTY_AUDIT_LOG", None)
    previous_hash: str | None = None
    run_id = str(events[0].get("run_id")) if events[0].get("run_id") else None
    expected_index = 1
    for event in events:
        event_index = int(event.get("event_index", -1))
        if event_index != expected_index:
            return AuditVerificationResult(
                False,
                len(events),
                run_id,
                "MISSING_OR_NONSEQUENTIAL_EVENT",
                event_index,
            )
        if event.get("previous_event_hash") != previous_hash:
            return AuditVerificationResult(
                False,
                len(events),
                run_id,
                "PREVIOUS_HASH_MISMATCH",
                event_index,
            )
        stored_hash = event.get("event_hash")
        if stored_hash != event_hash(event):
            return AuditVerificationResult(
                False,
                len(events),
                run_id,
                "EVENT_HASH_MISMATCH",
                event_index,
            )
        previous_hash = str(stored_hash)
        expected_index += 1
    return AuditVerificationResult(True, len(events), run_id)
