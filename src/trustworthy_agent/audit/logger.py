"""Append-only tamper-evident JSONL audit recorder."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.state_result import StateResult
from trustworthy_agent.audit.hash_chain import event_hash
from trustworthy_agent.audit.schemas import AUDIT_EVENT_FIELDS, AUDIT_SCHEMA_VERSION
from trustworthy_agent.exceptions import AuditWriteError
from trustworthy_agent.safety.base import GuardedTransition, SafetyRuleResult


class AuditRecorder(Protocol):
    """Protocol for explicit audit side-effect adapters."""

    def record_state_execution(self, context: AgentContext, result: StateResult) -> None:
        """Persist state execution evidence."""

    def record_transition(self, context: AgentContext, guarded: GuardedTransition) -> None:
        """Persist proposed and safety-gated transition evidence."""

    def finalize_outcome(self, context: AgentContext) -> AgentContext:
        """Finalize terminal audit evidence and return the audited context."""


class JsonlAuditRecorder:
    """Append-only JSONL audit recorder with hash chaining.

    Purpose:
        Persist state, transition, and terminal evidence in a tamper-evident
        audit file.
    Parameters:
        path: Audit JSONL path.
        git_commit: Optional source commit identity.
    Return value:
        Audit recorder instance.
    Raised exceptions:
        AuditWriteError when mandatory persistence fails.
    Scientific assumptions:
        Hash chaining detects tampering; it is not blockchain.
    Side effects:
        Creates parent directories and appends JSONL events.
    Reproducibility implications:
        Event hashes are deterministic after volatile timestamp/event ID fields
        are set.
    """

    def __init__(self, path: Path, *, git_commit: str | None = None) -> None:
        self.path = path
        self.git_commit = git_commit
        self._event_index, self._previous_hash = _last_chain_state(path)

    def record_state_execution(self, context: AgentContext, result: StateResult) -> None:
        """Persist state execution evidence."""

        strategy_name, strategy_version = active_strategy_for_context(context)
        self._append_event(
            context=context,
            event_type="STATE_EXECUTED",
            state_from=context.current_state.value if context.current_state else None,
            state_to=context.current_state.value if context.current_state else None,
            active_state_strategy=strategy_name,
            strategy_version=strategy_version,
            strategy_params_hash=safe_get(result.provenance, "strategy_params_hash"),
            decision_reason=first_reason(result.reason_codes, context),
            extra={"state_facts": to_jsonable(dict(result.facts))},
        )

    def record_transition(self, context: AgentContext, guarded: GuardedTransition) -> None:
        """Persist proposed and safety-gated transition evidence."""

        proposed = guarded.proposed_transition
        self._append_event(
            context=context,
            event_type="TRANSITION_COMMITTED",
            state_from=proposed.current_state.value,
            state_to=guarded.final_state.value,
            transition_policy=proposed.policy_name,
            transition_policy_version=proposed.policy_version,
            candidate_states=[state.value for state in proposed.candidate_states],
            raw_transition_scores={
                state.value: score for state, score in proposed.raw_scores.items()
            },
            normalized_probabilities={
                state.value: probability for state, probability in proposed.probabilities.items()
            },
            selected_state=proposed.selected_state.value,
            selection_mode=proposed.selection_mode,
            rng_seed_or_stream_id=proposed.rng_seed_or_stream_id,
            safety_rules_evaluated=[
                rule_result_json(result) for result in guarded.safety_rules_evaluated
            ],
            safety_rules_triggered=[
                rule_result_json(result) for result in guarded.safety_rules_triggered
            ],
            safety_override=guarded.safety_override.value if guarded.safety_override else None,
            decision_reason=proposed.decision_reason or context.decision_reason,
        )

    def finalize_outcome(self, context: AgentContext) -> AgentContext:
        """Write terminal audit evidence and return the externally valid context."""

        self._append_event(
            context=context,
            event_type="AUDIT_FINALIZED",
            state_from=context.current_state.value if context.current_state else None,
            state_to=context.current_state.value if context.current_state else None,
            decision_reason=context.decision_reason,
            final_action=context.final_action,
        )
        return context

    def _append_event(self, *, context: AgentContext, event_type: str, **fields: Any) -> None:
        event_index = self._event_index + 1
        event = base_event(context, event_type, event_index, self._previous_hash, self.git_commit)
        event.update({key: to_jsonable(value) for key, value in fields.items()})
        for field in AUDIT_EVENT_FIELDS:
            event.setdefault(field, None)
        event["event_hash"] = event_hash(event)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
                )
                handle.write("\n")
        except OSError as exc:
            raise AuditWriteError(f"Failed to append audit event to {self.path}: {exc}") from exc
        self._event_index = event_index
        self._previous_hash = str(event["event_hash"])


def fail_closed_context(context: AgentContext, reason: str = "AUDIT_UNAVAILABLE") -> AgentContext:
    """Return a context that cannot be treated as an external recommendation."""

    from dataclasses import replace

    return replace(
        context,
        final_action="NO_AUTOMATED_RECOMMENDATION",
        decision_reason=reason,
        derived_facts={
            **context.derived_facts,
            "audit_available": False,
            "external_decision_valid": False,
            "audit_failure_reason": reason,
        },
    )


def audit_path_for_run(project_root: Path, run_id: str) -> Path:
    """Return the canonical audit JSONL path for a run ID."""

    return project_root / "Output" / "Audit" / f"{run_id}.jsonl"


def base_event(
    context: AgentContext,
    event_type: str,
    event_index: int,
    previous_hash: str | None,
    git_commit: str | None,
) -> dict[str, Any]:
    """Build common audit event fields from an agent context."""

    run_id = context.run_id or "UNKNOWN_RUN"
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_type": event_type,
        "event_id": f"{run_id}:{event_index:06d}",
        "event_index": event_index,
        "run_id": run_id,
        "experiment_fingerprint": context.experiment_fingerprint,
        "case_id": context.case_id,
        "window_id": context.window_id,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "diagnosis": context.diagnosis or context.derived_facts.get("diagnosis"),
        "class_probabilities": context.class_probabilities
        or context.derived_facts.get("class_probabilities"),
        "confidence": context.confidence or context.derived_facts.get("confidence"),
        "spline_summary": context.spline_features
        or context.derived_facts.get("spline_features")
        or context.derived_facts.get("spline_summary"),
        "explanation_score": context.explanation_score
        or context.derived_facts.get("explanation_score"),
        "top_explanatory_features": context.top_explanatory_features
        or context.derived_facts.get("top_explanatory_features"),
        "risk_score": context.risk_score or context.derived_facts.get("risk_score"),
        "ood_score": context.ood_score or context.derived_facts.get("ood_score"),
        "decision_reason": context.decision_reason,
        "final_action": context.final_action,
        "input_hash": context.derived_facts.get("input_hash") or context.raw_record_reference,
        "config_hash": context.resolved_config_hash or context.derived_facts.get("config_hash"),
        "model_hash": context.derived_facts.get("model_hash"),
        "git_commit": git_commit or context.git_commit,
        "previous_event_hash": previous_hash,
        "event_hash": None,
    }


def read_audit_events(path: Path) -> list[dict[str, Any]]:
    """Read audit JSONL events from disk."""

    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            loaded = json.loads(stripped)
            if not isinstance(loaded, dict):
                raise AuditWriteError(f"Audit line {line_number} is not an object.")
            events.append(loaded)
    return events


def _last_chain_state(path: Path) -> tuple[int, str | None]:
    if not path.exists():
        return 0, None
    events = read_audit_events(path)
    if not events:
        return 0, None
    last = events[-1]
    return int(last.get("event_index", len(events))), str(last["event_hash"])


def rule_result_json(result: SafetyRuleResult) -> dict[str, Any]:
    return {
        "rule_id": result.rule_id,
        "priority": result.priority,
        "evaluated": result.evaluated,
        "triggered": result.triggered,
        "action": result.action.value,
        "forced_state": result.forced_state.value if result.forced_state else None,
        "reason_code": result.reason_code,
        "evidence": to_jsonable(result.evidence),
    }


def active_strategy_for_context(context: AgentContext) -> tuple[str | None, str | None]:
    if context.current_state is None:
        return None, None
    version = context.active_strategies.get(context.current_state.value)
    return context.current_state.value, version


def first_reason(reason_codes: tuple[str, ...], context: AgentContext) -> str | None:
    if reason_codes:
        return reason_codes[0]
    return context.decision_reason


def safe_get(mapping: Mapping[str, Any], key: str) -> Any:
    return mapping.get(key)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(to_jsonable(key)): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    return value
