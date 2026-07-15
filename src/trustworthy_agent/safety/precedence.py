"""Deterministic safety action precedence."""

from __future__ import annotations

from collections.abc import Mapping

from trustworthy_agent.safety.base import SafetyAction, SafetyRuleResult

PRECEDENCE: dict[SafetyAction, int] = {
    SafetyAction.ALLOW: 0,
    SafetyAction.BLOCK: 10,
    SafetyAction.FORCE_ESCALATION: 20,
    SafetyAction.FORCE_NO_DECISION: 30,
    SafetyAction.FAIL_CLOSED: 40,
}


def dominant_result(
    results: tuple[SafetyRuleResult, ...],
    action_precedence: Mapping[SafetyAction, int] | None = None,
) -> SafetyRuleResult | None:
    """Return the highest-precedence triggered safety result.

    Purpose:
        Make safety conflict resolution deterministic.
    Parameters:
        results: Evaluated rule results.
    Return value:
        Dominant triggered result, or `None` if no rule triggered.
    Raised exceptions:
        None.
    Scientific assumptions:
        Precedence is safety-control metadata, not a diagnostic score.
    Side effects:
        None.
    Reproducibility implications:
        Stable sorting prevents run-to-run variation in safety overrides.
    """

    precedence = action_precedence or PRECEDENCE
    triggered = [result for result in results if result.triggered]
    if not triggered:
        return None
    return max(triggered, key=lambda result: (precedence[result.action], result.priority))
