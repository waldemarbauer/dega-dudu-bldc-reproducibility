"""Authoritative SafetyGuard implementation boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.safety.base import (
    GuardedTransition,
    SafetyAction,
    SafetyRule,
    SafetyRuleResult,
)
from trustworthy_agent.safety.precedence import dominant_result
from trustworthy_agent.transitions.base import TransitionDecision


@dataclass(frozen=True)
class SafetyGuard:
    """Evaluate safety rules outside transition policies and strategies.

    Purpose:
        Preserve SafetyGuard supremacy by making it the only component that can
        override proposed transitions.
    Parameters:
        rules: Safety rules sorted deterministically by priority and ID.
    Return value:
        SafetyGuard instance.
    Raised exceptions:
        Rule-specific exceptions are allowed to propagate so failures are not
        silently swallowed.
    Scientific assumptions:
        Rule implementations must document their scientific assumptions.
    Side effects:
        None; audit persistence is handled by an audit service.
    Reproducibility implications:
        Deterministic rule order supports replayable safety evidence.
    """

    rules: tuple[SafetyRule, ...]
    action_precedence: Mapping[SafetyAction, int] | None = None
    fail_closed_on_rule_error: bool = True

    def evaluate(
        self,
        current_state: StateId,
        proposed_transition: TransitionDecision,
        context: AgentContext,
    ) -> GuardedTransition:
        """Evaluate all rules and return the final guarded transition.

        Purpose:
            Ensure both evaluated and triggered rules are represented.
        Parameters:
            current_state: Current semantic FSM state.
            proposed_transition: Policy proposal to gate.
            context: Current agent evidence context.
        Return value:
            Guarded transition with final state and override evidence.
        Raised exceptions:
            Rule-specific exceptions are propagated.
        Scientific assumptions:
            None in the guard itself.
        Side effects:
            None.
        Reproducibility implications:
            Stable final-state selection can be audited.
        """

        ordered_rules = tuple(sorted(self.rules, key=lambda rule: (rule.priority, rule.rule_id)))
        evaluated: list[SafetyRuleResult] = []
        for rule in ordered_rules:
            try:
                evaluated.append(rule.evaluate(current_state, proposed_transition, context))
            except Exception as exc:
                if not self.fail_closed_on_rule_error:
                    raise
                evaluated.append(
                    SafetyRuleResult(
                        rule_id=f"{rule.rule_id}_ERROR",
                        priority=rule.priority,
                        evaluated=True,
                        triggered=True,
                        action=SafetyAction.FAIL_CLOSED,
                        forced_state=StateId.NO_DECISION,
                        reason_code="SAFETY_RULE_ERROR_FAIL_CLOSED",
                        evidence={"error_type": type(exc).__name__, "error_message": str(exc)},
                    )
                )
        results = tuple(evaluated)
        dominant = dominant_result(results, self.action_precedence)
        override: SafetyAction | None
        if dominant is None or dominant.action == SafetyAction.ALLOW:
            final_state = proposed_transition.selected_state
            override = None
        elif dominant.action == SafetyAction.FAIL_CLOSED:
            final_state = dominant.forced_state or StateId.NO_DECISION
            override = dominant.action
        elif dominant.action == SafetyAction.BLOCK:
            final_state = dominant.forced_state or current_state
            override = dominant.action
        else:
            final_state = dominant.forced_state or proposed_transition.selected_state
            override = dominant.action
        return GuardedTransition(
            proposed_transition=proposed_transition,
            final_state=final_state,
            safety_rules_evaluated=results,
            safety_rules_triggered=tuple(result for result in results if result.triggered),
            safety_override=override,
        )
