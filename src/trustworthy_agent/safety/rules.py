"""Mandatory SafetyGuard rules required by the specification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.safety.base import SafetyAction, SafetyRule, SafetyRuleResult
from trustworthy_agent.transitions.base import TransitionDecision

REQUIRED_RULE_IDS: tuple[str, ...] = (
    "R1_NO_DIAGNOSIS_WITHOUT_DATA",
    "R2_VALIDATION_REQUIRED",
    "R3_CONFIDENCE_REQUIRED",
    "R4_AUDIT_REQUIRED",
    "R5_MODEL_CONFLICT_ESCALATE",
    "R6_POOR_DATA_NO_DECISION",
    "R7_STATE_PATH_REQUIRED",
    "R8_JUSTIFICATION_REQUIRED",
    "R9_HIGH_RISK_BLOCK_UNSUPPORTED_AUTO_ACTION",
    "R10_OOD_ESCALATE_OR_REJECT",
    "R11_EXPLANATION_REQUIRED",
    "R12_INVALID_TRANSITION_REJECT",
    "R13_INVALID_SPLINE_FIT",
    "R14_MODEL_ARTIFACT_INVALID",
    "R15_LOW_CONFIDENCE_ESCALATE",
    "R16_REPRESENTATION_DISAGREEMENT_ESCALATE",
    "R17_CLASSIFIER_DISAGREEMENT_ESCALATE",
    "R18_LOW_DOMAIN_CONSISTENCY",
    "R19_COMBINED_FAULT_EXPERT_REVIEW",
    "R20_AUDIT_CHAIN_FAILURE",
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_data_quality": 0.75,
    "max_missing_ratio": 0.10,
    "max_risk_for_auto_recommendation": 0.60,
    "max_ood_score": 0.50,
    "min_explanation_score": 0.70,
    "min_confidence_for_recommendation": 0.70,
    "max_representation_disagreement": 0.35,
    "max_classifier_disagreement": 0.35,
    "min_domain_consistency_score": 0.60,
}

AUTO_RECOMMENDATION_ACTIONS: frozenset[str] = frozenset(
    {
        "CONTINUE_MONITORING",
        "INCREASE_MONITORING_FREQUENCY",
        "SCHEDULE_MECHANICAL_INSPECTION",
        "INSPECT_ELECTRICAL_CONDITION",
        "PLAN_INSPECTION",
        "IMMEDIATE_EXPERT_REVIEW",
    }
)


@dataclass(frozen=True)
class MandatorySafetyRule:
    """Configurable predicate rule used by SafetyGuard.

    Purpose:
        Implement one mandatory safety rule while keeping SafetyGuard separate
        from transition policies and state strategies.
    Parameters:
        rule_id: Stable required rule identifier.
        priority: Deterministic precedence priority.
        thresholds: Safety thresholds for the active profile.
    Return value:
        Safety rule instance.
    Raised exceptions:
        None during normal construction.
    Scientific assumptions:
        Threshold defaults are safety-control defaults, not validated operating
        limits.
    Side effects:
        None.
    Reproducibility implications:
        Rule ID, priority, thresholds, and evidence are recorded in results.
    """

    rule_id: str
    priority: int
    thresholds: Mapping[str, float]

    def evaluate(
        self,
        current_state: StateId,
        proposed_transition: TransitionDecision,
        context: AgentContext,
    ) -> SafetyRuleResult:
        """Evaluate this rule for a proposed transition."""

        triggered, action, forced_state, reason_code, evidence = evaluate_rule(
            self.rule_id,
            current_state,
            proposed_transition,
            context,
            self.thresholds,
        )
        return SafetyRuleResult(
            rule_id=self.rule_id,
            priority=self.priority,
            evaluated=True,
            triggered=triggered,
            action=action if triggered else SafetyAction.ALLOW,
            forced_state=forced_state,
            reason_code=reason_code,
            evidence=evidence,
        )


def default_safety_rules(
    *,
    thresholds: Mapping[str, float] | None = None,
    precedence: Sequence[str] = REQUIRED_RULE_IDS,
) -> tuple[SafetyRule, ...]:
    """Build mandatory safety rules with configurable rule precedence.

    Purpose:
        Provide the canonical SafetyGuard rule set without embedding it in any
        transition policy.
    Parameters:
        thresholds: Optional safety thresholds.
        precedence: Rule IDs ordered from lower to higher priority.
    Return value:
        Tuple of safety rules.
    Raised exceptions:
        ValueError if a required rule is missing from precedence.
    Scientific assumptions:
        Defaults are experimental safety-control defaults.
    Side effects:
        None.
    Reproducibility implications:
        The precedence order is explicit and serializable.
    """

    missing = set(REQUIRED_RULE_IDS) - set(precedence)
    if missing:
        raise ValueError(f"Safety precedence missing required rules: {sorted(missing)}")
    active_thresholds = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    return tuple(
        MandatorySafetyRule(rule_id=rule_id, priority=index, thresholds=active_thresholds)
        for index, rule_id in enumerate(precedence, start=1)
        if rule_id in REQUIRED_RULE_IDS
    )


def evaluate_rule(
    rule_id: str,
    current_state: StateId,
    proposed_transition: TransitionDecision,
    context: AgentContext,
    thresholds: Mapping[str, float],
) -> tuple[bool, SafetyAction, StateId | None, str | None, dict[str, Any]]:
    if rule_id == "R1_NO_DIAGNOSIS_WITHOUT_DATA":
        missing = not any(
            (
                context.raw_record_reference,
                context.raw_features,
                context.classical_features,
                context.derived_facts.get("input_available"),
            )
        )
        return triggered_result(missing, SafetyAction.FORCE_NO_DECISION, StateId.NO_DECISION)
    if rule_id == "R2_VALIDATION_REQUIRED":
        skipped = (
            current_state
            not in (
                StateId.DATA_ACQUISITION,
                StateId.DATA_VALIDATION,
            )
            and context.validation_report is None
            and not bool(context.derived_facts.get("validation_passed"))
        )
        return triggered_result(skipped, SafetyAction.FORCE_NO_DECISION, StateId.NO_DECISION)
    if rule_id == "R3_CONFIDENCE_REQUIRED":
        missing = (
            proposed_transition.selected_state == StateId.RECOMMENDATION
            and numeric_value(context, "confidence") is None
        )
        return triggered_result(missing, SafetyAction.FORCE_NO_DECISION, StateId.NO_DECISION)
    if rule_id == "R4_AUDIT_REQUIRED":
        unavailable = context_value(context, "audit_available", True) is False and (
            proposed_transition.selected_state == StateId.RECOMMENDATION
            or StateId.NO_DECISION in allowed_targets(context)
        )
        return triggered_result(unavailable, SafetyAction.FAIL_CLOSED, StateId.NO_DECISION)
    if rule_id == "R5_MODEL_CONFLICT_ESCALATE":
        conflict = bool(context_value(context, "spline_classifier_conflict", False)) and (
            proposed_transition.selected_state == StateId.RECOMMENDATION
            or StateId.ESCALATION in allowed_targets(context)
        )
        return triggered_result(conflict, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R6_POOR_DATA_NO_DECISION":
        data_quality = numeric_value(context, "data_quality")
        missing_ratio = numeric_value(context, "missing_ratio")
        poor = (
            (data_quality is not None and data_quality < thresholds["min_data_quality"])
            or (missing_ratio is not None and missing_ratio > thresholds["max_missing_ratio"])
        ) and (
            proposed_transition.selected_state == StateId.RECOMMENDATION
            or StateId.NO_DECISION in allowed_targets(context)
        )
        return triggered_result(poor, SafetyAction.FORCE_NO_DECISION, StateId.NO_DECISION)
    if rule_id == "R7_STATE_PATH_REQUIRED":
        invalid = not context.state_path or (
            context.current_state is not None and context.state_path[-1] != context.current_state
        )
        return triggered_result(invalid, SafetyAction.FAIL_CLOSED, StateId.NO_DECISION)
    if rule_id == "R8_JUSTIFICATION_REQUIRED":
        reasons = context_value(context, "reason_codes", None)
        missing = proposed_transition.selected_state == StateId.RECOMMENDATION and not any(
            (context.decision_reason, proposed_transition.decision_reason, reasons)
        )
        return triggered_result(missing, SafetyAction.FORCE_NO_DECISION, StateId.NO_DECISION)
    if rule_id == "R9_HIGH_RISK_BLOCK_UNSUPPORTED_AUTO_ACTION":
        risk = numeric_value(context, "risk_score")
        action = action_code(context)
        unsafe = (
            risk is not None
            and risk > thresholds["max_risk_for_auto_recommendation"]
            and proposed_transition.selected_state == StateId.RECOMMENDATION
            and action in (AUTO_RECOMMENDATION_ACTIONS | {None})
        )
        return triggered_result(unsafe, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R10_OOD_ESCALATE_OR_REJECT":
        ood = numeric_value(context, "ood_score")
        trigger = (
            ood is not None
            and ood > thresholds["max_ood_score"]
            and StateId.ESCALATION in allowed_targets(context)
        )
        return triggered_result(trigger, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R11_EXPLANATION_REQUIRED":
        score = numeric_value(context, "explanation_score")
        available = context_value(context, "explanation_available", None)
        required = bool(context_value(context, "explanation_required", True))
        missing = (
            required
            and proposed_transition.selected_state == StateId.RECOMMENDATION
            and (
                available is not True
                or score is None
                or score < thresholds["min_explanation_score"]
            )
        )
        return triggered_result(missing, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R12_INVALID_TRANSITION_REJECT":
        allowed = context_value(context, "allowed_transitions", None)
        invalid = isinstance(allowed, Sequence) and proposed_transition.selected_state not in {
            StateId(state) for state in allowed
        }
        return triggered_result(invalid, SafetyAction.BLOCK, current_state)
    if rule_id == "R13_INVALID_SPLINE_FIT":
        valid = context_value(context, "spline_fit_valid", True)
        trigger = valid is False and StateId.NO_DECISION in allowed_targets(context)
        return triggered_result(trigger, SafetyAction.FORCE_NO_DECISION, StateId.NO_DECISION)
    if rule_id == "R14_MODEL_ARTIFACT_INVALID":
        valid = context_value(context, "model_artifact_valid", True)
        trigger = valid is False and StateId.NO_DECISION in allowed_targets(context)
        return triggered_result(trigger, SafetyAction.FORCE_NO_DECISION, StateId.NO_DECISION)
    if rule_id == "R15_LOW_CONFIDENCE_ESCALATE":
        confidence = numeric_value(context, "confidence")
        low = (
            proposed_transition.selected_state == StateId.RECOMMENDATION
            and confidence is not None
            and confidence < thresholds["min_confidence_for_recommendation"]
        )
        return triggered_result(low, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R16_REPRESENTATION_DISAGREEMENT_ESCALATE":
        disagreement = numeric_value(context, "representation_disagreement")
        trigger = (
            disagreement is not None
            and disagreement > thresholds["max_representation_disagreement"]
            and StateId.ESCALATION in allowed_targets(context)
        )
        return triggered_result(trigger, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R17_CLASSIFIER_DISAGREEMENT_ESCALATE":
        disagreement = numeric_value(context, "classifier_disagreement")
        trigger = (
            disagreement is not None
            and disagreement > thresholds["max_classifier_disagreement"]
            and StateId.ESCALATION in allowed_targets(context)
        )
        return triggered_result(trigger, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R18_LOW_DOMAIN_CONSISTENCY":
        consistency = numeric_value(context, "domain_consistency_score")
        trigger = (
            proposed_transition.selected_state == StateId.RECOMMENDATION
            and consistency is not None
            and consistency < thresholds["min_domain_consistency_score"]
        )
        return triggered_result(trigger, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R19_COMBINED_FAULT_EXPERT_REVIEW":
        diagnosis = context_value(context, "diagnosis", None)
        trigger = diagnosis == "Mech_Elec_Damage" and proposed_transition.selected_state == (
            StateId.RECOMMENDATION
        )
        return triggered_result(trigger, SafetyAction.FORCE_ESCALATION, StateId.ESCALATION)
    if rule_id == "R20_AUDIT_CHAIN_FAILURE":
        valid = context_value(context, "audit_chain_valid", True)
        trigger = valid is False and StateId.NO_DECISION in allowed_targets(context)
        return triggered_result(trigger, SafetyAction.FAIL_CLOSED, StateId.NO_DECISION)
    return False, SafetyAction.ALLOW, None, None, {}


def triggered_result(
    triggered: bool,
    action: SafetyAction,
    forced_state: StateId,
) -> tuple[bool, SafetyAction, StateId | None, str | None, dict[str, Any]]:
    if not triggered:
        return False, SafetyAction.ALLOW, None, None, {}
    return True, action, forced_state, action.value, {"forced_state": forced_state.value}


def context_value(context: AgentContext, key: str, default: Any) -> Any:
    if hasattr(context, key):
        value = getattr(context, key)
        if value is not None and value != () and value != {}:
            return value
    return context.derived_facts.get(key, default)


def numeric_value(context: AgentContext, key: str) -> float | None:
    value = context_value(context, key, None)
    if isinstance(value, int | float):
        return float(value)
    return None


def action_code(context: AgentContext) -> str | None:
    value = context_value(context, "action_code", None)
    if isinstance(value, str):
        return value
    if isinstance(context.final_action, str):
        return context.final_action
    return None


def allowed_targets(context: AgentContext) -> set[StateId]:
    allowed = context_value(context, "allowed_transitions", ())
    if not isinstance(allowed, Sequence):
        return set()
    return {StateId(state) for state in allowed}
