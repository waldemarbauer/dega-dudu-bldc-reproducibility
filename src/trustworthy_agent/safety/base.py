"""Safety rule protocol and result structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.transitions.base import TransitionDecision


class SafetyAction(StrEnum):
    """Actions a safety rule may return."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    FORCE_ESCALATION = "FORCE_ESCALATION"
    FORCE_NO_DECISION = "FORCE_NO_DECISION"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True)
class SafetyRuleResult:
    """Structured result from one safety rule evaluation.

    Purpose:
        Record evaluated and triggered safety evidence without prose parsing.
    Parameters:
        rule_id, priority, evaluated, triggered, action, optional forced state,
        reason code, and structured evidence.
    Return value:
        Immutable safety rule result.
    Raised exceptions:
        None.
    Scientific assumptions:
        Evidence is only as valid as the producing rule; this class does not
        infer diagnostic meaning.
    Side effects:
        None.
    Reproducibility implications:
        Stable fields can be serialized into audit events.
    """

    rule_id: str
    priority: int
    evaluated: bool
    triggered: bool
    action: SafetyAction
    forced_state: StateId | None = None
    reason_code: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


class SafetyRule(Protocol):
    """Protocol for SafetyGuard rules."""

    @property
    def rule_id(self) -> str:
        """Stable safety rule identifier."""

    @property
    def priority(self) -> int:
        """Deterministic safety precedence priority."""

    def evaluate(
        self,
        current_state: StateId,
        proposed_transition: TransitionDecision,
        context: AgentContext,
    ) -> SafetyRuleResult:
        """Evaluate one rule for the proposed transition."""


class SafetyRuleABC(ABC):
    """Abstract base class for nominal SafetyGuard rule implementations.

    Purpose:
        Provide a conventional inheritance contract while SafetyGuard continues
        to evaluate rules through the stable protocol.
    Parameters:
        Implementations expose deterministic rule identity and evaluation.
    Return value:
        Abstract base class.
    Raised exceptions:
        TypeError if instantiated without required members.
    Scientific assumptions:
        Rule subclasses must document scientific assumptions and thresholds.
    Side effects:
        Rule implementations must not perform transitions directly; they return
        structured safety evidence for SafetyGuard.
    Reproducibility implications:
        Rule ID and priority support replayable safety precedence.
    """

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Stable safety rule identifier."""

    @property
    @abstractmethod
    def priority(self) -> int:
        """Deterministic safety precedence priority."""

    @abstractmethod
    def evaluate(
        self,
        current_state: StateId,
        proposed_transition: TransitionDecision,
        context: AgentContext,
    ) -> SafetyRuleResult:
        """Evaluate one rule for the proposed transition."""


@dataclass(frozen=True)
class GuardedTransition:
    """Final transition after SafetyGuard evaluation.

    Purpose:
        Keep proposed and safety-gated transition evidence together.
    Parameters:
        proposed_transition, final_state, evaluated rules, triggered rules, and
        override action.
    Return value:
        Immutable guarded transition.
    Raised exceptions:
        None.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Provides the evidence unit later persisted by audit.
    """

    proposed_transition: TransitionDecision
    final_state: StateId
    safety_rules_evaluated: tuple[SafetyRuleResult, ...]
    safety_rules_triggered: tuple[SafetyRuleResult, ...]
    safety_override: SafetyAction | None
