"""Transition policy protocol and decision evidence."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.exceptions import TransitionPolicyError


class RandomGenerator(Protocol):
    """Minimal injected random generator protocol.

    Purpose:
        Avoid hidden global RNG state in transition selection.
    Parameters:
        Implementations provide random methods as needed by policies.
    Return value:
        Protocol type.
    Raised exceptions:
        Implementation-specific.
    Scientific assumptions:
        None.
    Side effects:
        Limited to the injected RNG stream.
    Reproducibility implications:
        Enables recording seed/stream identity for stochastic transitions.
    """

    @property
    def stream_id(self) -> str:
        """Stable identifier recorded with stochastic transition evidence."""

    def random(self) -> float:
        """Return the next deterministic pseudo-random value in `[0.0, 1.0)`."""


@dataclass(frozen=True)
class TransitionDecision:
    """Full evidence for a proposed transition.

    Purpose:
        Carry candidate states, scores, normalized probabilities, selected
        state, policy identity, RNG identity, and safety placeholders.
    Parameters:
        Fields correspond to the transition decision contract in `SPEC.md`.
    Return value:
        Immutable transition evidence.
    Raised exceptions:
        TransitionPolicyError for invalid candidate/probability evidence.
    Scientific assumptions:
        Probabilities are workflow control probabilities, not diagnostic
        confidence or physical risk.
    Side effects:
        None.
    Reproducibility implications:
        Complete decision evidence can be audited and replayed.
    """

    current_state: StateId
    candidate_states: tuple[StateId, ...]
    raw_scores: dict[StateId, float]
    probabilities: dict[StateId, float]
    selected_state: StateId
    selection_mode: str
    policy_name: str
    policy_version: str
    rng_seed_or_stream_id: str | None
    safety_override: str | None = None
    triggered_rule_ids: tuple[str, ...] = ()
    decision_reason: str | None = None
    numerical_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        missing = set(self.candidate_states) - set(self.probabilities)
        if missing:
            raise TransitionPolicyError(f"Missing probabilities for candidates: {missing}")
        if self.selected_state not in self.candidate_states:
            raise TransitionPolicyError("Selected state must be one of the candidate states.")
        probability_sum = 0.0
        for state in self.candidate_states:
            probability = self.probabilities[state]
            if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise TransitionPolicyError(
                    f"Invalid transition probability for {state}: {probability}"
                )
            probability_sum += probability
        if not math.isclose(probability_sum, 1.0, abs_tol=self.numerical_tolerance):
            raise TransitionPolicyError(
                f"Transition probabilities must sum to 1; got {probability_sum}."
            )


class TransitionPolicy(Protocol):
    """Protocol for replaceable transition policies.

    Purpose:
        Let the FSM engine depend on a stable transition interface rather than a
        concrete Markov implementation.
    Parameters:
        Implementations expose name/version and decide from allowed states.
    Return value:
        Protocol type.
    Raised exceptions:
        TransitionPolicyError for invalid policy evidence.
    Scientific assumptions:
        Markov interpretation is conditional workflow control, not proof of
        physical process dynamics.
    Side effects:
        None required; persistence belongs to audit services.
    Reproducibility implications:
        The injected RNG and policy identity make stochastic decisions replayable.
    """

    policy_name: str
    policy_version: str

    def decide(
        self,
        current_state: StateId,
        allowed_transitions: Sequence[StateId],
        context: AgentContext,
        rng: RandomGenerator,
    ) -> TransitionDecision:
        """Return a proposed transition over only allowed target states."""


class TransitionPolicyABC(ABC):
    """Abstract base class for nominal transition policy implementations.

    Purpose:
        Define a conventional inheritance contract without forcing the engine to
        import any concrete transition-policy implementation.
    Parameters:
        Implementations provide stable policy identity and transition proposal
        evidence.
    Return value:
        Abstract base class.
    Raised exceptions:
        TypeError if instantiated without required implementation members.
    Scientific assumptions:
        Markov interpretation is conditional workflow control, not physical
        degradation evidence.
    Side effects:
        Subclasses must not bypass SafetyGuard.
    Reproducibility implications:
        Injected RNG and policy identity make stochastic selection replayable.
    """

    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Stable transition-policy name."""

    @property
    @abstractmethod
    def policy_version(self) -> str:
        """Stable transition-policy version."""

    @abstractmethod
    def decide(
        self,
        current_state: StateId,
        allowed_transitions: Sequence[StateId],
        context: AgentContext,
        rng: RandomGenerator,
    ) -> TransitionDecision:
        """Return a proposed transition over only allowed target states."""


def unavailable_policy_decision(*_: Any, **__: Any) -> TransitionDecision:
    """Raise for policy classes whose algorithms are out of Phase 1 scope.

    Purpose:
        Prevent accidental fake transition behavior.
    Parameters:
        Arbitrary positional and keyword arguments from policy callers.
    Return value:
        Never returns.
    Raised exceptions:
        NotImplementedError always.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Makes unimplemented behavior explicit and fail-fast.
    """

    raise NotImplementedError("Transition policy algorithm is not implemented in Phase 1.")
