"""Finite-state engine skeleton for the diagnostic workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.profiles import StateProfile
from trustworthy_agent.agent.registry import StrategyRegistry
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.audit.logger import AuditRecorder, fail_closed_context
from trustworthy_agent.exceptions import AuditWriteError, ConfigurationError, InvalidTransitionError
from trustworthy_agent.safety.guards import SafetyGuard
from trustworthy_agent.transitions.base import RandomGenerator, TransitionPolicy
from trustworthy_agent.transitions.registry import TransitionPolicyRegistry


@dataclass(frozen=True)
class AgentRuntime:
    """Runtime collaborators required by the FSM engine.

    Purpose:
        Group explicit dependencies so the engine coordinates interfaces instead
        of constructing concrete strategies, policies, safety rules, or audit
        stores.
    Parameters:
        profile, strategy registry, strategy selection, transition-policy
        registry/selection, safety guard, audit recorder, RNG, and resolved
        config.
    Return value:
        Immutable runtime dependency bundle.
    Raised exceptions:
        None.
    Scientific assumptions:
        None; algorithms live behind the supplied interfaces.
    Side effects:
        None during construction.
    Reproducibility implications:
        Explicit dependencies make run configuration auditable.
    """

    profile: StateProfile
    strategy_registry: StrategyRegistry
    strategy_selection: Mapping[StateId, tuple[str, str]]
    transition_policy_registry: TransitionPolicyRegistry
    transition_policy_selection: tuple[str, str]
    safety_guard: SafetyGuard
    audit: AuditRecorder
    rng: RandomGenerator
    config: Mapping[str, Any]


class AgentEngine:
    """FSM coordinator that depends only on architectural interfaces.

    Purpose:
        Execute the declared state profile by invoking strategies, transition
        policy, SafetyGuard, and audit recorder in the required order.
    Parameters:
        runtime: Explicit runtime collaborators.
    Return value:
        Engine instance.
    Raised exceptions:
        ConfigurationError or InvalidTransitionError for invalid runtime/profile
        evidence; collaborator exceptions are not silently swallowed.
    Scientific assumptions:
        The engine makes no diagnostic assumptions and imports no concrete S3-S7
        strategies.
    Side effects:
        Only through the supplied audit recorder and strategy/policy adapters.
    Reproducibility implications:
        Preserves the ordered execution evidence needed for replay.
    """

    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime
        policy_name, policy_version = runtime.transition_policy_selection
        self._transition_policy: TransitionPolicy = runtime.transition_policy_registry.resolve(
            policy_name, policy_version
        ).create()

    def run_context(self, context: AgentContext, max_steps: int = 100) -> AgentContext:
        """Run one context through the configured state profile.

        Purpose:
            Provide the architecture loop without implementing scientific
            strategy algorithms.
        Parameters:
            context: Initial case context.
            max_steps: Guard against malformed profiles that cycle forever.
        Return value:
            Audited terminal context from the audit recorder.
        Raised exceptions:
            ConfigurationError when no strategy is selected; InvalidTransitionError
            when SafetyGuard returns a state outside declared transitions.
        Scientific assumptions:
            None.
        Side effects:
            Delegated to strategy execution and audit recorder interfaces.
        Reproducibility implications:
            Enforces state path recording and transition evidence ordering.
        """

        state = self._runtime.profile.initial_state
        working_context = context
        for _ in range(max_steps):
            working_context = working_context.with_current_state(state)
            working_context.validate_for_profile(self._runtime.profile)
            if self._runtime.profile.requires_audit_finalization(state):
                try:
                    return self._runtime.audit.finalize_outcome(working_context)
                except AuditWriteError:
                    return fail_closed_context(working_context)

            selection = self._runtime.strategy_selection.get(state)
            if selection is None:
                raise ConfigurationError(f"No strategy selected for state {state}.")
            strategy_name, strategy_version = selection
            descriptor = self._runtime.strategy_registry.resolve(
                state=state,
                strategy_name=strategy_name,
                version=strategy_version,
            )
            strategy = descriptor.create()
            strategy.validate_config(self._runtime.config)
            result = strategy.execute(working_context)
            working_context = working_context.apply_state_result(result)
            self._runtime.audit.record_state_execution(working_context, result)

            allowed = self._runtime.profile.allowed_next_states(state)
            proposed = self._transition_policy.decide(
                current_state=state,
                allowed_transitions=allowed,
                context=working_context,
                rng=self._runtime.rng,
            )
            if proposed.selected_state not in allowed:
                raise InvalidTransitionError(
                    f"Policy selected {proposed.selected_state}, allowed targets are {allowed}."
                )
            self._runtime.profile.validate_transition(state, proposed.selected_state)
            guarded = self._runtime.safety_guard.evaluate(state, proposed, working_context)
            if guarded.final_state not in allowed:
                raise InvalidTransitionError(
                    f"SafetyGuard selected {guarded.final_state}, allowed targets are {allowed}."
                )
            self._runtime.profile.validate_transition(state, guarded.final_state)
            self._runtime.audit.record_transition(working_context, guarded)
            state = guarded.final_state

        raise ConfigurationError("FSM execution exceeded max_steps; profile may contain a cycle.")
