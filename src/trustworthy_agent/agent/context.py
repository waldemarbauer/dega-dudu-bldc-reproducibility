"""Agent context passed between strategies, policies, safety, and audit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from trustworthy_agent.agent.profiles import StateProfile
from trustworthy_agent.agent.state_result import StateResult
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.exceptions import ConfigurationError


@dataclass(frozen=True)
class AgentContext:
    """Immutable decision context with explicit unknown values.

    Purpose:
        Hold the canonical fields required by the specification without using
        semantically meaningful default values for unknown evidence.
    Parameters:
        Fields correspond to the canonical context contract; unknown values are
        represented as `None` or empty audit/provenance collections.
    Return value:
        Immutable context instance.
    Raised exceptions:
        None.
    Scientific assumptions:
        `None` means unavailable/unknown, not zero, normal, or healthy.
    Side effects:
        None.
    Reproducibility implications:
        Immutable updates prevent hidden mutation across state execution.
    """

    run_id: str | None = None
    experiment_id: str | None = None
    experiment_fingerprint: str | None = None
    case_id: str | None = None
    window_id: str | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    dataset_checksum: str | None = None
    state_profile_id: str | None = None
    current_state: StateId | None = None
    state_path: tuple[StateId, ...] = ()
    raw_record_reference: str | None = None
    raw_features: dict[str, Any] | None = None
    classical_features: dict[str, Any] | None = None
    spline_features: dict[str, Any] | None = None
    data_quality: float | None = None
    missing_ratio: float | None = None
    validation_report: dict[str, Any] | None = None
    diagnosis: str | None = None
    class_probabilities: dict[str, float] | None = None
    confidence: float | None = None
    explanation_available: bool | None = None
    explanation_text: str | None = None
    top_explanatory_features: tuple[str, ...] = ()
    explanation_score: float | None = None
    domain_consistency_score: float | None = None
    model_agreement: float | None = None
    spline_classifier_conflict: bool | None = None
    risk_score: float | None = None
    ood_score: float | None = None
    transition_history: tuple[Any, ...] = ()
    safety_overrides: tuple[Any, ...] = ()
    final_action: str | None = None
    decision_reason: str | None = None
    active_strategies: dict[str, str] = field(default_factory=dict)
    active_transition_policy: str | None = None
    resolved_config_hash: str | None = None
    git_commit: str | None = None
    environment_fingerprint: str | None = None
    artifact_refs: tuple[str, ...] = ()
    derived_facts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def with_current_state(self, state: StateId) -> AgentContext:
        """Return a context updated to the current semantic state.

        Purpose:
            Record state entry without mutating prior context.
        Parameters:
            state: Semantic state being entered.
        Return value:
            New context with `current_state` and appended `state_path`.
        Raised exceptions:
            None.
        Scientific assumptions:
            None.
        Side effects:
            None.
        Reproducibility implications:
            Preserves complete path evidence for audit and replay.
        """

        return replace(self, current_state=state, state_path=(*self.state_path, state))

    def validate_for_profile(self, profile: StateProfile) -> None:
        """Validate context state evidence against a state profile.

        Purpose:
            Detect hidden or inconsistent state evidence before strategies,
            policies, safety rules, or audit consume the context.
        Parameters:
            profile: Active versioned state profile.
        Return value:
            None.
        Raised exceptions:
            ConfigurationError for undeclared states or inconsistent path data.
        Scientific assumptions:
            None; this method validates workflow evidence, not diagnostic facts.
        Side effects:
            None.
        Reproducibility implications:
            Prevents malformed state paths from entering replay/audit evidence.
        """

        declared_states = set(profile.semantic_to_ordinal_mapping)
        if self.current_state is not None and self.current_state not in declared_states:
            raise ConfigurationError(f"Current state is not declared: {self.current_state}")
        for state in self.state_path:
            if state not in declared_states:
                raise ConfigurationError(f"State path contains undeclared state: {state}")
        if self.current_state is None:
            return
        if not self.state_path:
            raise ConfigurationError("Current state requires a non-empty state path.")
        if self.state_path[-1] != self.current_state:
            raise ConfigurationError("Current state must match the last state path entry.")

    def apply_state_result(self, result: StateResult) -> AgentContext:
        """Merge structured state facts into a new context.

        Purpose:
            Carry state outputs forward while keeping transitions outside
            strategy execution.
        Parameters:
            result: Facts and provenance produced by a strategy.
        Return value:
            New context with merged `derived_facts` and provenance.
        Raised exceptions:
            None.
        Scientific assumptions:
            This method does not assign meaning to missing values or compute
            diagnostic results.
        Side effects:
            None.
        Reproducibility implications:
            Maintains provenance alongside derived facts.
        """

        return replace(
            self,
            derived_facts={**self.derived_facts, **dict(result.facts)},
            provenance={**self.provenance, **dict(result.provenance)},
        )
