"""Versioned state profile definitions and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trustworthy_agent.agent.states import CANONICAL_SEMANTIC_TO_ORDINAL, StateId
from trustworthy_agent.exceptions import ConfigurationError, InvalidTransitionError


@dataclass(frozen=True)
class StateProfile:
    """Versioned finite-state profile.

    Purpose:
        Declare valid semantic states, ordinal metadata, transitions, and audit
        finalization state before any run starts.
    Parameters:
        profile_id: Immutable profile identifier.
        profile_version: Immutable profile version.
        semantic_to_ordinal_mapping: Mapping from semantic IDs to display labels.
        allowed_transitions: Declared outgoing transitions by semantic state.
        initial_state: First semantic state.
        terminal_states: Terminal outcome states before audit finalization.
        audit_state: Semantic audit state.
    Return value:
        Validated profile instance.
    Raised exceptions:
        ConfigurationError for invalid profile definitions.
    Scientific assumptions:
        None; profile semantics come from the normative documents.
    Side effects:
        None.
    Reproducibility implications:
        Versioned profile evidence can be included in resolved configs and audit.
    """

    profile_id: str
    profile_version: str
    semantic_to_ordinal_mapping: dict[StateId, str]
    allowed_transitions: dict[StateId, tuple[StateId, ...]]
    initial_state: StateId
    terminal_states: tuple[StateId, ...]
    audit_state: StateId

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> StateProfile:
        """Build a profile from structured configuration data.

        Purpose:
            Parse declarative profile data while keeping state ordinals as
            metadata and semantic identifiers as the executable state identity.
        Parameters:
            mapping: Structured profile mapping, typically from validated
                configuration.
        Return value:
            Validated `StateProfile`.
        Raised exceptions:
            ConfigurationError for missing or malformed profile fields.
        Scientific assumptions:
            None; this method handles workflow metadata only.
        Side effects:
            None.
        Reproducibility implications:
            Deterministic parsing supports profile fingerprinting and replay.
        """

        raw_mapping = _require_mapping(
            mapping.get("semantic_to_ordinal_mapping"), "semantic_to_ordinal_mapping"
        )
        semantic_to_ordinal_mapping = {
            _parse_state_id(state): _require_text(ordinal, "ordinal")
            for state, ordinal in raw_mapping.items()
        }

        raw_transitions = _require_mapping(
            mapping.get("allowed_transitions"), "allowed_transitions"
        )
        allowed_transitions = {
            _parse_state_id(source): tuple(
                _parse_state_id(target) for target in _require_sequence(targets, "targets")
            )
            for source, targets in raw_transitions.items()
        }

        terminal_states = tuple(
            _parse_state_id(state)
            for state in _require_sequence(mapping.get("terminal_states"), "terminal_states")
        )

        return cls(
            profile_id=_require_text(mapping.get("profile_id"), "profile_id"),
            profile_version=_require_text(mapping.get("profile_version"), "profile_version"),
            semantic_to_ordinal_mapping=semantic_to_ordinal_mapping,
            allowed_transitions=allowed_transitions,
            initial_state=_parse_state_id(mapping.get("initial_state")),
            terminal_states=terminal_states,
            audit_state=_parse_state_id(mapping.get("audit_state")),
        )

    def validate(self) -> None:
        """Validate profile integrity before a run can start.

        Purpose:
            Reject undeclared states, duplicate ordinals, invalid targets, and
            terminal states that cannot reach audit finalization.
        Parameters:
            None.
        Return value:
            None.
        Raised exceptions:
            ConfigurationError on invalid profile structure.
        Scientific assumptions:
            None.
        Side effects:
            None.
        Reproducibility implications:
            Fails fast before audit/replay evidence can diverge.
        """

        states = set(self.semantic_to_ordinal_mapping)
        ordinals = tuple(self.semantic_to_ordinal_mapping.values())
        if len(ordinals) != len(set(ordinals)):
            raise ConfigurationError("State profile ordinal labels must be unique.")
        if self.initial_state not in states:
            raise ConfigurationError("Initial state must be declared in the profile.")
        if self.audit_state not in states:
            raise ConfigurationError("Audit state must be declared in the profile.")
        for source, targets in self.allowed_transitions.items():
            if source not in states:
                raise ConfigurationError(f"Transition source is undeclared: {source}")
            for target in targets:
                if target not in states:
                    raise ConfigurationError(f"Transition target is undeclared: {target}")
        for terminal in self.terminal_states:
            if terminal not in states:
                raise ConfigurationError(f"Terminal state is undeclared: {terminal}")
            if terminal != self.audit_state and self.audit_state not in self.allowed_next_states(
                terminal
            ):
                raise ConfigurationError(f"Terminal state cannot reach audit: {terminal}")

    def allowed_next_states(self, state: StateId) -> tuple[StateId, ...]:
        """Return declared outgoing transitions for a semantic state.

        Purpose:
            Ensure transition policies evaluate only active-profile targets.
        Parameters:
            state: Current semantic state.
        Return value:
            Tuple of allowed target states.
        Raised exceptions:
            InvalidTransitionError if the source state is undeclared.
        Scientific assumptions:
            None.
        Side effects:
            None.
        Reproducibility implications:
            Keeps candidate transition sets deterministic.
        """

        if state not in self.semantic_to_ordinal_mapping:
            raise InvalidTransitionError(f"State is not declared in profile: {state}")
        return self.allowed_transitions.get(state, ())

    def validate_transition(self, source: StateId, target: StateId) -> None:
        """Reject transitions outside the active profile graph.

        Purpose:
            Centralize graph validation for transition policies and SafetyGuard
            outputs.
        Parameters:
            source: Current semantic state.
            target: Proposed target semantic state.
        Return value:
            None.
        Raised exceptions:
            InvalidTransitionError when the target is not declared as allowed.
        Scientific assumptions:
            None.
        Side effects:
            None.
        Reproducibility implications:
            Ensures every accepted edge is profile-versioned evidence.
        """

        allowed = self.allowed_next_states(source)
        if target not in allowed:
            raise InvalidTransitionError(
                f"Transition {source} -> {target} is not allowed by profile "
                f"{self.profile_id}@{self.profile_version}."
            )

    def requires_audit_finalization(self, state: StateId) -> bool:
        """Report whether entering `state` should finalize audit.

        Purpose:
            Keep terminal audit handling explicit in the engine loop.
        Parameters:
            state: Current semantic state.
        Return value:
            `True` only for the configured audit state.
        Raised exceptions:
            None.
        Scientific assumptions:
            None.
        Side effects:
            None.
        Reproducibility implications:
            Provides a stable audit finalization boundary.
        """

        return state == self.audit_state

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this profile.

        Purpose:
            Provide deterministic profile evidence for manifests and audit.
        Parameters:
            None.
        Return value:
            Mapping with semantic state values and ordinal metadata.
        Raised exceptions:
            None.
        Scientific assumptions:
            None.
        Side effects:
            None.
        Reproducibility implications:
            Keeps persisted profile evidence independent of enum internals.
        """

        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "semantic_to_ordinal_mapping": {
                state.value: ordinal for state, ordinal in self.semantic_to_ordinal_mapping.items()
            },
            "allowed_transitions": {
                source.value: [target.value for target in targets]
                for source, targets in self.allowed_transitions.items()
            },
            "initial_state": self.initial_state.value,
            "terminal_states": [state.value for state in self.terminal_states],
            "audit_state": self.audit_state.value,
        }


def canonical_diagnostic_profile() -> StateProfile:
    """Create the canonical `diagnostic_full_v1` state profile.

    Purpose:
        Provide the baseline profile declared by the specification.
    Parameters:
        None.
    Return value:
        Validated `StateProfile`.
    Raised exceptions:
        ConfigurationError if the hard-coded scaffold profile is invalid.
    Scientific assumptions:
        None; this is workflow control only.
    Side effects:
        None.
    Reproducibility implications:
        Profile ID/version can be recorded in every run.
    """

    return StateProfile(
        profile_id="diagnostic_full_v1",
        profile_version="1.0.0",
        semantic_to_ordinal_mapping=dict(CANONICAL_SEMANTIC_TO_ORDINAL),
        allowed_transitions={
            StateId.DATA_ACQUISITION: (StateId.DATA_VALIDATION,),
            StateId.DATA_VALIDATION: (
                StateId.FEATURE_EXTRACTION,
                StateId.DATA_ACQUISITION,
                StateId.NO_DECISION,
            ),
            StateId.FEATURE_EXTRACTION: (StateId.SPLINE_MODELLING, StateId.NO_DECISION),
            StateId.SPLINE_MODELLING: (
                StateId.DIAGNOSTIC_INFERENCE,
                StateId.ESCALATION,
                StateId.NO_DECISION,
            ),
            StateId.DIAGNOSTIC_INFERENCE: (StateId.EXPLANATION_GENERATION, StateId.ESCALATION),
            StateId.EXPLANATION_GENERATION: (
                StateId.DECISION_CHECK,
                StateId.ESCALATION,
                StateId.NO_DECISION,
            ),
            StateId.DECISION_CHECK: (
                StateId.RECOMMENDATION,
                StateId.ESCALATION,
                StateId.NO_DECISION,
            ),
            StateId.RECOMMENDATION: (StateId.AUDIT,),
            StateId.ESCALATION: (StateId.AUDIT,),
            StateId.NO_DECISION: (StateId.AUDIT,),
            StateId.AUDIT: (),
        },
        initial_state=StateId.DATA_ACQUISITION,
        terminal_states=(
            StateId.RECOMMENDATION,
            StateId.ESCALATION,
            StateId.NO_DECISION,
            StateId.AUDIT,
        ),
        audit_state=StateId.AUDIT,
    )


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"State profile field `{field_name}` must be a non-empty string.")
    return value


def _require_mapping(value: object, field_name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"State profile field `{field_name}` must be a mapping.")
    return value


def _require_sequence(value: object, field_name: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigurationError(f"State profile field `{field_name}` must be a sequence.")
    return value


def _parse_state_id(value: object) -> StateId:
    try:
        return StateId(_require_text(value, "state_id"))
    except ValueError as exc:
        raise ConfigurationError(f"Unknown semantic state identifier: {value}") from exc
