"""Static transition-matrix policy for workflow-control probabilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.exceptions import TransitionPolicyError
from trustworthy_agent.transitions.base import RandomGenerator, TransitionDecision
from trustworthy_agent.transitions.selection import select_state


class StaticTransitionMatrixPolicy:
    """Use a fixed, versioned state-to-state transition-probability table.

    Purpose:
        Implement static workflow transition probabilities over only the active
        profile's allowed outgoing states, without estimating any parameter from
        validation or test data.
    Parameters:
        transition_matrix: Mapping from current state to target-state
            probabilities.
        policy_id: Stable configured policy ID.
        policy_version: Version of the static matrix configuration.
        selection_mode: `argmax` or `seeded_sampling`.
        numerical_tolerance: Absolute tolerance for row-sum validation.
    Return value:
        Transition policy instance.
    Raised exceptions:
        TransitionPolicyError for invalid, missing, forbidden, or non-normalized
        probability rows.
    Scientific assumptions:
        Matrix probabilities describe conditional workflow control, not physical
        degradation dynamics or diagnostic confidence.
    Side effects:
        None, except seeded sampling advances the injected RNG.
    Reproducibility implications:
        Matrix and selection mode are explicit and replayable.
    """

    def __init__(
        self,
        transition_matrix: Mapping[StateId | str, Mapping[StateId | str, float]] | None = None,
        *,
        policy_id: str = "StaticTransitionMatrixPolicy",
        policy_version: str = "1.0.0",
        selection_mode: str = "argmax",
        numerical_tolerance: float = 1e-9,
    ) -> None:
        self.policy_name = policy_id
        self.policy_version = policy_version
        self.selection_mode = selection_mode
        self.numerical_tolerance = numerical_tolerance
        self.transition_matrix = {
            StateId(source): {
                StateId(target): _validate_probability(float(probability), StateId(target))
                for target, probability in row.items()
            }
            for source, row in (transition_matrix or {}).items()
        }
        if selection_mode not in {"argmax", "seeded_sampling"}:
            raise TransitionPolicyError(f"Unknown transition selection mode: {selection_mode}")

    @classmethod
    def from_config_file(
        cls,
        path: Path,
        *,
        selection_mode: str | None = None,
    ) -> StaticTransitionMatrixPolicy:
        """Build a static policy from a versioned YAML matrix.

        Purpose:
            Resolve the executable policy from declarative configuration without
            arbitrary Python execution.
        Parameters:
            path: YAML configuration path.
            selection_mode: Optional explicit mode, usually used only in tests.
        Return value:
            Configured static transition-matrix policy.
        Raised exceptions:
            TransitionPolicyError for malformed configuration.
        Scientific assumptions:
            Configured probabilities are workflow-control defaults.
        Side effects:
            Reads the supplied YAML file.
        Reproducibility implications:
            Policy ID, version, mode, and tolerance come from the frozen
            configuration.
        """

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise TransitionPolicyError("Static transition matrix config must be a mapping.")
        selection = _mapping(loaded.get("selection"), "selection")
        matrix_config = _mapping(loaded.get("matrix"), "matrix")
        matrix: dict[str, dict[str, float]] = {}
        for source, row_config in matrix_config.items():
            row = _mapping(row_config, f"matrix.{source}")
            transitions = _mapping(row.get("transitions"), f"matrix.{source}.transitions")
            matrix[str(source)] = {
                str(target): float(probability) for target, probability in transitions.items()
            }
        return cls(
            matrix,
            policy_id=str(loaded.get("policy_id") or loaded.get("policy_name")),
            policy_version=str(loaded["policy_version"]),
            selection_mode=selection_mode or str(selection.get("default_mode", "argmax")),
            numerical_tolerance=float(selection.get("numerical_tolerance", 1e-9)),
        )

    def decide(
        self,
        current_state: StateId,
        allowed_transitions: Sequence[StateId],
        context: AgentContext,
        rng: RandomGenerator,
    ) -> TransitionDecision:
        raw_scores = self.transition_matrix.get(current_state)
        if raw_scores is None:
            raise TransitionPolicyError(f"Static matrix row is undeclared: {current_state.value}")
        allowed = tuple(allowed_transitions)
        row_states = tuple(raw_scores)
        forbidden = set(row_states) - set(allowed)
        if forbidden:
            raise TransitionPolicyError(
                f"Static matrix contains forbidden transition targets: {forbidden}"
            )
        missing = set(allowed) - set(row_states)
        if missing:
            raise TransitionPolicyError(f"Static matrix row omits allowed targets: {missing}")
        probabilities = {state: raw_scores[state] for state in allowed}
        probability_sum = sum(probabilities.values())
        if not math.isclose(probability_sum, 1.0, abs_tol=self.numerical_tolerance):
            raise TransitionPolicyError(
                f"Static matrix row for {current_state.value} must sum to 1; got {probability_sum}."
            )
        selected = select_state(probabilities, self.selection_mode, rng)
        return TransitionDecision(
            current_state=current_state,
            candidate_states=allowed,
            raw_scores=dict(raw_scores),
            probabilities=probabilities,
            selected_state=selected,
            selection_mode=self.selection_mode,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            rng_seed_or_stream_id=rng.stream_id
            if self.selection_mode == "seeded_sampling"
            else None,
            decision_reason="STATIC_TRANSITION_MATRIX_SELECTION",
            numerical_tolerance=self.numerical_tolerance,
        )


class StaticMarkovPolicy(StaticTransitionMatrixPolicy):
    """Backward-compatible name for the static transition-matrix policy."""

    def __init__(
        self,
        transition_matrix: Mapping[StateId | str, Mapping[StateId | str, float]] | None = None,
        *,
        selection_mode: str = "argmax",
    ) -> None:
        super().__init__(
            transition_matrix,
            policy_id="static_markov",
            policy_version="1.0.0",
            selection_mode=selection_mode,
        )


def _validate_probability(probability: float, target: StateId) -> float:
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise TransitionPolicyError(
            f"Invalid static transition probability for {target.value}: {probability}"
        )
    return probability


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TransitionPolicyError(f"Static policy config field `{field_name}` must be a mapping.")
    return value
