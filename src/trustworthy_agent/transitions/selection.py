"""Transition score normalization and selection helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from trustworthy_agent.agent.states import StateId
from trustworthy_agent.exceptions import TransitionPolicyError
from trustworthy_agent.transitions.base import RandomGenerator


def normalize_scores(
    allowed_transitions: Sequence[StateId],
    raw_scores: Mapping[StateId, float],
) -> dict[StateId, float]:
    """Normalize finite non-negative scores over exactly the allowed states.

    Purpose:
        Provide one audited probability normalization path for transition
        policies.
    Parameters:
        allowed_transitions: Targets permitted by the active profile.
        raw_scores: Raw transition scores keyed by state.
    Return value:
        Probability distribution over allowed states.
    Raised exceptions:
        TransitionPolicyError when scores are invalid or contain forbidden
        targets.
    Scientific assumptions:
        Probabilities are workflow-control probabilities, not physical process
        dynamics.
    Side effects:
        None.
    Reproducibility implications:
        Deterministic ordering follows `allowed_transitions`.
    """

    allowed = tuple(allowed_transitions)
    if not allowed:
        raise TransitionPolicyError("Transition policy requires at least one allowed target.")
    forbidden = set(raw_scores) - set(allowed)
    if forbidden:
        raise TransitionPolicyError(f"Scores include forbidden transition targets: {forbidden}")
    sanitized: dict[StateId, float] = {}
    for state in allowed:
        score = float(raw_scores.get(state, 0.0))
        if not math.isfinite(score) or score < 0.0:
            raise TransitionPolicyError(f"Invalid raw transition score for {state}: {score}")
        sanitized[state] = score
    total = sum(sanitized.values())
    if total <= 0.0:
        raise TransitionPolicyError("Transition scores have no positive probability mass.")
    return {state: sanitized[state] / total for state in allowed}


def select_state(
    probabilities: Mapping[StateId, float],
    mode: str,
    rng: RandomGenerator,
) -> StateId:
    """Select a state by deterministic argmax or injected seeded sampling.

    Purpose:
        Implement the two selection modes required by the specification without
        hidden global RNG state.
    Parameters:
        probabilities: Normalized probabilities.
        mode: `argmax` or `seeded_sampling`.
        rng: Injected RNG used only for seeded sampling.
    Return value:
        Selected state.
    Raised exceptions:
        TransitionPolicyError for unknown modes.
    Scientific assumptions:
        Selection randomness controls the workflow only.
    Side effects:
        `seeded_sampling` advances the injected RNG stream.
    Reproducibility implications:
        Identical injected RNG seed and call order reproduce identical choices.
    """

    ordered = tuple(probabilities)
    if mode == "argmax":
        return max(ordered, key=lambda state: (probabilities[state], -ordered.index(state)))
    if mode != "seeded_sampling":
        raise TransitionPolicyError(f"Unknown transition selection mode: {mode}")
    draw = rng.random()
    cumulative = 0.0
    for state in ordered:
        cumulative += probabilities[state]
        if draw <= cumulative:
            return state
    return ordered[-1]
