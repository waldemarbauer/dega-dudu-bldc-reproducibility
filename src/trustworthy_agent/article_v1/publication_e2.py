"""Publication-reproduction entry point for the 12 ArticleV1 DEGA runs.

The canonical implementation is used when the historical PyMC/NUTS posterior
artifacts are present. Otherwise a compact routing snapshot, reconstructed from
persisted ArticleV1 E2 audit evidence, is used only to replay the representative
workflow experiment. This fallback reproduces workflow routing; it is not a new
MCMC fit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.article_v1 import minimal_e2_execution as base
from trustworthy_agent.experiments.agent_evaluation import _build_bayesian_model, _load_mcmc_model
from trustworthy_agent.exceptions import TransitionPolicyError
from trustworthy_agent.transitions.base import RandomGenerator, TransitionDecision
from trustworthy_agent.transitions.bayesian import (
    BayesianMCMCTransitionPolicy,
    DeterministicPosteriorApproximationPolicy,
    HybridTransitionPolicy,
)
from trustworthy_agent.transitions.selection import normalize_scores
from trustworthy_agent.transitions.static_markov import StaticTransitionMatrixPolicy

SNAPSHOT = Path("reference_inputs/transition_policies/article_v1_routing_snapshot.json")


class FrozenRoutingSnapshotPolicy:
    """Deterministic policy backed by persisted article-routing probabilities."""

    def __init__(self, policy_name: str, policy_version: str, routing: Mapping[str, Any]) -> None:
        self._policy_name = policy_name
        self._policy_version = policy_version
        self._routing = dict(routing)

    @property
    def policy_name(self) -> str:
        return self._policy_name

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def decide(
        self,
        current_state: StateId,
        allowed_transitions: Sequence[StateId],
        context: AgentContext,
        rng: RandomGenerator,
    ) -> TransitionDecision:
        del context, rng
        allowed = tuple(allowed_transitions)
        if not allowed:
            raise TransitionPolicyError("Publication routing snapshot requires a non-empty candidate set")
        raw_record = self._routing.get(current_state.value)
        if raw_record is None:
            if len(allowed) == 1:
                raw_scores = {allowed[0]: 1.0}
            else:
                raise TransitionPolicyError(
                    f"No frozen publication routing record for {self.policy_name}/{current_state.value}"
                )
        else:
            raw_scores = {
                state: float(raw_record[state.value])
                for state in allowed
                if state.value in raw_record
            }
            if len(raw_scores) != len(allowed):
                missing = [state.value for state in allowed if state not in raw_scores]
                raise TransitionPolicyError(
                    f"Frozen routing snapshot misses allowed states {missing} for {current_state.value}"
                )
        probabilities = normalize_scores(allowed, raw_scores)
        selected = max(allowed, key=lambda state: (probabilities[state], -allowed.index(state)))
        return TransitionDecision(
            current_state=current_state,
            candidate_states=allowed,
            raw_scores=raw_scores,
            probabilities=probabilities,
            selected_state=selected,
            selection_mode="argmax",
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            rng_seed_or_stream_id=None,
            decision_reason="FROZEN_PUBLICATION_ROUTING_SNAPSHOT",
        )


def _snapshot_policies(root: Path) -> dict[str, Any]:
    static = StaticTransitionMatrixPolicy.from_config_file(
        root / "configs/transition_policies/static_transition_matrix_v1.yaml"
    )
    approx = DeterministicPosteriorApproximationPolicy(_build_bayesian_model())
    mcmc_model = _load_mcmc_model(root)
    if mcmc_model is not None:
        mcmc = BayesianMCMCTransitionPolicy(mcmc_model)
        hybrid = HybridTransitionPolicy(static, mcmc)
    else:
        payload = json.loads((root / SNAPSHOT).read_text(encoding="utf-8"))
        mcmc_record = payload["policies"]["BayesianMCMCTransitionPolicy"]
        hybrid_record = payload["policies"]["HybridTransitionPolicy"]
        mcmc = FrozenRoutingSnapshotPolicy(
            "BayesianMCMCTransitionPolicy",
            str(mcmc_record["policy_version"]),
            mcmc_record["routing"],
        )
        hybrid = FrozenRoutingSnapshotPolicy(
            "HybridTransitionPolicy",
            str(hybrid_record["policy_version"]),
            hybrid_record["routing"],
        )
    return {
        base.POLICY_IDS[0]: static,
        base.POLICY_IDS[1]: approx,
        base.POLICY_IDS[2]: mcmc,
        base.POLICY_IDS[3]: hybrid,
    }


def execute(root: Path) -> dict[str, Any]:
    """Run the canonical 12-run E2 orchestration with publication-safe policy resolution."""

    original = base._policy_instances
    base._policy_instances = _snapshot_policies
    try:
        result = base.execute(root)
    finally:
        base._policy_instances = original
    result["publication_routing_snapshot_used"] = not (
        root / "Output/Models/TransitionPolicies/BayesianMCMC/posterior_trace.nc"
    ).exists()
    return result
