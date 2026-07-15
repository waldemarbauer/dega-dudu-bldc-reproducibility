"""Bayesian and hybrid transition policies for controlled FSM proposals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.exceptions import TransitionPolicyError
from trustworthy_agent.transitions.base import RandomGenerator, TransitionDecision
from trustworthy_agent.transitions.selection import normalize_scores, select_state
from trustworthy_agent.transitions.static_markov import StaticTransitionMatrixPolicy

MCMC_COVARIATE_NAMES: tuple[str, ...] = (
    "data_quality",
    "missing_ratio",
    "confidence",
    "model_agreement",
    "representation_agreement",
    "explanation_score",
    "domain_consistency_score",
    "risk_score",
    "slope",
    "curvature",
    "distance_from_healthy",
    "ood_score",
)

FEATURE_NAMES: tuple[str, ...] = (
    *MCMC_COVARIATE_NAMES,
    "representation_agreement",
    "classifier_agreement",
    "distance_from_healthy",
    "spline_curvature",
    "ood_score",
    "explanation_score",
    "domain_consistency",
)


@dataclass(frozen=True)
class PosteriorDiagnostics:
    """Posterior quality evidence used by Bayesian transition policies.

    Purpose:
        Carry sampler diagnostics without coupling the transition interface to a
        concrete inference library.
    Parameters:
        r_hat: Largest recorded convergence diagnostic across coefficients.
        ess_bulk: Minimum effective sample size across coefficients.
        divergences: Number of divergent transitions reported by the sampler.
        posterior_entropy: Entropy of the current posterior transition
            distribution, normalized by the number of candidates when possible.
        posterior_confidence: Weight in `[0, 1]` used by hybrid composition.
        sampler: Human-readable sampler identity.
    Return value:
        Immutable posterior diagnostics.
    Raised exceptions:
        None.
    Scientific assumptions:
        These diagnostics summarize workflow-control uncertainty, not physical
        degradation uncertainty.
    Side effects:
        None.
    Reproducibility implications:
        Values are persisted in policy manifests and transition events.
    """

    r_hat: float
    ess_bulk: float
    divergences: int
    posterior_entropy: float
    posterior_confidence: float
    sampler: str

    @property
    def valid_for_hybrid(self) -> bool:
        """Return whether diagnostics allow posterior blending."""

        return (
            math.isfinite(self.r_hat)
            and self.r_hat <= 1.1
            and math.isfinite(self.ess_bulk)
            and self.ess_bulk > 0.0
            and self.divergences == 0
            and math.isfinite(self.posterior_confidence)
            and 0.0 <= self.posterior_confidence <= 1.0
        )


@dataclass(frozen=True)
class BayesianTransitionModel:
    """Multinomial-logistic posterior samples for transition probabilities.

    Purpose:
        Represent posterior samples of target-state logits conditioned on
        explicit, bounded context features.
    Parameters:
        coefficients: Sequence of posterior samples. Each sample maps target
            state to an intercept and feature weights.
        feature_names: Ordered feature names used by every sample.
        diagnostics: Global posterior diagnostics.
    Return value:
        Immutable Bayesian transition model.
    Raised exceptions:
        TransitionPolicyError for empty or malformed posterior evidence.
    Scientific assumptions:
        The model estimates workflow transition probabilities only. It does not
        estimate failure progression, RUL, or true degradation time.
    Side effects:
        None.
    Reproducibility implications:
        Samples, features, and diagnostics can be hashed and replayed.
    """

    coefficients: tuple[Mapping[StateId, Mapping[str, float]], ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    diagnostics: PosteriorDiagnostics = field(
        default_factory=lambda: PosteriorDiagnostics(
            r_hat=1.0,
            ess_bulk=1.0,
            divergences=0,
            posterior_entropy=0.0,
            posterior_confidence=1.0,
            sampler="configured_posterior",
        )
    )

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise TransitionPolicyError("Bayesian transition model requires posterior samples.")
        for sample in self.coefficients:
            if not sample:
                raise TransitionPolicyError("Posterior sample must include at least one state.")
            for state, weights in sample.items():
                if not isinstance(state, StateId):
                    raise TransitionPolicyError("Posterior coefficient keys must be StateId.")
                if "intercept" not in weights:
                    raise TransitionPolicyError("Posterior state weights require an intercept.")
                for name, value in weights.items():
                    if name != "intercept" and name not in self.feature_names:
                        raise TransitionPolicyError(f"Unknown posterior feature: {name}")
                    if not math.isfinite(float(value)):
                        raise TransitionPolicyError(f"Non-finite posterior coefficient: {name}")


class BayesianTransitionPolicy:
    """Propose FSM transitions from posterior multinomial-logistic evidence.

    Purpose:
        Implement a replaceable feature-conditioned transition policy without
        importing or bypassing SafetyGuard.
    Parameters:
        model: Fitted posterior transition model.
        selection_mode: `argmax` or `seeded_sampling`.
        policy_id: Stable policy name for audit/provenance.
        policy_version: Version of the policy implementation.
        numerical_tolerance: Tolerance used by `TransitionDecision`.
    Return value:
        Transition policy instance.
    Raised exceptions:
        TransitionPolicyError for invalid posterior probabilities or selection
        modes.
    Scientific assumptions:
        Feature-conditioned probabilities are workflow-control probabilities
        only and do not claim physical degradation dynamics.
    Side effects:
        Stores the most recent posterior evidence in-memory for immediate audit
        serialization by orchestration code.
    Reproducibility implications:
        Uses only the supplied posterior evidence and injected RNG stream.
    """

    def __init__(
        self,
        model: BayesianTransitionModel,
        *,
        selection_mode: str = "argmax",
        policy_id: str = "BayesianTransitionPolicy",
        policy_version: str = "1.0.0",
        numerical_tolerance: float = 1e-9,
    ) -> None:
        if selection_mode not in {"argmax", "seeded_sampling"}:
            raise TransitionPolicyError(f"Unknown transition selection mode: {selection_mode}")
        self._model = model
        self._selection_mode = selection_mode
        self._policy_name = policy_id
        self._policy_version = policy_version
        self._numerical_tolerance = numerical_tolerance
        self._last_evidence: dict[str, Any] = {}

    @property
    def policy_name(self) -> str:
        """Stable Bayesian policy name."""

        return self._policy_name

    @property
    def policy_version(self) -> str:
        """Stable Bayesian policy version."""

        return self._policy_version

    @property
    def last_evidence(self) -> Mapping[str, Any]:
        """Return posterior evidence for the most recent decision."""

        return dict(self._last_evidence)

    def decide(
        self,
        current_state: StateId,
        allowed_transitions: Sequence[StateId],
        context: AgentContext,
        rng: RandomGenerator,
    ) -> TransitionDecision:
        """Return a posterior mean transition proposal over allowed states."""

        allowed = tuple(allowed_transitions)
        if not allowed:
            raise TransitionPolicyError("Bayesian policy requires at least one allowed target.")
        feature_vector = extract_transition_features(context)
        sample_probabilities = [
            _sample_probabilities(sample, allowed, feature_vector)
            for sample in self._model.coefficients
        ]
        posterior_mean = {
            state: sum(sample[state] for sample in sample_probabilities) / len(sample_probabilities)
            for state in allowed
        }
        probabilities = normalize_scores(allowed, posterior_mean)
        posterior_variance = {
            state: _variance([sample[state] for sample in sample_probabilities])
            for state in allowed
        }
        hdi = {
            state: _interval([sample[state] for sample in sample_probabilities])
            for state in allowed
        }
        entropy = _normalized_entropy(probabilities)
        diagnostics = PosteriorDiagnostics(
            r_hat=self._model.diagnostics.r_hat,
            ess_bulk=self._model.diagnostics.ess_bulk,
            divergences=self._model.diagnostics.divergences,
            posterior_entropy=entropy,
            posterior_confidence=max(0.0, min(1.0, 1.0 - entropy)),
            sampler=self._model.diagnostics.sampler,
        )
        selected = select_state(probabilities, self._selection_mode, rng)
        self._last_evidence = {
            "features": dict(feature_vector),
            "posterior_mean": {state.value: probabilities[state] for state in allowed},
            "posterior_variance": {state.value: posterior_variance[state] for state in allowed},
            "posterior_hdi": {
                state.value: {"low": hdi[state][0], "high": hdi[state][1]} for state in allowed
            },
            "posterior_entropy": entropy,
            "posterior_confidence": diagnostics.posterior_confidence,
            "posterior_selected_state": selected.value,
            "diagnostics": {
                "r_hat": diagnostics.r_hat,
                "ess_bulk": diagnostics.ess_bulk,
                "divergences": diagnostics.divergences,
                "sampler": diagnostics.sampler,
            },
        }
        return TransitionDecision(
            current_state=current_state,
            candidate_states=allowed,
            raw_scores=posterior_mean,
            probabilities=probabilities,
            selected_state=selected,
            selection_mode=self._selection_mode,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            rng_seed_or_stream_id=rng.stream_id
            if self._selection_mode == "seeded_sampling"
            else None,
            decision_reason="BAYESIAN_POSTERIOR_SELECTION",
            numerical_tolerance=self._numerical_tolerance,
        )


class DeterministicPosteriorApproximationPolicy(BayesianTransitionPolicy):
    """Non-MCMC posterior approximation baseline."""

    def __init__(
        self,
        model: BayesianTransitionModel,
        *,
        selection_mode: str = "argmax",
    ) -> None:
        """Create a deterministic non-sampling posterior approximation policy."""

        super().__init__(
            model,
            selection_mode=selection_mode,
            policy_id="DeterministicPosteriorApproximationPolicy",
            policy_version="1.0.0",
        )


class BayesianMCMCTransitionPolicy(BayesianTransitionPolicy):
    """Transition policy backed by executed PyMC NUTS posterior samples."""

    def __init__(
        self,
        model: BayesianTransitionModel,
        *,
        selection_mode: str = "argmax",
    ) -> None:
        """Create an MCMC posterior transition policy.

        Purpose:
            Reject deterministic approximations so MCMC and non-MCMC policies
            cannot be confused in manifests or audit events.
        Parameters:
            model: Posterior samples with diagnostics from PyMC NUTS.
            selection_mode: `argmax` or `seeded_sampling`.
        Return value:
            Bayesian MCMC transition policy.
        Raised exceptions:
            TransitionPolicyError if the model is not identified as PyMC NUTS.
        Scientific assumptions:
            Posterior probabilities are workflow-control probabilities only.
        Side effects:
            None beyond the base class latest-evidence cache.
        Reproducibility implications:
            Requires explicit sampler identity and recorded diagnostics.
        """

        if not model.diagnostics.sampler.startswith("pymc_nuts"):
            raise TransitionPolicyError("BayesianMCMCTransitionPolicy requires PyMC NUTS evidence.")
        super().__init__(
            model,
            selection_mode=selection_mode,
            policy_id="BayesianMCMCTransitionPolicy",
            policy_version="1.0.0",
        )


class HybridTransitionPolicy:
    """Blend static and Bayesian transition proposals by composition.

    Purpose:
        Combine an existing static transition matrix with posterior transition
        evidence while keeping SafetyGuard external and authoritative.
    Parameters:
        static_policy: Existing static policy implementation.
        bayesian_policy: Bayesian policy implementation.
        selection_mode: `argmax` or `seeded_sampling`.
        minimum_posterior_confidence: Below this value, fall back to static
            probabilities.
    Return value:
        Transition policy instance.
    Raised exceptions:
        TransitionPolicyError for invalid probabilities or modes.
    Scientific assumptions:
        Blended probabilities remain workflow-control evidence only.
    Side effects:
        Stores the latest blend evidence for audit serialization.
    Reproducibility implications:
        Composition records static, posterior, and blend weights.
    """

    def __init__(
        self,
        static_policy: StaticTransitionMatrixPolicy,
        bayesian_policy: BayesianTransitionPolicy,
        *,
        selection_mode: str = "argmax",
        policy_id: str = "HybridTransitionPolicy",
        policy_version: str = "1.0.0",
        minimum_posterior_confidence: float = 0.20,
        high_entropy_threshold: float = 0.82,
        numerical_tolerance: float = 1e-9,
    ) -> None:
        if selection_mode not in {"argmax", "seeded_sampling"}:
            raise TransitionPolicyError(f"Unknown transition selection mode: {selection_mode}")
        self.static_policy = static_policy
        self.bayesian_policy = bayesian_policy
        self._selection_mode = selection_mode
        self._policy_name = policy_id
        self._policy_version = policy_version
        self.minimum_posterior_confidence = minimum_posterior_confidence
        self.high_entropy_threshold = high_entropy_threshold
        self._numerical_tolerance = numerical_tolerance
        self._last_evidence: dict[str, Any] = {}

    @property
    def policy_name(self) -> str:
        """Stable hybrid policy name."""

        return self._policy_name

    @property
    def policy_version(self) -> str:
        """Stable hybrid policy version."""

        return self._policy_version

    @property
    def last_evidence(self) -> Mapping[str, Any]:
        """Return blend evidence for the most recent decision."""

        return dict(self._last_evidence)

    def decide(
        self,
        current_state: StateId,
        allowed_transitions: Sequence[StateId],
        context: AgentContext,
        rng: RandomGenerator,
    ) -> TransitionDecision:
        """Return a static/posterior blended transition proposal."""

        allowed = tuple(allowed_transitions)
        static_decision = self.static_policy.decide(current_state, allowed, context, rng)
        bayesian_decision = self.bayesian_policy.decide(current_state, allowed, context, rng)
        bayesian_evidence = self.bayesian_policy.last_evidence
        diagnostics = bayesian_evidence.get("diagnostics", {})
        confidence = float(bayesian_evidence.get("posterior_confidence", 0.0))
        entropy = float(bayesian_evidence.get("posterior_entropy", 0.0))
        valid = (
            math.isfinite(confidence)
            and confidence >= self.minimum_posterior_confidence
            and float(diagnostics.get("r_hat", math.inf)) <= 1.1
            and float(diagnostics.get("ess_bulk", 0.0)) > 0.0
            and int(diagnostics.get("divergences", 1)) == 0
        )
        blend_weight = confidence if valid else 0.0
        posterior_overlap = _terminal_hdi_overlap(bayesian_evidence)
        terminal_states = {StateId.RECOMMENDATION, StateId.ESCALATION, StateId.NO_DECISION}
        static_mcmc_disagreement = (
            static_decision.selected_state != bayesian_decision.selected_state
            and static_decision.selected_state in terminal_states
            and bayesian_decision.selected_state in terminal_states
        )
        raw_scores = {
            state: (1.0 - blend_weight) * static_decision.probabilities[state]
            + blend_weight * bayesian_decision.probabilities[state]
            for state in allowed
        }
        fallback_reason = ""
        if not valid:
            fallback_reason = "INVALID_MCMC_DIAGNOSTICS_STATIC_FALLBACK"
        elif entropy >= self.high_entropy_threshold and StateId.ESCALATION in allowed:
            raw_scores = {state: 0.05 for state in allowed}
            raw_scores[StateId.ESCALATION] = 0.90
            fallback_reason = "HIGH_POSTERIOR_ENTROPY_ESCALATION"
        elif posterior_overlap and StateId.ESCALATION in allowed:
            raw_scores = {state: 0.05 for state in allowed}
            raw_scores[StateId.ESCALATION] = 0.90
            fallback_reason = "POSTERIOR_TERMINAL_HDI_OVERLAP_ESCALATION"
        probabilities = normalize_scores(allowed, raw_scores)
        selected = select_state(probabilities, self._selection_mode, rng)
        self._last_evidence = {
            "static_policy": static_decision.policy_name,
            "bayesian_policy": bayesian_decision.policy_name,
            "static_probabilities": {
                state.value: static_decision.probabilities[state] for state in allowed
            },
            "posterior_probabilities": {
                state.value: bayesian_decision.probabilities[state] for state in allowed
            },
            "hybrid_probabilities": {state.value: probabilities[state] for state in allowed},
            "posterior_confidence": confidence,
            "posterior_entropy": entropy,
            "blend_weight": blend_weight,
            "fallback_to_static": not valid,
            "fallback_reason": fallback_reason,
            "posterior_terminal_hdi_overlap": posterior_overlap,
            "static_mcmc_terminal_disagreement": static_mcmc_disagreement,
            "hybrid_selected_state": selected.value,
            "posterior_selected_state": bayesian_decision.selected_state.value,
            "static_selected_state": static_decision.selected_state.value,
        }
        return TransitionDecision(
            current_state=current_state,
            candidate_states=allowed,
            raw_scores=raw_scores,
            probabilities=probabilities,
            selected_state=selected,
            selection_mode=self._selection_mode,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            rng_seed_or_stream_id=rng.stream_id
            if self._selection_mode == "seeded_sampling"
            else None,
            decision_reason="HYBRID_STATIC_BAYESIAN_BLEND"
            if valid and not fallback_reason
            else fallback_reason or "HYBRID_FALLBACK_TO_STATIC",
            numerical_tolerance=self._numerical_tolerance,
        )


def extract_transition_features(context: AgentContext) -> dict[str, float]:
    """Extract bounded transition features from explicit context evidence."""

    facts = context.derived_facts
    return {
        "data_quality": _bounded(_fact(context.data_quality, facts, "data_quality", default=1.0)),
        "missing_ratio": _bounded(_fact(context.missing_ratio, facts, "missing_ratio")),
        "confidence": _bounded(_fact(context.confidence, facts, "confidence")),
        "model_agreement": _bounded(_fact(context.model_agreement, facts, "model_agreement")),
        "risk_score": _bounded(_fact(context.risk_score, facts, "risk_score")),
        "representation_agreement": _bounded(
            1.0 - _fact(None, facts, "representation_disagreement", default=0.0)
        ),
        "classifier_agreement": _bounded(
            1.0 - _fact(None, facts, "classifier_disagreement", default=0.0)
        ),
        "distance_from_healthy": _bounded(_fact(None, facts, "distance_from_healthy")),
        "slope": _bounded(_fact(None, facts, "slope")),
        "curvature": _bounded(_fact(None, facts, "curvature")),
        "spline_curvature": _bounded(_fact(None, facts, "spline_curvature")),
        "ood_score": _bounded(_fact(context.ood_score, facts, "ood_score")),
        "explanation_score": _bounded(_fact(context.explanation_score, facts, "explanation_score")),
        "domain_consistency_score": _bounded(
            _fact(context.domain_consistency_score, facts, "domain_consistency_score")
        ),
        "domain_consistency": _bounded(
            _fact(context.domain_consistency_score, facts, "domain_consistency_score")
        ),
    }


def _sample_probabilities(
    sample: Mapping[StateId, Mapping[str, float]],
    allowed: Sequence[StateId],
    feature_vector: Mapping[str, float],
) -> dict[StateId, float]:
    logits: dict[StateId, float] = {}
    for state in allowed:
        weights = sample.get(state)
        if weights is None:
            logits[state] = -20.0
            continue
        logits[state] = float(weights["intercept"]) + sum(
            float(weights.get(feature, 0.0)) * value for feature, value in feature_vector.items()
        )
    max_logit = max(logits.values())
    raw = {state: math.exp(logit - max_logit) for state, logit in logits.items()}
    return normalize_scores(allowed, raw)


def _fact(
    direct_value: Any,
    facts: Mapping[str, Any],
    key: str,
    *,
    default: float = 0.0,
) -> float:
    value = direct_value if direct_value is not None else facts.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _interval(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return (0.0, 0.0)
    low_index = int(math.floor(0.025 * (len(ordered) - 1)))
    high_index = int(math.ceil(0.975 * (len(ordered) - 1)))
    return (ordered[low_index], ordered[high_index])


def _normalized_entropy(probabilities: Mapping[StateId, float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    return entropy / math.log(len(probabilities))


def _terminal_hdi_overlap(evidence: Mapping[str, Any]) -> bool:
    hdi = evidence.get("posterior_hdi", {})
    if not isinstance(hdi, Mapping):
        return False
    recommendation = _hdi_tuple(hdi.get(StateId.RECOMMENDATION.value))
    escalation = _hdi_tuple(hdi.get(StateId.ESCALATION.value))
    no_decision = _hdi_tuple(hdi.get(StateId.NO_DECISION.value))
    pairs = (
        (recommendation, escalation),
        (recommendation, no_decision),
        (escalation, no_decision),
    )
    return any(_intervals_overlap(left, right) for left, right in pairs)


def _hdi_tuple(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        low = float(value["low"])
        high = float(value["high"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    return (low, high)


def _intervals_overlap(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
) -> bool:
    if left is None or right is None:
        return False
    return max(left[0], right[0]) <= min(left[1], right[1])
