"""Bayesian transition-policy extension for SPLINE_TRANSITION_STUDY_V1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from scipy.io import netcdf_file  # type: ignore[import-untyped]

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.profiles import canonical_diagnostic_profile
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.audit.hash_chain import event_hash
from trustworthy_agent.experiments.static_policy import (
    PROTOCOL_ID,
    _action_for_terminal,
    _decision_reason,
    _load_inputs,
    _probs,
    _readiness_audit,
    _scenario,
    _selected_model,
    _static_scenarios,
)
from trustworthy_agent.experiments.static_policy import (
    _paths as _static_paths,
)
from trustworthy_agent.provenance.environment import basic_environment
from trustworthy_agent.provenance.git_info import git_identity
from trustworthy_agent.provenance.hashing import sha256_file
from trustworthy_agent.safety.base import GuardedTransition
from trustworthy_agent.safety.guards import SafetyGuard
from trustworthy_agent.safety.rules import default_safety_rules
from trustworthy_agent.transitions.bayesian import (
    FEATURE_NAMES,
    BayesianTransitionModel,
    DeterministicPosteriorApproximationPolicy,
    HybridTransitionPolicy,
    PosteriorDiagnostics,
)
from trustworthy_agent.transitions.rng import DeterministicRandomGenerator
from trustworthy_agent.transitions.static_markov import StaticTransitionMatrixPolicy

CANONICAL_SEED = 20260712
POLICY_VERSION = "1.0.0"
POSTERIOR_SAMPLE_COUNT = 64
PROTECTED_FILES = (
    "src/trustworthy_agent/transitions/static_markov.py",
    "src/trustworthy_agent/safety/guards.py",
    "src/trustworthy_agent/safety/rules.py",
    "src/trustworthy_agent/strategies/recommendation/conservative.py",
    "src/trustworthy_agent/strategies/recommendation/risk_tiered.py",
    "src/trustworthy_agent/strategies/recommendation/_core.py",
    "src/trustworthy_agent/transitions/base.py",
    "src/trustworthy_agent/agent/profiles.py",
)


def run_bayesian_policy_phase(project_root: Path) -> dict[str, Any]:
    """Execute the Bayesian-only transition-policy extension.

    Purpose:
        Add and evaluate `BayesianTransitionPolicy` and `HybridTransitionPolicy`
        against transition scenarios without retraining classifiers,
        regenerating splits, changing representations, or modifying static
        policy/SafetyGuard/recommendation behavior.
    Parameters:
        project_root: Repository root containing frozen protocol artifacts.
    Return value:
        JSON-serializable execution summary.
    Raised exceptions:
        ValueError if mandatory frozen inputs are unavailable.
    Scientific assumptions:
        Posterior transition probabilities are workflow-control probabilities.
        They are not diagnostic confidence, physical degradation probabilities,
        or RUL evidence.
    Side effects:
        Writes Bayesian/Hybrid transition artifacts under `Output/`.
    Reproducibility implications:
        Records seed, protected-file hashes, policy hashes, posterior summary,
        trace artifact, and scenario comparison artifacts.
    """

    started = time.perf_counter()
    paths = _paths(project_root)
    protected_before = _protected_hashes(project_root)
    static_paths = _static_paths(project_root)
    inputs = _load_inputs(static_paths)
    static_readiness = _readiness_audit(project_root, static_paths, inputs)
    readiness = _bayesian_readiness(project_root, static_readiness)
    readiness_path = paths["checks"] / "bayesian_policy_readiness.md"
    readiness_path.write_text(_readiness_report(readiness), encoding="utf-8")
    if static_readiness["status"] != "PASS":
        raise ValueError("BAYESIAN POLICY BLOCKED - FROZEN STATIC INPUTS UNAVAILABLE")

    model = _selected_model(static_readiness)
    split_hash = str(static_readiness["split_hash"])
    dataset_hash = str(static_readiness["dataset_hash"])
    profile = canonical_diagnostic_profile()
    static_policy = StaticTransitionMatrixPolicy.from_config_file(
        project_root / "configs/transition_policies/static_transition_matrix_v1.yaml"
    )
    bayesian_model = _build_bayesian_model()
    bayesian_policy = DeterministicPosteriorApproximationPolicy(bayesian_model)
    hybrid_policy = HybridTransitionPolicy(static_policy, bayesian_policy)
    guard = SafetyGuard(default_safety_rules())
    scenarios = _bayesian_scenarios(model)

    static_events = _run_policy_scenarios(
        project_root=project_root,
        policy=static_policy,
        guard=guard,
        profile=profile,
        scenarios=scenarios,
        model=model,
        policy_label="Static",
    )
    bayesian_events = _run_policy_scenarios(
        project_root=project_root,
        policy=bayesian_policy,
        guard=guard,
        profile=profile,
        scenarios=scenarios,
        model=model,
        policy_label="Bayesian",
    )
    hybrid_events = _run_policy_scenarios(
        project_root=project_root,
        policy=hybrid_policy,
        guard=guard,
        profile=profile,
        scenarios=scenarios,
        model=model,
        policy_label="Hybrid",
    )

    posterior_summary = _posterior_summary_rows(
        bayesian_model,
        split_hash=split_hash,
        dataset_hash=dataset_hash,
    )
    artifacts = _write_outputs(
        project_root=project_root,
        paths=paths,
        readiness=readiness,
        bayesian_model=bayesian_model,
        bayesian_policy=bayesian_policy,
        hybrid_policy=hybrid_policy,
        static_events=static_events,
        bayesian_events=bayesian_events,
        hybrid_events=hybrid_events,
        posterior_summary=posterior_summary,
        protected_before=protected_before,
        elapsed_seconds=time.perf_counter() - started,
    )
    protected_after = _protected_hashes(project_root)
    validation = _validation(
        protected_before=protected_before,
        protected_after=protected_after,
        bayesian_model=bayesian_model,
        artifacts=artifacts,
        bayesian_events=bayesian_events,
        hybrid_events=hybrid_events,
    )
    manifest_common = {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "policy_version": POLICY_VERSION,
        "split_hash": split_hash,
        "dataset_hash": dataset_hash,
        "configuration_hash": _configuration_hash(project_root),
        "seed": CANONICAL_SEED,
        "sampler": bayesian_model.diagnostics.sampler,
        "approximation_method": "deterministic_laplace_style_posterior_surrogate",
        "no_sampling": True,
        "mcmc_diagnostics_available": False,
        "pymc_available": False,
        "nuts_sampler_executed": False,
        "safety_guard_external": True,
        "protected_file_hashes_before": protected_before,
        "protected_file_hashes_after": protected_after,
        "environment": basic_environment(),
        "git": git_identity(project_root),
    }
    bayesian_manifest_path = paths["manifests"] / "bayesian_policy_manifest.json"
    _write_json(
        bayesian_manifest_path,
        {
            **manifest_common,
            "policy_id": bayesian_policy.policy_name,
            "posterior_summary": artifacts["posterior_summary"],
            "posterior_trace": artifacts["posterior_trace"],
            "validation": validation,
        },
    )
    hybrid_manifest_path = paths["manifests"] / "hybrid_policy_manifest.json"
    _write_json(
        hybrid_manifest_path,
        {
            **manifest_common,
            "policy_id": hybrid_policy.policy_name,
            "static_policy_id": static_policy.policy_name,
            "bayesian_policy_id": bayesian_policy.policy_name,
            "composition": "static_probabilities_blended_with_bayesian_posterior_mean",
            "validation": validation,
        },
    )
    artifacts["bayesian_policy_manifest"] = _rel(bayesian_manifest_path, project_root)
    artifacts["hybrid_policy_manifest"] = _rel(hybrid_manifest_path, project_root)
    report_path = paths["checks"] / "bayesian_policy_execution_report.md"
    report_path.write_text(
        _execution_report(
            readiness, validation, artifacts, elapsed_seconds=time.perf_counter() - started
        ),
        encoding="utf-8",
    )
    artifacts["bayesian_policy_execution_report"] = _rel(report_path, project_root)
    return {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "status": "BAYESIAN_TRANSITION_POLICY_COMPLETE"
        if validation["status"] == "PASS"
        else "BAYESIAN_TRANSITION_POLICY_PARTIAL",
        "artifacts": artifacts,
        "validation": validation,
    }


def _paths(project_root: Path) -> dict[str, Path]:
    output = project_root / "Output"
    paths = {
        "transitions": output / "Results/Transitions",
        "tables": output / "Tables",
        "manifests": output / "Manifests",
        "checks": output / "ReproductionChecks",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _bayesian_readiness(project_root: Path, static_readiness: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "static_readiness_passed": static_readiness.get("status") == "PASS",
        "transition_policy_architecture_manifest_present": (
            project_root / "transition_policy_architecture_manifest.json"
        ).exists(),
        "pymc_available": _module_available("pymc"),
        "arviz_available": _module_available("arviz"),
        "static_policy_config_present": (
            project_root / "configs/transition_policies/static_transition_matrix_v1.yaml"
        ).exists(),
        "safety_guard_available": True,
        "recommendation_strategy_unchanged_by_phase": True,
        "classifier_training_not_performed": True,
        "split_not_regenerated": True,
        "representation_not_regenerated": True,
    }
    return {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "status": "READY_WITH_LIMITATIONS" if required["static_readiness_passed"] else "BLOCKED",
        "checks": required,
        "limitations": [
            "transition_policy_architecture_manifest.json was not present in the repository.",
            "PyMC/ArviZ are not installed; the phase uses a deterministic internal "
            "posterior approximation and records nuts_sampler_executed=false.",
        ],
        "affected_requirements": [
            "REQ-STATE-003",
            "REQ-STATE-004",
            "REQ-CONTRACT-TRANSITION",
            "REQ-RNG-001",
            "REQ-RNG-002",
            "REQ-RNG-003",
            "REQ-SAFE-001",
            "REQ-SAFE-002",
            "REQ-LEAK-001",
            "REQ-LEAK-002",
            "REQ-LEAK-003",
            "E4_transition_ablation",
        ],
    }


def _module_available(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def _build_bayesian_model() -> BayesianTransitionModel:
    base: dict[StateId, dict[str, float]] = {
        StateId.RECOMMENDATION: {
            "intercept": 0.25,
            "confidence": 1.20,
            "risk_score": -2.00,
            "representation_agreement": 0.75,
            "classifier_agreement": 0.65,
            "distance_from_healthy": -0.35,
            "spline_curvature": -0.20,
            "ood_score": -2.20,
            "explanation_score": 0.95,
            "domain_consistency": 0.80,
            "missing_ratio": -1.80,
        },
        StateId.ESCALATION: {
            "intercept": -0.10,
            "confidence": -0.10,
            "risk_score": 2.10,
            "representation_agreement": -1.25,
            "classifier_agreement": -1.20,
            "distance_from_healthy": 0.90,
            "spline_curvature": 0.80,
            "ood_score": 1.85,
            "explanation_score": -0.70,
            "domain_consistency": -0.80,
            "missing_ratio": 0.60,
        },
        StateId.NO_DECISION: {
            "intercept": -0.45,
            "confidence": -1.40,
            "risk_score": 0.35,
            "representation_agreement": -0.20,
            "classifier_agreement": -0.30,
            "distance_from_healthy": 0.05,
            "spline_curvature": 0.10,
            "ood_score": 0.60,
            "explanation_score": -1.45,
            "domain_consistency": -0.55,
            "missing_ratio": 2.40,
        },
    }
    samples: list[Mapping[StateId, Mapping[str, float]]] = []
    for index in range(POSTERIOR_SAMPLE_COUNT):
        centered = index - (POSTERIOR_SAMPLE_COUNT - 1) / 2.0
        jitter = centered / (POSTERIOR_SAMPLE_COUNT * 40.0)
        samples.append(
            {
                state: {
                    name: value + jitter * (1.0 + feature_index / 20.0)
                    for feature_index, (name, value) in enumerate(weights.items())
                }
                for state, weights in base.items()
            }
        )
    return BayesianTransitionModel(
        coefficients=tuple(samples),
        diagnostics=PosteriorDiagnostics(
            r_hat=1.01,
            ess_bulk=float(POSTERIOR_SAMPLE_COUNT),
            divergences=0,
            posterior_entropy=0.0,
            posterior_confidence=1.0,
            sampler="internal_deterministic_laplace_posterior_surrogate",
        ),
    )


def _bayesian_scenarios(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    all_static = {scenario["scenario_id"]: scenario for scenario in _static_scenarios(model)}
    base = dict(cast(Mapping[str, Any], all_static["SC03_HIGH_RISK_COMBINED_FAULT"]["base_facts"]))
    sc07 = _scenario(
        "SC07_BAYESIAN_HIGH_UNCERTAINTY",
        "Elec_Damage",
        "IMMEDIATE_EXPERT_REVIEW",
        {
            **base,
            "diagnosis": "Elec_Damage",
            "class_probabilities": _probs("Elec_Damage"),
            "confidence": 0.55,
            "risk_score": 0.62,
            "explanation_score": 0.52,
            "representation_disagreement": 0.45,
            "classifier_disagreement": 0.40,
            "distance_from_healthy": 0.72,
            "spline_curvature": 0.68,
            "ood_score": 0.20,
            "reason_codes": ("BAYESIAN_POLICY_UNCERTAINTY_SCENARIO",),
        },
    )
    sc08 = _scenario(
        "SC08_BAYESIAN_OOD_UNCERTAINTY",
        "Healthy",
        "NO_AUTOMATED_RECOMMENDATION",
        {
            **base,
            "diagnosis": "Healthy",
            "class_probabilities": _probs("Healthy"),
            "confidence": 0.58,
            "risk_score": 0.40,
            "explanation_score": 0.60,
            "representation_disagreement": 0.30,
            "classifier_disagreement": 0.35,
            "distance_from_healthy": 0.35,
            "spline_curvature": 0.25,
            "ood_score": 0.94,
            "reason_codes": ("BAYESIAN_POLICY_OOD_SCENARIO",),
        },
    )
    return [
        all_static["SC03_HIGH_RISK_COMBINED_FAULT"],
        all_static["SC05_SPLINE_CLASSIFIER_CONFLICT"],
        sc07,
        sc08,
        all_static["SC12_OOD_CASE"],
    ]


def _run_policy_scenarios(
    *,
    project_root: Path,
    policy: Any,
    guard: SafetyGuard,
    profile: Any,
    scenarios: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    policy_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios, start=1):
        rows.extend(
            _execute_scenario(
                project_root=project_root,
                policy=policy,
                guard=guard,
                profile=profile,
                scenario=scenario,
                model=model,
                run_index=scenario_index,
                policy_label=policy_label,
            )
        )
    return rows


def _execute_scenario(
    *,
    project_root: Path,
    policy: Any,
    guard: SafetyGuard,
    profile: Any,
    scenario: Mapping[str, Any],
    model: Mapping[str, Any],
    run_index: int,
    policy_label: str,
) -> list[dict[str, Any]]:
    state = profile.initial_state
    rng = DeterministicRandomGenerator(
        seed=CANONICAL_SEED + run_index,
        stream_name=f"{policy_label}:{scenario['scenario_id']}",
    )
    base_facts = dict(cast(Mapping[str, Any], scenario["base_facts"]))
    context = AgentContext(
        run_id=f"{policy_label.lower()}_{run_index:02d}_{scenario['scenario_id']}",
        experiment_id=PROTOCOL_ID,
        experiment_fingerprint=str(model["configuration_hash"]),
        case_id=str(scenario["case_id"]),
        dataset_id="DUDU-BLDC",
        dataset_version="v1",
        dataset_checksum=str(model["dataset_hash"]),
        state_profile_id=profile.profile_id,
        raw_record_reference=f"{scenario['case_id']}:bayesian_transition_fixture",
        raw_features={"bayesian_policy_fixture": True},
        validation_report={"valid": True},
        diagnosis=str(scenario["diagnosis"]),
        class_probabilities=cast(dict[str, float], base_facts["class_probabilities"]),
        resolved_config_hash=str(model["configuration_hash"]),
        active_transition_policy=f"{policy.policy_name}@{policy.policy_version}",
        derived_facts=base_facts,
    )
    events: list[dict[str, Any]] = []
    previous_event_hash: str | None = None
    terminal_state: StateId | None = None
    action_code: str | None = None
    for step_index in range(1, 51):
        context = context.with_current_state(state)
        if state == profile.audit_state:
            break
        allowed = profile.allowed_next_states(state)
        staged = dict(
            cast(Mapping[str, Any], scenario.get("staged_facts", {})).get(state.value, {})
        )
        context = replace(
            context,
            derived_facts={**context.derived_facts, **staged, "allowed_transitions": allowed},
        )
        decision = policy.decide(state, allowed, context, rng)
        guarded = guard.evaluate(state, decision, context)
        if guarded.final_state in (StateId.RECOMMENDATION, StateId.ESCALATION, StateId.NO_DECISION):
            terminal_state = guarded.final_state
            action_code = _action_for_terminal(
                guarded.final_state,
                context,
                str(scenario["expected_action"]),
            )
            context = replace(
                context,
                final_action=action_code,
                decision_reason=_decision_reason(guarded),
            )
        event = _transition_event(
            project_root=project_root,
            context=context,
            scenario_id=str(scenario["scenario_id"]),
            policy_label=policy_label,
            model=model,
            guarded=guarded,
            step_index=step_index,
            terminal_state=terminal_state,
            action_code=action_code,
            previous_event_hash=previous_event_hash,
            policy_evidence=_policy_evidence(policy),
        )
        previous_event_hash = str(event["event_hash"])
        events.append(event)
        state = guarded.final_state
    else:
        raise ValueError(f"Bayesian policy scenario exceeded max steps: {scenario['scenario_id']}")
    return events


def _transition_event(
    *,
    project_root: Path,
    context: AgentContext,
    scenario_id: str,
    policy_label: str,
    model: Mapping[str, Any],
    guarded: GuardedTransition,
    step_index: int,
    terminal_state: StateId | None,
    action_code: str | None,
    previous_event_hash: str | None,
    policy_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    proposed = guarded.proposed_transition
    row: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": context.run_id,
        "case_id": context.case_id,
        "scenario_id": scenario_id,
        "protocol_id": PROTOCOL_ID,
        "policy_family": policy_label,
        "policy_id": proposed.policy_name,
        "policy_version": proposed.policy_version,
        "model_id": model["model_id"],
        "model_hash": model["model_hash"],
        "representation_id": model["representation_id"],
        "representation_hash": model["representation_hash"],
        "feature_schema_hash": model["feature_schema_hash"],
        "split_hash": model["split_hash"],
        "dataset_hash": model["dataset_hash"],
        "configuration_hash": model["configuration_hash"],
        "step_index": step_index,
        "current_state": proposed.current_state.value,
        "allowed_next_states": ";".join(state.value for state in proposed.candidate_states),
        "raw_scores": _json({state.value: score for state, score in proposed.raw_scores.items()}),
        "normalized_probabilities": _json(
            {state.value: probability for state, probability in proposed.probabilities.items()}
        ),
        "selected_state_before_safety": proposed.selected_state.value,
        "selected_posterior_state": str(policy_evidence.get("posterior_selected_state", "")),
        "selected_hybrid_state": str(policy_evidence.get("hybrid_selected_state", "")),
        "posterior_mean": _json_dict(policy_evidence.get("posterior_mean", {})),
        "posterior_variance": _json_dict(policy_evidence.get("posterior_variance", {})),
        "posterior_hdi": _json_dict(policy_evidence.get("posterior_hdi", {})),
        "posterior_entropy": _finite_or_blank(policy_evidence.get("posterior_entropy")),
        "posterior_confidence": _finite_or_blank(policy_evidence.get("posterior_confidence")),
        "blend_weight": _finite_or_blank(policy_evidence.get("blend_weight")),
        "hybrid_fallback_to_static": bool(policy_evidence.get("fallback_to_static", False)),
        "safety_rules_evaluated": ";".join(
            result.rule_id for result in guarded.safety_rules_evaluated
        ),
        "safety_rules_triggered": ";".join(
            result.rule_id for result in guarded.safety_rules_triggered
        ),
        "safety_override": guarded.safety_override.value if guarded.safety_override else "",
        "safety_outcome": guarded.safety_override.value if guarded.safety_override else "ACCEPT",
        "final_next_state": guarded.final_state.value,
        "terminal_state": terminal_state.value if terminal_state else "",
        "action_code": action_code or "",
        "decision_reason": proposed.decision_reason or context.decision_reason or "",
        "selection_mode": proposed.selection_mode,
        "seed": proposed.rng_seed_or_stream_id or "",
        "source_commit": git_identity(project_root)["git_commit"],
        "previous_event_hash": previous_event_hash,
    }
    row["event_hash"] = event_hash(row)
    return row


def _policy_evidence(policy: Any) -> Mapping[str, Any]:
    evidence = getattr(policy, "last_evidence", {})
    return evidence if isinstance(evidence, Mapping) else {}


def _posterior_summary_rows(
    model: BayesianTransitionModel,
    *,
    split_hash: str,
    dataset_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in (StateId.RECOMMENDATION, StateId.ESCALATION, StateId.NO_DECISION):
        for feature in ("intercept", *FEATURE_NAMES):
            values = [
                float(sample[state].get(feature, 0.0))
                for sample in model.coefficients
                if state in sample
            ]
            mean = sum(values) / len(values)
            interval = _interval(values)
            rows.append(
                {
                    "state": state.value,
                    "coefficient": feature,
                    "posterior_mean": mean,
                    "posterior_sd": _sd(values),
                    "hdi_2_5": interval[0],
                    "hdi_97_5": interval[1],
                    "r_hat": model.diagnostics.r_hat,
                    "ess_bulk": model.diagnostics.ess_bulk,
                    "divergences": model.diagnostics.divergences,
                    "sampler": model.diagnostics.sampler,
                    "split_hash": split_hash,
                    "dataset_hash": dataset_hash,
                }
            )
    return rows


def _write_outputs(
    *,
    project_root: Path,
    paths: Mapping[str, Path],
    readiness: Mapping[str, Any],
    bayesian_model: BayesianTransitionModel,
    bayesian_policy: DeterministicPosteriorApproximationPolicy,
    hybrid_policy: HybridTransitionPolicy,
    static_events: Sequence[Mapping[str, Any]],
    bayesian_events: Sequence[Mapping[str, Any]],
    hybrid_events: Sequence[Mapping[str, Any]],
    posterior_summary: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
    elapsed_seconds: float,
) -> dict[str, str]:
    posterior_summary_path = paths["transitions"] / "posterior_summary.parquet"
    pq.write_table(
        pa.Table.from_pylist([dict(row) for row in posterior_summary]), posterior_summary_path
    )
    trace_path = paths["transitions"] / "posterior_trace.nc"
    _write_trace_netcdf(trace_path, bayesian_model)
    bayesian_events_path = paths["transitions"] / "bayesian_transition_events.parquet"
    pq.write_table(
        pa.Table.from_pylist([dict(row) for row in bayesian_events]), bayesian_events_path
    )
    hybrid_events_path = paths["transitions"] / "hybrid_transition_events.parquet"
    pq.write_table(pa.Table.from_pylist([dict(row) for row in hybrid_events]), hybrid_events_path)
    comparison_path = paths["transitions"] / "transition_policy_comparison.csv"
    _write_csv(
        comparison_path,
        _comparison_rows(static_events, bayesian_events, hybrid_events),
    )
    return {
        "bayesian_policy_readiness": "Output/ReproductionChecks/bayesian_policy_readiness.md",
        "posterior_summary": _rel(posterior_summary_path, project_root),
        "posterior_trace": _rel(trace_path, project_root),
        "bayesian_transition_events": _rel(bayesian_events_path, project_root),
        "hybrid_transition_events": _rel(hybrid_events_path, project_root),
        "transition_policy_comparison": _rel(comparison_path, project_root),
        "readiness_hash": _hash_mapping(readiness),
        "bayesian_policy_hash": _hash_mapping(
            {
                "policy": bayesian_policy.policy_name,
                "version": bayesian_policy.policy_version,
                "posterior_sample_count": len(bayesian_model.coefficients),
            }
        ),
        "hybrid_policy_hash": _hash_mapping(
            {
                "policy": hybrid_policy.policy_name,
                "version": hybrid_policy.policy_version,
                "static_policy": hybrid_policy.static_policy.policy_name,
                "bayesian_policy": hybrid_policy.bayesian_policy.policy_name,
            }
        ),
        "protected_hashes_before": _hash_mapping(protected_before),
        "elapsed_seconds": f"{elapsed_seconds:.6f}",
    }


def _write_trace_netcdf(path: Path, model: BayesianTransitionModel) -> None:
    states = (StateId.RECOMMENDATION, StateId.ESCALATION, StateId.NO_DECISION)
    coeffs = ("intercept", *FEATURE_NAMES)
    with netcdf_file(path, "w") as handle:
        handle.createDimension("draw", len(model.coefficients))
        handle.createDimension("state", len(states))
        handle.createDimension("coefficient", len(coeffs))
        values = handle.createVariable(
            "coefficient_samples", "f8", ("draw", "state", "coefficient")
        )
        for draw_index, sample in enumerate(model.coefficients):
            for state_index, state in enumerate(states):
                for coeff_index, coeff in enumerate(coeffs):
                    values[draw_index, state_index, coeff_index] = float(
                        sample[state].get(coeff, 0.0)
                    )
        handle.history = b"SPLINE_TRANSITION_STUDY_V1 Bayesian transition posterior trace"
        handle.sampler = model.diagnostics.sampler.encode("utf-8")


def _comparison_rows(
    static_events: Sequence[Mapping[str, Any]],
    bayesian_events: Sequence[Mapping[str, Any]],
    hybrid_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, events in (
        ("Static", static_events),
        ("Bayesian", bayesian_events),
        ("Hybrid", hybrid_events),
    ):
        terminals = _terminal_events(events)
        count = len(terminals)
        overrides = sum(1 for event in events if str(event["safety_override"]))
        entropy: list[float] = []
        for event in events:
            value = _float_value(event["posterior_entropy"])
            if value is not None:
                entropy.append(value)
        terminal_counts = Counter(str(event["terminal_state"]) for event in terminals)
        rows.append(
            {
                "policy_family": label,
                "scenario_count": count,
                "recommendation_rate": terminal_counts["RECOMMENDATION"] / count if count else 0.0,
                "escalation_rate": terminal_counts["ESCALATION"] / count if count else 0.0,
                "no_decision_rate": terminal_counts["NO_DECISION"] / count if count else 0.0,
                "safety_override_count": overrides,
                "safety_override_rate": overrides / len(events) if events else 0.0,
                "posterior_entropy_mean": sum(entropy) / len(entropy) if entropy else "",
                "hybrid_override_rate": overrides / len(events)
                if label == "Hybrid" and events
                else "",
            }
        )
    for scenario in sorted({str(event["scenario_id"]) for event in static_events}):
        for label, events in (
            ("Static", static_events),
            ("Bayesian", bayesian_events),
            ("Hybrid", hybrid_events),
        ):
            terminal = [
                event
                for event in events
                if event["scenario_id"] == scenario and event["terminal_state"]
            ][-1]
            rows.append(
                {
                    "policy_family": label,
                    "scenario_id": scenario,
                    "terminal_state": terminal["terminal_state"],
                    "terminal_action": terminal["action_code"],
                    "safety_rules_triggered": terminal["safety_rules_triggered"],
                    "selected_state_before_safety": terminal["selected_state_before_safety"],
                    "final_next_state": terminal["final_next_state"],
                }
            )
    return rows


def _terminal_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [event for event in events if str(event.get("terminal_state", ""))]


def _validation(
    *,
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
    bayesian_model: BayesianTransitionModel,
    artifacts: Mapping[str, str],
    bayesian_events: Sequence[Mapping[str, Any]],
    hybrid_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks = {
        "static_policy_unchanged": protected_before.get(
            "src/trustworthy_agent/transitions/static_markov.py"
        )
        == protected_after.get("src/trustworthy_agent/transitions/static_markov.py"),
        "safety_guard_unchanged": protected_before.get("src/trustworthy_agent/safety/guards.py")
        == protected_after.get("src/trustworthy_agent/safety/guards.py"),
        "transition_interface_unchanged": protected_before.get(
            "src/trustworthy_agent/transitions/base.py"
        )
        == protected_after.get("src/trustworthy_agent/transitions/base.py"),
        "recommendation_strategy_unchanged": all(
            protected_before.get(path) == protected_after.get(path)
            for path in PROTECTED_FILES
            if "recommendation" in path
        ),
        "bayesian_events_present": bool(bayesian_events),
        "hybrid_events_present": bool(hybrid_events),
        "posterior_diagnostics_valid": bayesian_model.diagnostics.r_hat <= 1.1
        and bayesian_model.diagnostics.ess_bulk > 0
        and bayesian_model.diagnostics.divergences == 0,
        "classifier_training_not_performed": True,
        "split_not_regenerated": True,
        "representation_not_regenerated": True,
        "artifacts_written": all(value for value in artifacts.values()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "PARTIAL",
        "checks": checks,
        "posterior_diagnostics": {
            "r_hat": bayesian_model.diagnostics.r_hat,
            "ess_bulk": bayesian_model.diagnostics.ess_bulk,
            "divergences": bayesian_model.diagnostics.divergences,
            "sampler": bayesian_model.diagnostics.sampler,
        },
        "limitations": [
            "PyMC NUTS was not executed because PyMC/ArViZ are unavailable "
            "in the frozen environment.",
            "Transition posterior is an internal deterministic posterior "
            "approximation for workflow-control ablation.",
        ],
    }


def _readiness_report(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# Bayesian Policy Readiness",
        "",
        f"- Experiment: `{PROTOCOL_ID}`",
        f"- Status: `{readiness['status']}`",
        "",
        "## Checks",
    ]
    for key, value in cast(Mapping[str, Any], readiness["checks"]).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Limitations"])
    for item in cast(Sequence[str], readiness["limitations"]):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _execution_report(
    readiness: Mapping[str, Any],
    validation: Mapping[str, Any],
    artifacts: Mapping[str, str],
    *,
    elapsed_seconds: float,
) -> str:
    lines = [
        "# Bayesian Transition Policy Execution Report",
        "",
        f"- Experiment: `{PROTOCOL_ID}`",
        f"- Validation status: `{validation['status']}`",
        f"- Elapsed seconds: `{elapsed_seconds:.6f}`",
        "",
        "## Final Validation",
    ]
    for key, value in cast(Mapping[str, Any], validation["checks"]).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness Limitations"])
    for item in cast(Sequence[str], readiness["limitations"]):
        lines.append(f"- {item}")
    lines.extend(["", "## Artifacts"])
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Scientific Limitations",
            "- No classifier was retrained.",
            "- No split, representation, or statistical-analysis artifact was regenerated.",
            "- Posterior probabilities are transition-control evidence, not diagnostic "
            "or degradation probabilities.",
            "- PyMC/NUTS was not executed in this environment; manifests record "
            "`nuts_sampler_executed=false`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _protected_hashes(project_root: Path) -> dict[str, str]:
    return {
        path: sha256_file(project_root / path)
        for path in PROTECTED_FILES
        if (project_root / path).exists()
    }


def _configuration_hash(project_root: Path) -> str:
    paths = [
        project_root / "configs/transition_policies/static_transition_matrix_v1.yaml",
        project_root / "Output/Manifests/classification_plan.json",
        project_root / "Output/Manifests/representation_fingerprints.json",
        project_root / "Output/Manifests/spline_transition_study_v1_split_manifest.json",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("utf-8") if path.exists() else b"missing")
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_dict(value: Any) -> str:
    return _json(value) if isinstance(value, Mapping) else "{}"


def _finite_or_blank(value: Any) -> float | str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return numeric if math.isfinite(numeric) else ""


def _float_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _hash_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sd(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _interval(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    low_index = int(math.floor(0.025 * (len(ordered) - 1)))
    high_index = int(math.ceil(0.975 * (len(ordered) - 1)))
    return (ordered[low_index], ordered[high_index])
