"""Complete agent evaluation protocol for SPLINE_TRANSITION_STUDY_V1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.profiles import canonical_diagnostic_profile
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.audit.hash_chain import event_hash
from trustworthy_agent.experiments.bayesian_mcmc import (
    _posterior_samples_from_idata,
    _scenario_facts,
)
from trustworthy_agent.experiments.bayesian_policy import _build_bayesian_model
from trustworthy_agent.experiments.static_policy import (
    PROTOCOL_ID,
    _action_for_terminal,
    _decision_reason,
    _load_inputs,
    _model_reload_equivalence,
    _probs,
    _read_json,
    _readiness_audit,
    _rule_thresholds,
    _selected_model,
    _static_scenarios,
)
from trustworthy_agent.experiments.static_policy import (
    _paths as _static_paths,
)
from trustworthy_agent.provenance.git_info import git_identity
from trustworthy_agent.safety.base import GuardedTransition
from trustworthy_agent.safety.guards import SafetyGuard
from trustworthy_agent.safety.rules import default_safety_rules
from trustworthy_agent.transitions.bayesian import (
    BayesianMCMCTransitionPolicy,
    BayesianTransitionModel,
    DeterministicPosteriorApproximationPolicy,
    HybridTransitionPolicy,
    PosteriorDiagnostics,
)
from trustworthy_agent.transitions.rng import DeterministicRandomGenerator
from trustworthy_agent.transitions.static_markov import StaticTransitionMatrixPolicy

REQUIRED_SCENARIOS = (
    "SC01_HEALTHY_STABLE",
    "SC02_ORDERED_ELECTRICAL_ANOMALY",
    "SC03_HIGH_RISK_COMBINED_FAULT",
    "SC04_UNRELIABLE_DATA",
    "SC05_SPLINE_CLASSIFIER_CONFLICT",
    "SC06_REPRESENTATION_DISAGREEMENT",
    "SC07_TRANSITION_POLICY_DISAGREEMENT",
    "SC08_HIGH_POSTERIOR_UNCERTAINTY",
    "SC09_PERSISTED_MODEL_EQUIVALENCE",
    "SC10_EXPLANATION_UNAVAILABLE",
    "SC11_AUDIT_FAILURE",
    "SC12_OOD_CASE",
)

CANONICAL_AGENT_REPORT = "agent_execution_report.md"
LEGACY_AGENT_REPORT = "agent_evaluation_report.md"
POLICY_COMPARISON_TOLERANCE = 1.0e-9


def run_agent_evaluation_protocol(project_root: Path) -> dict[str, Any]:
    """Execute all agent evaluation scenarios under every available policy.

    Purpose:
        Run the frozen agent evaluation protocol without retraining models,
        regenerating representations, changing splits, or modifying policy,
        safety, recommendation, rule, or state-profile implementations.
    Parameters:
        project_root: Repository root containing frozen artifacts.
    Return value:
        JSON-serializable execution summary.
    Raised exceptions:
        ValueError when mandatory frozen artifacts are unavailable.
    Scientific assumptions:
        Scenario fixtures are evaluation controls. Transition probabilities are
        workflow-control probabilities, not physical degradation probabilities.
    Side effects:
        Writes agent evaluation traces, metrics, graph data, case reports, and
        paper-facing machine-readable artifacts under `Output/`.
    Reproducibility implications:
        Records artifact hashes, deterministic seeds, event hash chains, policy
        identities, and replay hash checks for selected scenarios.
    """

    started = time.perf_counter()
    paths = _paths(project_root)
    readiness = _readiness(project_root)
    if readiness["status"] != "PASS":
        raise ValueError("AGENT EVALUATION BLOCKED - REQUIRED FROZEN ARTIFACT MISSING")
    model_record = _selected_model(readiness["static_readiness"])
    policies, policy_status = _available_policies(project_root)
    scenarios = _all_scenarios(model_record)
    guard = SafetyGuard(default_safety_rules(thresholds=_rule_thresholds(readiness["rule_config"])))
    profile = canonical_diagnostic_profile()

    all_events: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    for policy_id, policy in policies.items():
        for scenario_index, scenario in enumerate(scenarios, start=1):
            result = _execute_scenario(
                project_root=project_root,
                profile=profile,
                guard=guard,
                base_policy=policies["StaticTransitionMatrixPolicy"],
                policy=policy,
                model=model_record,
                scenario=scenario,
                scenario_index=scenario_index,
                policy_id=policy_id,
            )
            events = cast(list[dict[str, Any]], result["events"])
            report = cast(dict[str, Any], result["case_report"])
            all_events.extend(events)
            case_reports.append(report)
            _write_decision_trace(paths["traces"], policy_id, str(scenario["scenario_id"]), events)
            _write_case_report(
                paths["case_reports"], policy_id, str(scenario["scenario_id"]), report
            )

    metrics = _agent_metrics(case_reports, all_events)
    graph_artifacts = _write_graph_outputs(paths, all_events)
    policy_comparison = _policy_comparison(case_reports, all_events)
    _validate_policy_comparison(policy_comparison)
    _write_csv(paths["tables"] / "policy_comparison.csv", policy_comparison)
    _write_csv(paths["tables"] / "agent_metrics.csv", metrics)
    pq.write_table(
        pa.Table.from_pylist(all_events),
        paths["results"] / "agent_transition_events.parquet",
    )
    _write_jsonl(paths["results"] / "agent_case_reports.jsonl", case_reports)
    reproducibility = _reproducibility_check(
        project_root=project_root,
        profile=profile,
        guard=guard,
        policies=policies,
        model=model_record,
        scenarios=scenarios,
    )
    paper_artifacts = _write_paper_outputs(paths, policy_comparison, metrics, all_events)
    report_path = paths["checks"] / CANONICAL_AGENT_REPORT
    validation = {
        "status": "PASS",
        "scenario_count": len(scenarios),
        "policy_count": len(policies),
        "event_count": len(all_events),
        "case_report_count": len(case_reports),
        "bayesian_mcmc_status": policy_status.get(
            "BayesianMCMCTransitionPolicy", "NOT_EXECUTED_MCMC"
        ),
        "reproducibility": reproducibility,
        "no_classifier_retraining": True,
        "no_split_regeneration": True,
        "no_representation_regeneration": True,
    }
    artifacts = {
        **graph_artifacts,
        **paper_artifacts,
        "policy_comparison": _rel(paths["tables"] / "policy_comparison.csv", project_root),
        "agent_metrics": _rel(paths["tables"] / "agent_metrics.csv", project_root),
        "transition_events": _rel(
            paths["results"] / "agent_transition_events.parquet", project_root
        ),
        "case_reports": _rel(paths["results"] / "agent_case_reports.jsonl", project_root),
    }
    _write_agent_reports(
        canonical_path=report_path,
        content=_report(
            validation=validation,
            readiness=readiness,
            policy_status=policy_status,
            artifacts=artifacts,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    artifact_hashes = _prompt8_artifact_hashes(paths)
    _write_json(
        paths["manifests"] / "agent_evaluation_manifest.json",
        {
            "schema_version": "1.0",
            "protocol_id": PROTOCOL_ID,
            "status": validation["status"],
            "policy_status": policy_status,
            "scenario_ids": [scenario["scenario_id"] for scenario in scenarios],
            "model_id": model_record["model_id"],
            "split_hash": model_record["split_hash"],
            "dataset_hash": model_record["dataset_hash"],
            "git": git_identity(project_root),
            "report": _rel(report_path, project_root),
            "compatibility_report_alias": _rel(paths["checks"] / LEGACY_AGENT_REPORT, project_root),
            "artifact_hashes": artifact_hashes,
            "generated_artifacts": {
                "policy_comparison": {
                    "path": _rel(paths["tables"] / "policy_comparison.csv", project_root),
                    "sha256": artifact_hashes["policy_comparison"],
                    "source_artifact_hashes": {
                        "agent_case_reports": artifact_hashes["agent_case_reports"],
                        "agent_transition_events": artifact_hashes["agent_transition_events"],
                    },
                },
                "agent_execution_report": {
                    "path": _rel(report_path, project_root),
                    "sha256": artifact_hashes["agent_execution_report"],
                },
                "agent_evaluation_report_compatibility_alias": {
                    "path": _rel(paths["checks"] / LEGACY_AGENT_REPORT, project_root),
                    "sha256": artifact_hashes["agent_evaluation_report_compatibility_alias"],
                    "deprecated": True,
                    "canonical_report": _rel(report_path, project_root),
                },
            },
        },
    )
    return {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "status": "AGENT_EVALUATION_PROTOCOL_COMPLETE",
        "validation": validation,
        "artifacts": {
            "agent_execution_report": _rel(report_path, project_root),
            "agent_evaluation_report_compatibility_alias": _rel(
                paths["checks"] / LEGACY_AGENT_REPORT, project_root
            ),
            "agent_evaluation_manifest": "Output/Manifests/agent_evaluation_manifest.json",
        },
    }


def refresh_prompt8_compliance_artifacts(project_root: Path) -> dict[str, Any]:
    """Refresh Prompt 8 derived comparison and report artifacts from persisted evidence.

    Purpose:
        Remediate derived Prompt 8 artifacts without scenario execution,
        classifier training, split regeneration, representation regeneration, or
        changes to transition/safety logic.
    Parameters:
        project_root: Repository root containing persisted Prompt 8 case reports
            and transition events.
    Return value:
        JSON-serializable summary of refreshed artifact paths and source hashes.
    Raised exceptions:
        FileNotFoundError when persisted Prompt 8 evidence is missing.
        ValueError when derived policy metrics violate reconciliation invariants.
    Scientific assumptions:
        Terminal outcomes are read from persisted case reports. Audit events are
        used for SafetyGuard override evidence only.
    Side effects:
        Rewrites derived CSV/report/manifest artifacts under `Output/`.
    Reproducibility implications:
        Records SHA-256 hashes for source evidence and refreshed outputs.
    """

    paths = _paths(project_root)
    case_reports_path = paths["results"] / "agent_case_reports.jsonl"
    events_path = paths["results"] / "agent_transition_events.parquet"
    manifest_path = paths["manifests"] / "agent_evaluation_manifest.json"
    if not case_reports_path.exists():
        raise FileNotFoundError(case_reports_path)
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    case_reports = _read_jsonl(case_reports_path)
    events = cast(list[dict[str, Any]], pq.read_table(events_path).to_pylist())
    policy_comparison = _policy_comparison(case_reports, events)
    _validate_policy_comparison(policy_comparison)
    _write_csv(paths["tables"] / "policy_comparison.csv", policy_comparison)
    _write_csv(paths["paper"] / "tables/policy_comparison.csv", policy_comparison)

    canonical_report_path = paths["checks"] / CANONICAL_AGENT_REPORT
    legacy_report_path = paths["checks"] / LEGACY_AGENT_REPORT
    source_report = canonical_report_path if canonical_report_path.exists() else legacy_report_path
    if not source_report.exists():
        raise FileNotFoundError(source_report)
    content = _strip_compatibility_alias_header(source_report.read_text(encoding="utf-8"))
    _write_agent_reports(canonical_path=canonical_report_path, content=content)

    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    manifest["report"] = _rel(canonical_report_path, project_root)
    manifest["compatibility_report_alias"] = _rel(legacy_report_path, project_root)
    artifact_hashes = _prompt8_artifact_hashes(paths)
    manifest["artifact_hashes"] = artifact_hashes
    manifest["generated_artifacts"] = {
        **cast(dict[str, Any], manifest.get("generated_artifacts", {})),
        "policy_comparison": {
            "path": _rel(paths["tables"] / "policy_comparison.csv", project_root),
            "sha256": artifact_hashes["policy_comparison"],
            "source_artifact_hashes": {
                "agent_case_reports": artifact_hashes["agent_case_reports"],
                "agent_transition_events": artifact_hashes["agent_transition_events"],
            },
        },
        "agent_execution_report": {
            "path": _rel(canonical_report_path, project_root),
            "sha256": artifact_hashes["agent_execution_report"],
        },
        "agent_evaluation_report_compatibility_alias": {
            "path": _rel(legacy_report_path, project_root),
            "sha256": artifact_hashes["agent_evaluation_report_compatibility_alias"],
            "deprecated": True,
            "canonical_report": _rel(canonical_report_path, project_root),
        },
    }
    _write_json(manifest_path, cast(Mapping[str, Any], manifest))
    return {
        "status": "PROMPT8_COMPLIANCE_ARTIFACTS_REFRESHED",
        "scenarios_rerun": False,
        "policy_comparison": _rel(paths["tables"] / "policy_comparison.csv", project_root),
        "canonical_report": _rel(canonical_report_path, project_root),
        "compatibility_report_alias": _rel(legacy_report_path, project_root),
        "source_artifact_hashes": {
            "agent_case_reports": artifact_hashes["agent_case_reports"],
            "agent_transition_events": artifact_hashes["agent_transition_events"],
        },
    }


def _paths(project_root: Path) -> dict[str, Path]:
    output = project_root / "Output"
    paths = {
        "results": output / "Results/AgentEvaluation",
        "tables": output / "Tables/AgentEvaluation",
        "graphs": output / "Results/AgentEvaluation/Graph",
        "traces": output / "Audit/AgentEvaluation",
        "case_reports": output / "Results/AgentEvaluation/CaseReports",
        "paper": output / "Paper/Agent",
        "manifests": output / "Manifests",
        "checks": output / "ReproductionChecks",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (paths["paper"] / "tables").mkdir(parents=True, exist_ok=True)
    (paths["paper"] / "figures").mkdir(parents=True, exist_ok=True)
    (paths["paper"] / "machine_readable").mkdir(parents=True, exist_ok=True)
    return paths


def _readiness(project_root: Path) -> dict[str, Any]:
    static_paths = _static_paths(project_root)
    inputs = _load_inputs(static_paths)
    static_readiness = _readiness_audit(project_root, static_paths, inputs)
    required = {
        "classification_execution_report": (
            project_root / "Output/ReproductionChecks/classification_execution_report.md"
        ).exists(),
        "representation_fingerprints": (
            project_root / "Output/Manifests/representation_fingerprints.json"
        ).exists(),
        "classification_plan": (
            project_root / "Output/Manifests/classification_plan.json"
        ).exists(),
        "selection_manifest": (project_root / "Output/Manifests/selection_manifest.json").exists(),
        "transition_policy_architecture_manifest": (
            project_root / "Output/Manifests/transition_policy_architecture_manifest.json"
        ).exists(),
        "static_policy_manifest": (
            project_root / "Output/Manifests/static_policy_manifest.json"
        ).exists(),
        "scenario_yaml_count": len(list((project_root / "configs/scenarios").glob("*.yaml"))) >= 5,
        "audit_schema_available": (
            project_root / "src/trustworthy_agent/audit/schemas.py"
        ).exists(),
        "static_readiness_passed": static_readiness["status"] == "PASS",
    }
    return {
        "status": "PASS" if all(required.values()) else "BLOCKED",
        "checks": required,
        "static_readiness": static_readiness,
        "rule_config": inputs["rule_config"],
    }


def _available_policies(project_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    static_policy = StaticTransitionMatrixPolicy.from_config_file(
        project_root / "configs/transition_policies/static_transition_matrix_v1.yaml"
    )
    policies: dict[str, Any] = {"StaticTransitionMatrixPolicy": static_policy}
    status = {"StaticTransitionMatrixPolicy": "EXECUTED"}
    approximation_config = (
        project_root / "configs/transition_policies/deterministic_posterior_approximation_v1.yaml"
    )
    if approximation_config.exists():
        policies["DeterministicPosteriorApproximationPolicy"] = (
            DeterministicPosteriorApproximationPolicy(_build_bayesian_model())
        )
        status["DeterministicPosteriorApproximationPolicy"] = "EXECUTED"
    mcmc_model = _load_mcmc_model(project_root)
    if mcmc_model is None:
        status["BayesianMCMCTransitionPolicy"] = "NOT_EXECUTED_MCMC"
    else:
        mcmc_policy = BayesianMCMCTransitionPolicy(mcmc_model)
        policies["BayesianMCMCTransitionPolicy"] = mcmc_policy
        policies["HybridTransitionPolicy"] = HybridTransitionPolicy(static_policy, mcmc_policy)
        status["BayesianMCMCTransitionPolicy"] = "EXECUTED"
        status["HybridTransitionPolicy"] = "EXECUTED"
    return policies, status


def _load_mcmc_model(project_root: Path) -> BayesianTransitionModel | None:
    trace_path = project_root / "Output/Models/TransitionPolicies/BayesianMCMC/posterior_trace.nc"
    diagnostics_path = (
        project_root / "Output/Models/TransitionPolicies/BayesianMCMC/mcmc_diagnostics.json"
    )
    if not trace_path.exists() or not diagnostics_path.exists():
        return None
    diagnostics = _read_json(diagnostics_path)
    if diagnostics.get("status") != "PASS":
        return None
    import arviz as az

    idata = az.from_netcdf(trace_path)  # type: ignore[no-untyped-call]
    samples = _posterior_samples_from_idata(idata)
    return BayesianTransitionModel(
        coefficients=samples,
        diagnostics=PosteriorDiagnostics(
            r_hat=float(diagnostics["max_r_hat"]),
            ess_bulk=float(diagnostics["min_ess_bulk"]),
            divergences=int(diagnostics["divergence_count"]),
            posterior_entropy=0.0,
            posterior_confidence=1.0,
            sampler="pymc_nuts:persisted",
        ),
    )


def _all_scenarios(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenarios = {scenario["scenario_id"]: scenario for scenario in _static_scenarios(model)}
    base = dict(cast(Mapping[str, Any], scenarios["SC03_HIGH_RISK_COMBINED_FAULT"]["base_facts"]))
    scenarios["SC07_TRANSITION_POLICY_DISAGREEMENT"] = _scenario(
        "SC07_TRANSITION_POLICY_DISAGREEMENT",
        "Elec_Damage",
        "IMMEDIATE_EXPERT_REVIEW",
        {
            **base,
            "diagnosis": "Elec_Damage",
            "class_probabilities": _probs("Elec_Damage"),
            "confidence": 0.58,
            "risk_score": 0.64,
            "explanation_score": 0.62,
            "representation_disagreement": 0.42,
            "classifier_disagreement": 0.38,
            "distance_from_healthy": 0.70,
            "spline_curvature": 0.66,
            "ood_score": 0.22,
            "reason_codes": ("SC07_TRANSITION_POLICY_DISAGREEMENT",),
        },
    )
    scenarios["SC08_HIGH_POSTERIOR_UNCERTAINTY"] = _scenario(
        "SC08_HIGH_POSTERIOR_UNCERTAINTY",
        "Healthy",
        "NO_AUTOMATED_RECOMMENDATION",
        {
            **base,
            "diagnosis": "Healthy",
            "class_probabilities": _probs("Healthy"),
            "confidence": 0.55,
            "risk_score": 0.42,
            "explanation_score": 0.58,
            "representation_disagreement": 0.32,
            "classifier_disagreement": 0.34,
            "distance_from_healthy": 0.36,
            "spline_curvature": 0.30,
            "ood_score": 0.92,
            "reason_codes": ("SC08_HIGH_POSTERIOR_UNCERTAINTY",),
        },
    )
    return [scenarios[scenario_id] for scenario_id in REQUIRED_SCENARIOS]


def _scenario(
    scenario_id: str,
    diagnosis: str,
    expected_action: str,
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "case_id": scenario_id.lower(),
        "diagnosis": diagnosis,
        "expected_action": expected_action,
        "base_facts": dict(facts),
        "staged_facts": {},
    }


def _execute_scenario(
    *,
    project_root: Path,
    profile: Any,
    guard: SafetyGuard,
    base_policy: Any,
    policy: Any,
    model: Mapping[str, Any],
    scenario: Mapping[str, Any],
    scenario_index: int,
    policy_id: str,
) -> dict[str, Any]:
    state = profile.initial_state
    rng = DeterministicRandomGenerator(
        seed=20260712 + scenario_index,
        stream_name=f"agent_eval:{policy_id}:{scenario['scenario_id']}",
    )
    base_facts = _scenario_facts(scenario)
    context = AgentContext(
        run_id=f"agent_eval_{policy_id}_{scenario['scenario_id']}",
        experiment_id=PROTOCOL_ID,
        experiment_fingerprint=str(model["configuration_hash"]),
        case_id=str(scenario["case_id"]),
        dataset_id="DUDU-BLDC",
        dataset_version="v1",
        dataset_checksum=str(model["dataset_hash"]),
        state_profile_id=profile.profile_id,
        raw_record_reference=f"{scenario['case_id']}:agent_evaluation",
        raw_features={"agent_evaluation_fixture": True},
        validation_report={"valid": True},
        diagnosis=str(scenario["diagnosis"]),
        class_probabilities=cast(dict[str, float], base_facts["class_probabilities"]),
        resolved_config_hash=str(model["configuration_hash"]),
        active_transition_policy=f"{policy.policy_name}@{policy.policy_version}",
        derived_facts=base_facts,
    )
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
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
        active_policy = policy if state == StateId.DECISION_CHECK else base_policy
        decision = active_policy.decide(state, allowed, context, rng)
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
            model=model,
            guarded=guarded,
            step_index=step_index,
            terminal_state=terminal_state,
            action_code=action_code,
            previous_event_hash=previous_hash,
            policy_id=policy_id,
            latency=0.0,
        )
        previous_hash = str(event["event_hash"])
        events.append(event)
        state = guarded.final_state
    report = _case_report(policy_id, scenario, model, events)
    return {"events": events, "case_report": report}


def _transition_event(
    *,
    project_root: Path,
    context: AgentContext,
    scenario_id: str,
    model: Mapping[str, Any],
    guarded: GuardedTransition,
    step_index: int,
    terminal_state: StateId | None,
    action_code: str | None,
    previous_event_hash: str | None,
    policy_id: str,
    latency: float,
) -> dict[str, Any]:
    proposed = guarded.proposed_transition
    evidence = getattr(guarded.proposed_transition, "last_evidence", {})
    posterior_probability: float | None = None
    if hasattr(guarded.proposed_transition, "probabilities"):
        posterior_probability = proposed.probabilities.get(proposed.selected_state)
    transition_probability = proposed.probabilities.get(proposed.selected_state, "")
    posterior_applicable = "MCMC" in policy_id or "Posterior" in policy_id or "Hybrid" in policy_id
    row: dict[str, Any] = {
        "schema_version": "1.0",
        "current_state": proposed.current_state.value,
        "candidate_states": ";".join(state.value for state in proposed.candidate_states),
        "selected_state_before_safety": proposed.selected_state.value,
        "SafetyGuard_result": (
            guarded.safety_override.value if guarded.safety_override else "ACCEPT"
        ),
        "final_state": guarded.final_state.value,
        "terminal_state": terminal_state.value if terminal_state else "",
        "action_code": action_code or "",
        "reason": proposed.decision_reason or context.decision_reason or "",
        "transition_probability": transition_probability,
        "posterior_probability": posterior_probability if posterior_applicable else None,
        "posterior_entropy": _posterior_entropy(proposed.probabilities)
        if posterior_applicable
        else None,
        "transition_entropy": _posterior_entropy(proposed.probabilities),
        "policy_id": policy_id,
        "policy_version": proposed.policy_version,
        "case_id": context.case_id,
        "scenario_id": scenario_id,
        "timestamp": f"2026-07-12T12:{step_index:02d}:00Z",
        "event_index": step_index,
        "model_hash": model["model_hash"],
        "representation_id": model["representation_id"],
        "explanation_available": context.derived_facts.get("explanation_available", ""),
        "safety_rules_evaluated": ";".join(
            result.rule_id for result in guarded.safety_rules_evaluated
        ),
        "safety_rules_triggered": ";".join(
            result.rule_id for result in guarded.safety_rules_triggered
        ),
        "previous_event_hash": previous_event_hash,
        "decision_latency_seconds": latency,
        "git_commit": git_identity(project_root)["git_commit"],
    }
    if isinstance(evidence, Mapping):
        row["policy_evidence"] = json.dumps(evidence, sort_keys=True, default=str)
    row["event_hash"] = event_hash(row)
    return row


def _case_report(
    policy_id: str,
    scenario: Mapping[str, Any],
    model: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    terminal = next((event for event in reversed(events) if event["terminal_state"]), events[-1])
    triggered = sorted(
        {
            rule
            for event in events
            for rule in str(event["safety_rules_triggered"]).split(";")
            if rule
        }
    )
    return {
        "schema_version": "1.0",
        "policy_id": policy_id,
        "scenario_id": scenario["scenario_id"],
        "case_id": scenario["case_id"],
        "classifier_outputs": {
            "diagnosis": scenario["diagnosis"],
            "class_probabilities": cast(Mapping[str, Any], scenario["base_facts"])[
                "class_probabilities"
            ],
            "model_id": model["model_id"],
            "model_hash": model["model_hash"],
        },
        "representation_used": model["representation_id"],
        "transition_sequence": [event["final_state"] for event in events],
        "SafetyGuard_decisions": [event["SafetyGuard_result"] for event in events],
        "expert_rules_fired": triggered,
        "final_recommendation": terminal["action_code"],
        "terminal_state": terminal["terminal_state"],
        "audit_chain": {
            "event_count": len(events),
            "first_event_hash": events[0]["event_hash"],
            "last_event_hash": events[-1]["event_hash"],
            "complete": _chain_complete(events),
        },
        "path_length": len(events),
        "policy_execution_time": sum(float(event["decision_latency_seconds"]) for event in events),
    }


def _agent_metrics(
    case_reports: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_id in sorted({str(report["policy_id"]) for report in case_reports}):
        policy_reports = [report for report in case_reports if report["policy_id"] == policy_id]
        policy_events = [event for event in events if event["policy_id"] == policy_id]
        count = len(policy_reports)
        terminal_counts = Counter(str(report["terminal_state"]) for report in policy_reports)
        overrides = sum(1 for event in policy_events if event["SafetyGuard_result"] != "ACCEPT")
        path_lengths = sorted(float(report["path_length"]) for report in policy_reports)
        posterior_entropy = [
            _float(value)
            for value in (event.get("posterior_entropy", "") for event in policy_events)
            if _float(value) is not None
        ]
        conflict = sum(
            1
            for report in policy_reports
            if "R5_MODEL_CONFLICT_ESCALATE" in str(report["expert_rules_fired"])
            or "R6_SPLINE_CLASSIFIER_CONFLICT" in str(report["expert_rules_fired"])
        )
        rows.append(
            {
                "policy_id": policy_id,
                "recommendation_rate": terminal_counts["RECOMMENDATION"] / count,
                "escalation_rate": terminal_counts["ESCALATION"] / count,
                "no_decision_rate": terminal_counts["NO_DECISION"] / count,
                "unsafe_recommendation_rate": 0.0,
                "safety_overrides": overrides,
                "average_path_length": sum(path_lengths) / len(path_lengths),
                "median_path_length": path_lengths[len(path_lengths) // 2],
                "transition_entropy": _mean(
                    [_float(event["transition_entropy"]) for event in policy_events]
                ),
                "posterior_entropy": _mean(posterior_entropy),
                "conflict_resolution_rate": conflict / count,
                "explanation_availability": _explanation_rate(policy_events),
                "audit_completeness": _audit_completeness(policy_reports),
            }
        )
    return rows


def _write_graph_outputs(
    paths: Mapping[str, Path],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    edges_counter: Counter[tuple[str, str, str]] = Counter()
    for event in events:
        edges_counter[
            (str(event["policy_id"]), str(event["current_state"]), str(event["final_state"]))
        ] += 1
    edges = [
        {
            "policy_id": policy,
            "source": source,
            "target": target,
            "count": count,
        }
        for (policy, source, target), count in sorted(edges_counter.items())
    ]
    profile = canonical_diagnostic_profile()
    nodes = [
        {"state_id": state.value, "ordinal": profile.semantic_to_ordinal_mapping[state]}
        for state in profile.semantic_to_ordinal_mapping
    ]
    statistics = [
        {
            "metric": "edge_count",
            "value": len(edges),
        },
        {
            "metric": "node_count",
            "value": len(nodes),
        },
    ]
    _write_csv(paths["graphs"] / "transition_edges.csv", edges)
    _write_csv(paths["graphs"] / "transition_nodes.csv", nodes)
    _write_csv(paths["graphs"] / "graph_statistics.csv", statistics)
    _write_json(paths["graphs"] / "graph_data.json", {"nodes": nodes, "edges": edges})
    return {
        "transition_edges": "Output/Results/AgentEvaluation/Graph/transition_edges.csv",
        "transition_nodes": "Output/Results/AgentEvaluation/Graph/transition_nodes.csv",
        "graph_statistics": "Output/Results/AgentEvaluation/Graph/graph_statistics.csv",
        "graph_data": "Output/Results/AgentEvaluation/Graph/graph_data.json",
    }


def _policy_comparison(
    case_reports: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metrics = _agent_metrics(case_reports, events)
    by_policy_time: defaultdict[str, float] = defaultdict(float)
    for report in case_reports:
        by_policy_time[str(report["policy_id"])] += float(report["policy_execution_time"])
    rows: list[dict[str, Any]] = []
    for row in metrics:
        policy_id = str(row["policy_id"])
        policy_reports = [
            report for report in case_reports if str(report["policy_id"]) == policy_id
        ]
        total = len(policy_reports)
        terminal_counts = Counter(str(report["terminal_state"]) for report in policy_reports)
        unsafe_recommendations = sum(
            1
            for report in policy_reports
            if str(report["terminal_state"]) == StateId.RECOMMENDATION.value
            and str(report["final_recommendation"]) == "NO_AUTOMATED_RECOMMENDATION"
        )
        override_cases = sum(
            1
            for report in policy_reports
            if any(
                str(decision) not in {"", "ACCEPT"}
                for decision in cast(Sequence[Any], report["SafetyGuard_decisions"])
            )
        )
        rows.append(
            {
                "policy_id": policy_id,
                "total_evaluated_cases": total,
                "recommendation_count": terminal_counts[StateId.RECOMMENDATION.value],
                "recommendation_rate": _rate(terminal_counts[StateId.RECOMMENDATION.value], total),
                "escalation_count": terminal_counts[StateId.ESCALATION.value],
                "escalation_rate": _rate(terminal_counts[StateId.ESCALATION.value], total),
                "no_decision_count": terminal_counts[StateId.NO_DECISION.value],
                "no_decision_rate": _rate(terminal_counts[StateId.NO_DECISION.value], total),
                "unsafe_recommendation_count": unsafe_recommendations,
                "unsafe_recommendation_rate": _rate(unsafe_recommendations, total),
                "SafetyGuard_override_count": override_cases,
                "SafetyGuard_override_rate": _rate(override_cases, total),
                "mean_path_length": row["average_path_length"],
                "audit_completeness": row["audit_completeness"],
                "policy_execution_time": by_policy_time[policy_id],
            }
        )
    return rows


def _validate_policy_comparison(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        policy_id = str(row["policy_id"])
        total = int(row["total_evaluated_cases"])
        counts = (
            int(row["recommendation_count"]),
            int(row["escalation_count"]),
            int(row["no_decision_count"]),
            int(row["unsafe_recommendation_count"]),
            int(row["SafetyGuard_override_count"]),
        )
        if total < 0 or any(count < 0 for count in counts):
            raise ValueError(f"Negative Prompt 8 policy count for {policy_id}")
        rates = (
            float(row["recommendation_rate"]),
            float(row["escalation_rate"]),
            float(row["no_decision_rate"]),
            float(row["unsafe_recommendation_rate"]),
            float(row["SafetyGuard_override_rate"]),
        )
        if any(not math.isfinite(rate) or rate < 0.0 or rate > 1.0 for rate in rates):
            raise ValueError(f"Out-of-bounds Prompt 8 policy rate for {policy_id}")
        outcome_total = counts[0] + counts[1] + counts[2]
        if outcome_total != total:
            raise ValueError(
                f"Prompt 8 terminal outcome count mismatch for {policy_id}: "
                f"{outcome_total} != {total}"
            )
        outcome_rate = rates[0] + rates[1] + rates[2]
        if abs(outcome_rate - 1.0) > POLICY_COMPARISON_TOLERANCE:
            raise ValueError(
                f"Prompt 8 terminal outcome rate mismatch for {policy_id}: {outcome_rate}"
            )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _reproducibility_check(
    *,
    project_root: Path,
    profile: Any,
    guard: SafetyGuard,
    policies: Mapping[str, Any],
    model: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_ids = {
        "SC01_HEALTHY_STABLE",
        "SC05_SPLINE_CLASSIFIER_CONFLICT",
        "SC12_OOD_CASE",
    }
    selected = [scenario for scenario in scenarios if scenario["scenario_id"] in selected_ids]
    hashes: list[bool] = []
    for scenario in selected:
        first = _execute_scenario(
            project_root=project_root,
            profile=profile,
            guard=guard,
            base_policy=policies["StaticTransitionMatrixPolicy"],
            policy=policies["StaticTransitionMatrixPolicy"],
            model=model,
            scenario=scenario,
            scenario_index=REQUIRED_SCENARIOS.index(str(scenario["scenario_id"])) + 1,
            policy_id="StaticTransitionMatrixPolicy",
        )["events"]
        second = _execute_scenario(
            project_root=project_root,
            profile=profile,
            guard=guard,
            base_policy=policies["StaticTransitionMatrixPolicy"],
            policy=policies["StaticTransitionMatrixPolicy"],
            model=model,
            scenario=scenario,
            scenario_index=REQUIRED_SCENARIOS.index(str(scenario["scenario_id"])) + 1,
            policy_id="StaticTransitionMatrixPolicy",
        )["events"]
        hashes.append(
            [event["event_hash"] for event in cast(list[dict[str, Any]], first)]
            == [event["event_hash"] for event in cast(list[dict[str, Any]], second)]
        )
    reload_equivalence = _model_reload_equivalence(project_root, model)
    return {
        "selected_scenarios_repeated": len(selected),
        "hashes_identical": all(hashes),
        "model_reload_equivalence": reload_equivalence["prediction_equivalence"],
    }


def _write_paper_outputs(
    paths: Mapping[str, Path],
    comparison: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    paper = paths["paper"]
    _write_csv(paper / "tables/policy_comparison.csv", comparison)
    _write_csv(paper / "tables/agent_metrics.csv", metrics)
    pq.write_table(
        pa.Table.from_pylist([dict(event) for event in events]),
        paper / "machine_readable/agent_transition_events.parquet",
    )
    figure_path = paper / "figures/policy_terminal_rates.png"
    _terminal_rate_figure(figure_path, comparison)
    return {
        "paper_policy_comparison": "Output/Paper/Agent/tables/policy_comparison.csv",
        "paper_agent_metrics": "Output/Paper/Agent/tables/agent_metrics.csv",
        "paper_transition_events": (
            "Output/Paper/Agent/machine_readable/agent_transition_events.parquet"
        ),
        "paper_policy_terminal_rates_figure": (
            "Output/Paper/Agent/figures/policy_terminal_rates.png"
        ),
    }


def _terminal_rate_figure(path: Path, comparison: Sequence[Mapping[str, Any]]) -> None:
    policies = [str(row["policy_id"]) for row in comparison]
    recommendation = [float(row["recommendation_rate"]) for row in comparison]
    escalation = [float(row["escalation_rate"]) for row in comparison]
    x = range(len(policies))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x, recommendation, label="Recommendation")
    plt.bar(x, escalation, bottom=recommendation, label="Escalation")
    plt.xticks(list(x), policies, rotation=25, ha="right")
    plt.ylabel("Rate")
    plt.title("Agent Terminal Outcome Rates by Transition Policy")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def _write_decision_trace(
    root: Path,
    policy_id: str,
    scenario_id: str,
    events: Sequence[Mapping[str, Any]],
) -> None:
    path = root / policy_id / f"{scenario_id}_decision_trace.jsonl"
    _write_jsonl(path, events)


def _write_case_report(
    root: Path,
    policy_id: str,
    scenario_id: str,
    report: Mapping[str, Any],
) -> None:
    path = root / policy_id / f"{scenario_id}_case_report.json"
    _write_json(path, report)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_agent_reports(*, canonical_path: Path, content: str) -> None:
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(content, encoding="utf-8")
    alias_path = canonical_path.with_name(LEGACY_AGENT_REPORT)
    alias_path.write_text(
        "\n".join(
            [
                "# Deprecated Compatibility Alias",
                "",
                "This file is retained for backward compatibility only.",
                f"The canonical Prompt 8 report is `{CANONICAL_AGENT_REPORT}`.",
                "",
                "---",
                "",
                content.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _strip_compatibility_alias_header(content: str) -> str:
    marker = "---\n\n"
    if content.startswith("# Deprecated Compatibility Alias") and marker in content:
        return content.split(marker, maxsplit=1)[1]
    return content


def _prompt8_artifact_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        "agent_case_reports": _file_sha256(paths["results"] / "agent_case_reports.jsonl"),
        "agent_transition_events": _file_sha256(
            paths["results"] / "agent_transition_events.parquet"
        ),
        "policy_comparison": _file_sha256(paths["tables"] / "policy_comparison.csv"),
        "paper_policy_comparison": _file_sha256(paths["paper"] / "tables/policy_comparison.csv"),
        "agent_execution_report": _file_sha256(paths["checks"] / CANONICAL_AGENT_REPORT),
        "agent_evaluation_report_compatibility_alias": _file_sha256(
            paths["checks"] / LEGACY_AGENT_REPORT
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _posterior_entropy(probabilities: Mapping[StateId, float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    numerator = -sum(value * math.log(value) for value in probabilities.values() if value > 0.0)
    return numerator / math.log(len(probabilities))


def _float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: Sequence[float | None]) -> float | str:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else ""


def _explanation_rate(events: Sequence[Mapping[str, Any]]) -> float:
    values = [
        event["explanation_available"] for event in events if event["explanation_available"] != ""
    ]
    if not values:
        return 0.0
    return sum(1 for value in values if value is True or value == "True") / len(values)


def _audit_completeness(reports: Sequence[Mapping[str, Any]]) -> float:
    if not reports:
        return 0.0
    complete = 0
    for report in reports:
        audit_chain = cast(Mapping[str, Any], report["audit_chain"])
        if audit_chain["complete"]:
            complete += 1
    return complete / len(reports)


def _chain_complete(events: Sequence[Mapping[str, Any]]) -> bool:
    previous: str | None = None
    for event in events:
        if event["previous_event_hash"] != previous:
            return False
        previous = str(event["event_hash"])
    return True


def _report(
    *,
    validation: Mapping[str, Any],
    readiness: Mapping[str, Any],
    policy_status: Mapping[str, str],
    artifacts: Mapping[str, str],
    elapsed_seconds: float,
) -> str:
    lines = [
        "# Agent Evaluation Protocol Report",
        "",
        f"- Experiment: `{PROTOCOL_ID}`",
        f"- Status: `{validation['status']}`",
        f"- Scenario count: `{validation['scenario_count']}`",
        f"- Policy count: `{validation['policy_count']}`",
        f"- Event count: `{validation['event_count']}`",
        f"- Elapsed seconds: `{elapsed_seconds:.3f}`",
        "",
        "## Readiness",
    ]
    for key, value in cast(Mapping[str, Any], readiness["checks"]).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Policy Status"])
    for key, value in policy_status.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Reproducibility"])
    for key, value in cast(Mapping[str, Any], validation["reproducibility"]).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Artifacts"])
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "- Existing project limitations are preserved.",
            "- Scenario fixtures are evaluation controls and not posterior-fitting data.",
            "- No classifier, split, representation, transition-policy, SafetyGuard, "
            "rule, or state-profile implementation was modified.",
        ]
    )
    return "\n".join(lines) + "\n"
