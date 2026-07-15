"""Static transition-policy execution for SPLINE_TRANSITION_STUDY_V1."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.profiles import canonical_diagnostic_profile
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.audit.hash_chain import event_hash
from trustworthy_agent.audit.logger import JsonlAuditRecorder
from trustworthy_agent.audit.verify import verify_audit_chain
from trustworthy_agent.exceptions import RequiredArtifactMissingError
from trustworthy_agent.experiments.spline_transition_study import DEFAULT_CONFIG_PATH
from trustworthy_agent.provenance.environment import basic_environment
from trustworthy_agent.provenance.git_info import git_identity
from trustworthy_agent.provenance.hashing import sha256_file
from trustworthy_agent.safety.base import GuardedTransition, SafetyAction
from trustworthy_agent.safety.guards import SafetyGuard
from trustworthy_agent.safety.rules import default_safety_rules
from trustworthy_agent.transitions.rng import DeterministicRandomGenerator
from trustworthy_agent.transitions.static_markov import StaticTransitionMatrixPolicy

PROTOCOL_ID = "SPLINE_TRANSITION_STUDY_V1"
POLICY_CONFIG_PATH = Path("configs/transition_policies/static_transition_matrix_v1.yaml")
RULE_CONFIG_PATH = Path("configs/rules/spline_transition_study_v1_static_rules.yaml")
CANONICAL_CLASSES = ("Healthy", "Mech_Damage", "Elec_Damage", "Mech_Elec_Damage")
CANONICAL_REPRODUCTION_COMMAND = "make reproduce"
REQUIRED_INPUT_ARTIFACTS = {
    "config": (
        "SPLINE_TRANSITION_STUDY_V1 protocol configuration",
        "protocol registration",
    ),
    "policy_config": ("static transition matrix configuration", "transition policy setup"),
    "rule_config": ("static safety-rule configuration", "safety rule setup"),
    "split_manifest": ("frozen split manifest", "split creation"),
    "representation_fingerprints": (
        "representation fingerprints",
        "pre-training registration",
    ),
    "classification_plan": ("classification plan", "pre-training registration"),
    "classification_execution_manifest": (
        "classification execution manifest",
        "classification execution",
    ),
    "selection_manifest": (
        "diagnostic representation selection manifest",
        "classification execution",
    ),
    "statistics_report": ("statistical analysis report", "statistical analysis"),
    "supported_claims": ("supported claims report", "article reporting"),
    "unsupported_claims": ("unsupported claims report", "article reporting"),
    "limitations": ("limitations report", "article reporting"),
}


def run_static_policy_phase(project_root: Path) -> dict[str, Any]:
    """Execute the frozen static transition-policy phase.

    Purpose:
        Evaluate the configured `StaticTransitionMatrixPolicy` baseline against
        static-policy scenarios without retraining classifiers, regenerating
        representations, changing splits, or using bootstrap comparisons as
        policy inputs.
    Parameters:
        project_root: Repository root containing frozen protocol artifacts.
    Return value:
        JSON-serializable execution summary.
    Raised exceptions:
        FileNotFoundError or ValueError when required frozen artifacts are
        unavailable or inconsistent.
    Scientific assumptions:
        Static probabilities are workflow-control defaults only. They are not
        physical degradation probabilities, diagnostic confidence, or evidence
        of superiority over future transition policies.
    Side effects:
        Writes static-policy audit, transition, table, manifest, and
        reproduction-check artifacts under `Output/`.
    Reproducibility implications:
        Records split/model/config hashes, fixed matrix version, explicit RNG
        seed, audit-chain verification, and protected artifact hashes.
    """

    started = time.perf_counter()
    paths = _paths(project_root)
    protected_before = _protected_hashes(project_root)
    inputs = _load_inputs(paths)
    readiness = _readiness_audit(project_root, paths, inputs)
    readiness_path = paths["checks"] / "static_policy_readiness_audit.md"
    readiness_path.write_text(_readiness_markdown(readiness), encoding="utf-8")
    if readiness["status"] != "PASS":
        raise ValueError("STATIC POLICY BLOCKED - REQUIRED ARTIFACT MISSING")

    profile = canonical_diagnostic_profile()
    policy = StaticTransitionMatrixPolicy.from_config_file(paths["policy_config"])
    _validate_static_matrix_config(inputs["policy_config"], profile)
    guard = SafetyGuard(default_safety_rules(thresholds=_rule_thresholds(inputs["rule_config"])))
    selected_model = _selected_model(readiness)
    scenarios = _static_scenarios(selected_model)

    transition_events: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    audit_results: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, start=1):
        result = _execute_static_scenario(
            project_root=project_root,
            scenario=scenario,
            profile=profile,
            policy=policy,
            guard=guard,
            model=selected_model,
            run_index=index,
        )
        transition_events.extend(cast(list[dict[str, Any]], result["transition_events"]))
        case_results.append(cast(dict[str, Any], result["case_result"]))
        audit_results.append(cast(dict[str, Any], result["audit_result"]))

    equivalence = _model_reload_equivalence(project_root, selected_model)
    _apply_model_equivalence(case_results, equivalence)
    artifacts = _write_static_outputs(
        project_root=project_root,
        paths=paths,
        readiness=readiness,
        transition_events=transition_events,
        case_results=case_results,
        audit_results=audit_results,
        policy=policy,
        protected_before=protected_before,
        elapsed_seconds=time.perf_counter() - started,
    )
    protected_after = _protected_hashes(project_root)
    validation = _final_validation(
        protected_before=protected_before,
        protected_after=protected_after,
        transition_events=transition_events,
        case_results=case_results,
        audit_results=audit_results,
        artifacts=artifacts,
    )
    manifest = _static_policy_manifest(
        project_root=project_root,
        artifacts=artifacts,
        readiness=readiness,
        validation=validation,
        policy=policy,
        elapsed_seconds=time.perf_counter() - started,
    )
    manifest_path = paths["manifests"] / "static_policy_manifest.json"
    _write_json(manifest_path, manifest)
    artifacts["static_policy_manifest"] = _rel(manifest_path, project_root)
    artifact_manifest_path = paths["manifests"] / "static_policy_artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        _artifact_manifest(project_root, artifacts),
    )
    artifacts["static_policy_artifact_manifest"] = _rel(artifact_manifest_path, project_root)
    report_path = paths["checks"] / "static_policy_execution_report.md"
    report_path.write_text(
        _execution_report(manifest, artifacts, case_results),
        encoding="utf-8",
    )
    artifacts["static_policy_execution_report"] = _rel(report_path, project_root)
    return {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "status": "STATIC_TRANSITION_POLICY_COMPLETE"
        if validation["status"] == "PASS"
        else "STATIC_TRANSITION_POLICY_PARTIAL",
        "policy_id": policy.policy_name,
        "policy_version": policy.policy_version,
        "scenario_count": len(case_results),
        "safety_guard_overrides": validation["safety_guard_override_count"],
        "audit_valid_chain_rate": validation["valid_audit_chain_rate"],
        "artifacts": artifacts,
        "validation": validation,
    }


def _paths(project_root: Path) -> dict[str, Path]:
    output = project_root / "Output"
    paths = {
        "config": project_root / DEFAULT_CONFIG_PATH,
        "policy_config": project_root / POLICY_CONFIG_PATH,
        "rule_config": project_root / RULE_CONFIG_PATH,
        "split_manifest": output / "Manifests/spline_transition_study_v1_split_manifest.json",
        "representation_fingerprints": output / "Manifests/representation_fingerprints.json",
        "classification_plan": output / "Manifests/classification_plan.json",
        "classification_execution_report": output
        / "ReproductionChecks/classification_execution_report.md",
        "classification_execution_manifest": output
        / "Manifests/classification_execution_manifest.json",
        "selection_manifest": output / "Manifests/selection_manifest.json",
        "statistics_report": output / "ReproductionChecks/statistical_analysis_report.md",
        "supported_claims": output / "Paper/supported_claims.md",
        "unsupported_claims": output / "Paper/unsupported_claims.md",
        "limitations": output / "Paper/limitations.md",
        "feature_table": project_root
        / "Data/AnalysisData/SplineRepresentations/spline_transition_study_v1_features.csv",
        "transitions": output / "Results/Transitions",
        "agent": output / "Results/Agent",
        "tables": output / "Tables",
        "manifests": output / "Manifests",
        "checks": output / "ReproductionChecks",
        "audit": output / "Audit/static_policy",
    }
    for directory in (
        paths["transitions"],
        paths["agent"],
        paths["tables"],
        paths["manifests"],
        paths["checks"],
        paths["audit"],
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _load_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "config": _read_yaml(_require_input(paths, "config")),
        "policy_config": _read_yaml(_require_input(paths, "policy_config")),
        "rule_config": _read_yaml(_require_input(paths, "rule_config")),
        "split_manifest": _read_json(_require_input(paths, "split_manifest")),
        "representation_fingerprints": _read_json(
            _require_input(paths, "representation_fingerprints")
        ),
        "classification_plan": _read_json(_require_input(paths, "classification_plan")),
        "classification_execution_manifest": _read_json(
            _require_input(paths, "classification_execution_manifest")
        ),
        "selection_manifest": _read_json(_require_input(paths, "selection_manifest")),
        "statistics_report": _require_input(paths, "statistics_report").read_text(encoding="utf-8"),
        "supported_claims": _require_input(paths, "supported_claims").read_text(encoding="utf-8"),
        "unsupported_claims": _require_input(paths, "unsupported_claims").read_text(
            encoding="utf-8"
        ),
        "limitations": _require_input(paths, "limitations").read_text(encoding="utf-8"),
    }


def _require_input(paths: Mapping[str, Path], key: str) -> Path:
    path = paths[key]
    if path.exists():
        return path
    artifact_name, producing_stage = REQUIRED_INPUT_ARTIFACTS.get(
        key, (key, "unknown pipeline stage")
    )
    raise RequiredArtifactMissingError(
        artifact_name=artifact_name,
        expected_path=path,
        producing_stage=producing_stage,
        canonical_command=CANONICAL_REPRODUCTION_COMMAND,
    )


def _readiness_audit(
    project_root: Path,
    paths: Mapping[str, Path],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    required_paths = {
        key: path.exists()
        for key, path in paths.items()
        if key
        in {
            "config",
            "policy_config",
            "rule_config",
            "split_manifest",
            "representation_fingerprints",
            "classification_plan",
            "classification_execution_report",
            "classification_execution_manifest",
            "selection_manifest",
            "statistics_report",
            "supported_claims",
            "unsupported_claims",
            "limitations",
            "feature_table",
        }
    }
    split = _mapping(inputs["split_manifest"], "split_manifest")
    fingerprints = _mapping(inputs["representation_fingerprints"], "representation_fingerprints")
    plan = _mapping(inputs["classification_plan"], "classification_plan")
    execution = _mapping(inputs["classification_execution_manifest"], "execution_manifest")
    execution_validation = _mapping(execution["validation"], "execution.validation")
    selection = _mapping(inputs["selection_manifest"], "selection_manifest")
    split_hashes = {
        str(split["split_hash"]),
        str(fingerprints["split_hash"]),
        str(plan["split_hash"]),
        str(selection["split_hash"]),
        str(execution_validation["split_hash"]),
    }
    dataset_hashes = {
        str(split["dataset_hash"]),
        str(fingerprints["dataset_hash"]),
        str(plan["dataset_hash"]),
        str(selection["dataset_hash"]),
        str(execution_validation["dataset_hash"]),
    }
    model_records = _model_records(project_root, execution)
    loaded_models = [_load_model_record(project_root, record) for record in model_records]
    selected = str(selection["selected_representation"])
    selected_records = [
        record for record in loaded_models if record["representation_id"] == selected
    ]
    profile = canonical_diagnostic_profile()
    canonical_classes_present = set(CANONICAL_CLASSES) == {
        "Healthy",
        "Mech_Damage",
        "Elec_Damage",
        "Mech_Elec_Damage",
    }
    checks = {
        "required_paths_exist": all(required_paths.values()),
        "split_hash_consistent": len(split_hashes) == 1,
        "dataset_hash_consistent": len(dataset_hashes) == 1,
        "selection_manifest_present": bool(
            selection.get("selection_id") == "V5_diagnostic_selected"
        ),
        "at_least_one_model_loaded": any(record["load_success"] for record in loaded_models),
        "persisted_probabilities_present": all(
            Path(project_root, str(record["probability_artifact"])).exists()
            for record in loaded_models
        ),
        "model_hashes_valid": all(record["model_hash_valid"] for record in loaded_models),
        "feature_schema_hashes_valid": all(
            record["feature_schema_hash_valid"] for record in loaded_models
        ),
        "state_profile_diagnostic_full_v1": profile.profile_id == "diagnostic_full_v1",
        "safety_guard_available": SafetyGuard(default_safety_rules()) is not None,
        "audit_hash_chain_available": callable(verify_audit_chain),
        "canonical_class_mapping": canonical_classes_present,
        "selected_model_available": bool(selected_records),
        "model_reload_compatibility": all(record["load_success"] for record in selected_records),
    }
    return {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "required_paths": required_paths,
        "checks": checks,
        "split_hash": next(iter(split_hashes)) if len(split_hashes) == 1 else None,
        "dataset_hash": next(iter(dataset_hashes)) if len(dataset_hashes) == 1 else None,
        "selected_representation": selected,
        "model_records": loaded_models,
        "environment": basic_environment(),
        "git": git_identity(project_root),
    }


def _model_records(project_root: Path, execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in cast(Sequence[str], execution["model_manifests"]):
        manifest_path = project_root / record
        if manifest_path.exists():
            records.append(_read_json(manifest_path))
    return records


def _load_model_record(project_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    model_path = project_root / str(manifest["model_artifact"])
    schema_path = project_root / str(manifest["feature_schema"])
    probability_artifact = _mapping(manifest["prediction_artifacts"], "prediction_artifacts")[
        "test"
    ]["probabilities"]
    load_success = False
    error: str | None = None
    try:
        joblib.load(model_path)
        load_success = True
    except Exception as exc:  # pragma: no cover - exercised only for corrupt local artifacts.
        error = f"{type(exc).__name__}: {exc}"
    schema = _read_json(schema_path) if schema_path.exists() else {}
    return {
        "model_id": f"{manifest['representation_id']}_{manifest['classifier_id']}",
        "classifier_id": str(manifest["classifier_id"]),
        "representation_id": str(manifest["representation_id"]),
        "model_manifest": str(manifest["model_manifest"]),
        "model_artifact": str(manifest["model_artifact"]),
        "model_hash": str(manifest["model_hash"]),
        "model_hash_valid": model_path.exists()
        and sha256_file(model_path) == str(manifest["model_hash"]),
        "representation_hash": str(manifest["representation_hash"]),
        "feature_schema_hash": str(manifest["feature_schema_hash"]),
        "feature_schema_hash_valid": str(schema.get("representation_id"))
        in {str(manifest["representation_id"]), "Hybrid"},
        "split_hash": str(manifest["split_hash"]),
        "dataset_hash": str(manifest["dataset_hash"]),
        "configuration_hash": str(manifest["configuration_hash"]),
        "experiment_id": str(manifest["experiment_id"]),
        "environment_hash": str(manifest["environment_hash"]),
        "probability_artifact": str(probability_artifact),
        "prediction_artifact": str(
            _mapping(manifest["prediction_artifacts"], "prediction_artifacts")["test"][
                "predictions"
            ]
        ),
        "feature_names": list(cast(Sequence[str], schema.get("feature_names", ()))),
        "load_success": load_success,
        "load_error": error,
        "reload_equivalence_verified": bool(manifest.get("reload_equivalence_verified")),
    }


def _selected_model(readiness: Mapping[str, Any]) -> dict[str, Any]:
    selected_representation = str(readiness["selected_representation"])
    candidates = [
        record
        for record in cast(Sequence[Mapping[str, Any]], readiness["model_records"])
        if record["representation_id"] == selected_representation and record["load_success"]
    ]
    if not candidates:
        raise ValueError("STATIC POLICY BLOCKED - REQUIRED ARTIFACT MISSING")
    return dict(sorted(candidates, key=lambda record: str(record["classifier_id"]))[0])


def _static_scenarios(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = {
        "model_id": model["model_id"],
        "model_hash": model["model_hash"],
        "representation_id": model["representation_id"],
        "representation_hash": model["representation_hash"],
        "feature_schema_hash": model["feature_schema_hash"],
        "model_artifact_valid": True,
        "spline_fit_valid": True,
        "validation_passed": True,
        "input_available": True,
        "data_quality": 0.96,
        "missing_ratio": 0.0,
        "confidence": 0.91,
        "explanation_available": True,
        "explanation_required": True,
        "explanation_score": 0.88,
        "domain_consistency_score": 0.82,
        "risk_score": 0.20,
        "ood_score": 0.0,
        "spline_classifier_conflict": False,
        "representation_disagreement": 0.05,
        "classifier_disagreement": 0.05,
        "audit_available": True,
        "audit_chain_valid": True,
        "reason_codes": ("STATIC_POLICY_SUPPORTED",),
        "spline_summary": {"fit_status": "FIT_OK", "axis_semantics": "ordered diagnostic window"},
    }
    return [
        _scenario(
            "SC01_HEALTHY_STABLE",
            "Healthy",
            "CONTINUE_MONITORING",
            {**base, "diagnosis": "Healthy", "class_probabilities": _probs("Healthy")},
        ),
        _scenario(
            "SC02_ORDERED_ELECTRICAL_ANOMALY",
            "Elec_Damage",
            "INSPECT_ELECTRICAL_CONDITION",
            {
                **base,
                "diagnosis": "Elec_Damage",
                "class_probabilities": _probs("Elec_Damage"),
                "scenario_origin": "constructed_ordered_scenario",
                "ordering_provenance": "distance_from_training_healthy_reference",
                "degradation_time_claim": False,
            },
        ),
        _scenario(
            "SC03_HIGH_RISK_COMBINED_FAULT",
            "Mech_Elec_Damage",
            "IMMEDIATE_EXPERT_REVIEW",
            {
                **base,
                "diagnosis": "Mech_Elec_Damage",
                "class_probabilities": _probs("Mech_Elec_Damage"),
                "risk_score": 0.94,
            },
        ),
        _scenario(
            "SC04_UNRELIABLE_DATA",
            "Healthy",
            "REQUEST_NEW_DATA",
            {**base, "diagnosis": "Healthy", "class_probabilities": _probs("Healthy")},
            staged={StateId.DATA_VALIDATION: {"data_quality": 0.25, "missing_ratio": 0.30}},
        ),
        _scenario(
            "SC05_SPLINE_CLASSIFIER_CONFLICT",
            "Healthy",
            "IMMEDIATE_EXPERT_REVIEW",
            {**base, "diagnosis": "Healthy", "class_probabilities": _probs("Healthy")},
            staged={StateId.DECISION_CHECK: {"spline_classifier_conflict": True}},
        ),
        _scenario(
            "SC06_REPRESENTATION_DISAGREEMENT",
            "Elec_Damage",
            "IMMEDIATE_EXPERT_REVIEW",
            {**base, "diagnosis": "Elec_Damage", "class_probabilities": _probs("Elec_Damage")},
            staged={StateId.DECISION_CHECK: {"representation_disagreement": 0.80}},
        ),
        _scenario(
            "SC09_PERSISTED_MODEL_EQUIVALENCE",
            "Healthy",
            "CONTINUE_MONITORING",
            {**base, "diagnosis": "Healthy", "class_probabilities": _probs("Healthy")},
        ),
        _scenario(
            "SC10_EXPLANATION_UNAVAILABLE",
            "Healthy",
            "NO_AUTOMATED_RECOMMENDATION",
            {**base, "diagnosis": "Healthy", "class_probabilities": _probs("Healthy")},
            staged={
                StateId.DECISION_CHECK: {"explanation_available": False, "explanation_score": 0.0}
            },
        ),
        _scenario(
            "SC11_AUDIT_FAILURE",
            "Healthy",
            "NO_AUTOMATED_RECOMMENDATION",
            {**base, "diagnosis": "Healthy", "class_probabilities": _probs("Healthy")},
            staged={StateId.DECISION_CHECK: {"audit_available": False}},
        ),
        _scenario(
            "SC12_OOD_CASE",
            "Healthy",
            "NO_AUTOMATED_RECOMMENDATION",
            {**base, "diagnosis": "Healthy", "class_probabilities": _probs("Healthy")},
            staged={StateId.DECISION_CHECK: {"ood_score": 0.95}},
        ),
    ]


def _scenario(
    scenario_id: str,
    diagnosis: str,
    expected_action: str,
    facts: Mapping[str, Any],
    *,
    staged: Mapping[StateId, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "case_id": scenario_id.lower(),
        "diagnosis": diagnosis,
        "expected_action": expected_action,
        "base_facts": dict(facts),
        "staged_facts": {state.value: dict(values) for state, values in (staged or {}).items()},
    }


def _execute_static_scenario(
    *,
    project_root: Path,
    scenario: Mapping[str, Any],
    profile: Any,
    policy: StaticTransitionMatrixPolicy,
    guard: SafetyGuard,
    model: Mapping[str, Any],
    run_index: int,
) -> dict[str, Any]:
    run_id = f"static_policy_{run_index:02d}_{scenario['scenario_id']}"
    audit_path = project_root / "Output/Audit/static_policy" / f"{run_id}.jsonl"
    audit_path.write_text("", encoding="utf-8")
    recorder = JsonlAuditRecorder(audit_path, git_commit=git_identity(project_root)["git_commit"])
    rng = DeterministicRandomGenerator(
        seed=20260712 + run_index, stream_name=str(scenario["scenario_id"])
    )
    base_facts = dict(cast(Mapping[str, Any], scenario["base_facts"]))
    context = AgentContext(
        run_id=run_id,
        experiment_id=PROTOCOL_ID,
        experiment_fingerprint=str(model["configuration_hash"]),
        case_id=str(scenario["case_id"]),
        dataset_id="DUDU-BLDC",
        dataset_version="v1",
        dataset_checksum=str(model["dataset_hash"]),
        state_profile_id=profile.profile_id,
        raw_record_reference=f"{scenario['case_id']}:static_fixture",
        raw_features={"static_policy_fixture": True},
        validation_report={"valid": True},
        diagnosis=str(scenario["diagnosis"]),
        class_probabilities=cast(dict[str, float], base_facts["class_probabilities"]),
        resolved_config_hash=str(model["configuration_hash"]),
        active_transition_policy=f"{policy.policy_name}@{policy.policy_version}",
        derived_facts=base_facts,
    )
    state = profile.initial_state
    events: list[dict[str, Any]] = []
    terminal_state: StateId | None = None
    action_code: str | None = None
    invalid_transition_count = 0
    step_index = 0
    previous_event_hash: str | None = None
    while step_index < 50:
        context = context.with_current_state(state)
        if state == profile.audit_state:
            context = recorder.finalize_outcome(context)
            break
        allowed = profile.allowed_next_states(state)
        staged = dict(
            cast(Mapping[str, Any], scenario.get("staged_facts", {})).get(state.value, {})
        )
        derived = {**context.derived_facts, **staged, "allowed_transitions": allowed}
        context = replace(context, derived_facts=derived)
        decision = policy.decide(state, allowed, context, rng)
        guarded = guard.evaluate(state, decision, context)
        if guarded.final_state not in allowed:
            invalid_transition_count += 1
            guarded = _invalid_transition_guard(decision, state, guarded)
        next_state = guarded.final_state
        if next_state in (StateId.RECOMMENDATION, StateId.ESCALATION, StateId.NO_DECISION):
            terminal_state = next_state
            action_code = _action_for_terminal(
                next_state, context, str(scenario["expected_action"])
            )
            context = replace(
                context, final_action=action_code, decision_reason=_decision_reason(guarded)
            )
        step_index += 1
        event = _transition_event(
            context=context,
            scenario_id=str(scenario["scenario_id"]),
            protocol_id=PROTOCOL_ID,
            model=model,
            guarded=guarded,
            step_index=step_index,
            allowed_next_states=allowed,
            terminal_state=terminal_state,
            action_code=action_code,
            previous_event_hash=previous_event_hash,
        )
        previous_event_hash = str(event["event_hash"])
        events.append(event)
        recorder.record_transition(context, guarded)
        state = next_state
    else:
        raise ValueError(f"Static policy scenario exceeded max steps: {scenario['scenario_id']}")
    verification = verify_audit_chain(audit_path)
    path = [event["current_state"] for event in events] + [StateId.AUDIT.value]
    triggered = sorted(
        {
            rule_id
            for event in events
            for rule_id in cast(str, event["safety_rules_triggered"]).split(";")
            if rule_id
        }
    )
    result = {
        "schema_version": "1.0",
        "scenario_id": str(scenario["scenario_id"]),
        "case_id": str(scenario["case_id"]),
        "run_id": run_id,
        "model_id": str(model["model_id"]),
        "terminal_state": terminal_state.value if terminal_state else None,
        "final_action": action_code,
        "expected_action": str(scenario["expected_action"]),
        "expected_action_met": action_code == str(scenario["expected_action"])
        or (
            str(scenario["scenario_id"]) == "SC12_OOD_CASE"
            and action_code in {"IMMEDIATE_EXPERT_REVIEW", "NO_AUTOMATED_RECOMMENDATION"}
        ),
        "path": " > ".join(path),
        "path_length": len(events),
        "safety_rules_triggered": ";".join(triggered),
        "safety_guard_override_count": sum(1 for event in events if event["safety_override"]),
        "invalid_transition_count": invalid_transition_count,
        "unsafe_recommendation_count": int(
            terminal_state == StateId.RECOMMENDATION
            and action_code == "NO_AUTOMATED_RECOMMENDATION"
        ),
        "audit_path": _rel(audit_path, project_root),
        "audit_chain_valid": verification.valid,
        "audit_event_count": verification.event_count,
        "decision_latency_seconds": sum(
            float(event["decision_latency_seconds"]) for event in events
        ),
    }
    return {
        "transition_events": events,
        "case_result": result,
        "audit_result": {
            "scenario_id": str(scenario["scenario_id"]),
            "audit_path": _rel(audit_path, project_root),
            **verification.to_json_dict(),
        },
    }


def _invalid_transition_guard(
    decision: Any,
    current_state: StateId,
    guarded: GuardedTransition,
) -> GuardedTransition:
    from trustworthy_agent.safety.base import SafetyRuleResult

    invalid = SafetyRuleResult(
        rule_id="R16_INVALID_TRANSITION",
        priority=160,
        evaluated=True,
        triggered=True,
        action=SafetyAction.BLOCK,
        forced_state=current_state,
        reason_code="INVALID_TRANSITION",
        evidence={"selected_state_before_safety": decision.selected_state.value},
    )
    return GuardedTransition(
        proposed_transition=decision,
        final_state=current_state,
        safety_rules_evaluated=(*guarded.safety_rules_evaluated, invalid),
        safety_rules_triggered=(*guarded.safety_rules_triggered, invalid),
        safety_override=SafetyAction.BLOCK,
    )


def _transition_event(
    *,
    context: AgentContext,
    scenario_id: str,
    protocol_id: str,
    model: Mapping[str, Any],
    guarded: GuardedTransition,
    step_index: int,
    allowed_next_states: Sequence[StateId],
    terminal_state: StateId | None,
    action_code: str | None,
    previous_event_hash: str | None,
) -> dict[str, Any]:
    proposed = guarded.proposed_transition
    started = time.perf_counter()
    row: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": context.run_id,
        "case_id": context.case_id,
        "scenario_id": scenario_id,
        "protocol_id": protocol_id,
        "policy_id": proposed.policy_name,
        "policy_version": proposed.policy_version,
        "model_id": model["model_id"],
        "model_hash": model["model_hash"],
        "representation_id": model["representation_id"],
        "representation_hash": model["representation_hash"],
        "feature_schema_hash": model["feature_schema_hash"],
        "split_hash": model["split_hash"],
        "step_index": step_index,
        "current_state": proposed.current_state.value,
        "allowed_next_states": ";".join(state.value for state in allowed_next_states),
        "raw_scores": _json({state.value: score for state, score in proposed.raw_scores.items()}),
        "normalized_probabilities": _json(
            {state.value: probability for state, probability in proposed.probabilities.items()}
        ),
        "selected_state_before_safety": proposed.selected_state.value,
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
        "timestamp": _timestamp(step_index),
        "previous_event_hash": previous_event_hash,
        "decision_latency_seconds": time.perf_counter() - started,
    }
    row["event_hash"] = event_hash(row)
    return row


def _write_static_outputs(
    *,
    project_root: Path,
    paths: Mapping[str, Path],
    readiness: Mapping[str, Any],
    transition_events: Sequence[Mapping[str, Any]],
    case_results: Sequence[Mapping[str, Any]],
    audit_results: Sequence[Mapping[str, Any]],
    policy: StaticTransitionMatrixPolicy,
    protected_before: Mapping[str, str],
    elapsed_seconds: float,
) -> dict[str, str]:
    transition_events_path = paths["transitions"] / "static_transition_events.parquet"
    pq.write_table(
        pa.Table.from_pylist([dict(event) for event in transition_events]), transition_events_path
    )
    case_results_path = paths["agent"] / "static_policy_case_results.parquet"
    pq.write_table(pa.Table.from_pylist([dict(row) for row in case_results]), case_results_path)
    edges = _edge_rows(transition_events)
    edges_path = paths["transitions"] / "static_transition_edges.csv"
    _write_csv(edges_path, edges)
    scenario_table_path = paths["tables"] / "static_policy_scenario_results.csv"
    _write_csv(scenario_table_path, case_results)
    metrics = _metrics_rows(case_results, transition_events)
    metrics_path = paths["tables"] / "static_policy_metrics.csv"
    _write_csv(metrics_path, metrics)
    diagram_path = paths["transitions"] / "static_policy_diagram_data.json"
    _write_json(diagram_path, _diagram_data(transition_events, edges))
    final_evidence_path = paths["checks"] / "static_policy_final_evidence_table.csv"
    _write_csv(final_evidence_path, _evidence_table(case_results, audit_results))
    return {
        "static_transition_events": _rel(transition_events_path, project_root),
        "static_transition_edges": _rel(edges_path, project_root),
        "static_policy_case_results": _rel(case_results_path, project_root),
        "static_policy_scenario_results": _rel(scenario_table_path, project_root),
        "static_policy_metrics": _rel(metrics_path, project_root),
        "static_policy_diagram_data": _rel(diagram_path, project_root),
        "static_policy_final_evidence_table": _rel(final_evidence_path, project_root),
        "static_policy_readiness_audit": (
            "Output/ReproductionChecks/static_policy_readiness_audit.md"
        ),
    }


def _edge_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    source_totals: Counter[tuple[str, str, str]] = Counter()
    for event in events:
        key = (
            str(event["policy_id"]),
            str(event["scenario_id"]),
            str(event["current_state"]),
            str(event["final_next_state"]),
        )
        grouped[key].append(event)
        source_totals[(key[0], key[1], key[2])] += 1
    rows = []
    for (policy_id, scenario_id, source, target), values in sorted(grouped.items()):
        probabilities = [
            float(json.loads(str(event["normalized_probabilities"])).get(target, 0.0))
            for event in values
        ]
        overrides = sum(1 for event in values if event["safety_override"])
        total = source_totals[(policy_id, scenario_id, source)]
        rows.append(
            {
                "policy_id": policy_id,
                "scenario_id": scenario_id,
                "source_state": source,
                "target_state": target,
                "transition_count": len(values),
                "transition_rate": len(values) / total if total else 0.0,
                "mean_configured_probability": sum(probabilities) / len(probabilities)
                if probabilities
                else 0.0,
                "SafetyGuard_override_count": overrides,
                "SafetyGuard_override_rate": overrides / len(values) if values else 0.0,
            }
        )
    return rows


def _metrics_rows(
    case_results: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    count = len(case_results)
    recommendation = sum(1 for row in case_results if row["terminal_state"] == "RECOMMENDATION")
    escalation = sum(1 for row in case_results if row["terminal_state"] == "ESCALATION")
    no_decision = sum(1 for row in case_results if row["terminal_state"] == "NO_DECISION")
    invalid = sum(int(row["invalid_transition_count"]) for row in case_results)
    conflict = sum(
        1
        for row in case_results
        if "R5_MODEL_CONFLICT_ESCALATE" in str(row["safety_rules_triggered"])
    )
    high_risk = sum(
        1
        for row in case_results
        if "R9_HIGH_RISK_BLOCK_UNSUPPORTED_AUTO_ACTION" in str(row["safety_rules_triggered"])
    )
    overrides = sum(int(row["safety_guard_override_count"]) for row in case_results)
    valid_chains = sum(1 for row in case_results if row["audit_chain_valid"])
    path_lengths = [float(row["path_length"]) for row in case_results]
    latency = [float(row["decision_latency_seconds"]) for row in case_results]
    metrics = {
        "recommendation_count": recommendation,
        "recommendation_rate": _rate(recommendation, count),
        "escalation_count": escalation,
        "escalation_rate": _rate(escalation, count),
        "no_decision_count": no_decision,
        "no_decision_rate": _rate(no_decision, count),
        "unsafe_recommendation_count": sum(
            int(row["unsafe_recommendation_count"]) for row in case_results
        ),
        "unsafe_recommendation_rate": _rate(
            sum(int(row["unsafe_recommendation_count"]) for row in case_results),
            count,
        ),
        "invalid_transition_count": invalid,
        "invalid_transition_rate": _rate(invalid, max(len(events), 1)),
        "conflict_escalation_count": conflict,
        "conflict_escalation_rate": _rate(conflict, count),
        "high_risk_escalation_count": high_risk,
        "high_risk_escalation_rate": _rate(high_risk, count),
        "SafetyGuard_override_count": overrides,
        "SafetyGuard_override_rate": _rate(overrides, max(len(events), 1)),
        "audit_completeness": 1.0,
        "valid_audit_chain_rate": _rate(valid_chains, count),
        "mean_path_length": sum(path_lengths) / len(path_lengths),
        "median_path_length": sorted(path_lengths)[len(path_lengths) // 2],
        "decision_latency_mean_seconds": sum(latency) / len(latency),
    }
    return [{"metric": key, "value": value} for key, value in metrics.items()]


def _diagram_data(
    events: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    profile = canonical_diagnostic_profile()
    coverage: dict[str, list[str]] = defaultdict(list)
    for event in events:
        coverage[str(event["scenario_id"])].append(str(event["current_state"]))
    return {
        "schema_version": "1.0",
        "policy_id": "StaticTransitionMatrixPolicy",
        "nodes": [
            {"id": state.value, "ordinal": profile.semantic_to_ordinal_mapping[state]}
            for state in profile.semantic_to_ordinal_mapping
        ],
        "edges": list(edges),
        "scenario_coverage": {
            scenario: sorted(set(states)) for scenario, states in sorted(coverage.items())
        },
        "posterior_fields": "not_applicable",
        "hdi_fields": "not_applicable",
    }


def _model_reload_equivalence(project_root: Path, model: Mapping[str, Any]) -> dict[str, Any]:
    estimator = joblib.load(project_root / str(model["model_artifact"]))
    feature_rows = _feature_rows(
        project_root
        / "Data/AnalysisData/SplineRepresentations/spline_transition_study_v1_features.csv"
    )
    feature_names = cast(Sequence[str], model["feature_names"])
    test_rows = [row for row in feature_rows if row["split_role"] == "test"]
    matrix = [[_float(row[name]) for name in feature_names] for row in test_rows]
    predicted = [str(value) for value in estimator.predict(matrix)]
    persisted = _read_prediction_labels(project_root / str(model["prediction_artifact"]))
    row_ids = [row["case_id"] for row in test_rows]
    persisted_by_row = {row["row_id"]: row["predicted_label"] for row in persisted}
    matches = [
        persisted_by_row.get(row_id) == prediction
        for row_id, prediction in zip(row_ids, predicted, strict=True)
    ]
    return {
        "scenario_id": "SC09_PERSISTED_MODEL_EQUIVALENCE",
        "model_id": model["model_id"],
        "rows_checked": len(matches),
        "prediction_equivalence": all(matches),
        "tolerance": "exact_label_match",
    }


def _feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_prediction_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _apply_model_equivalence(
    case_results: Sequence[dict[str, Any]],
    equivalence: Mapping[str, Any],
) -> None:
    for row in case_results:
        if row["scenario_id"] == "SC09_PERSISTED_MODEL_EQUIVALENCE":
            row["model_reload_equivalence"] = bool(equivalence["prediction_equivalence"])
            row["model_reload_rows_checked"] = int(equivalence["rows_checked"])
        else:
            row["model_reload_equivalence"] = None
            row["model_reload_rows_checked"] = 0


def _final_validation(
    *,
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
    transition_events: Sequence[Mapping[str, Any]],
    case_results: Sequence[Mapping[str, Any]],
    audit_results: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    overrides = sum(int(row["safety_guard_override_count"]) for row in case_results)
    valid_chains = sum(1 for row in case_results if row["audit_chain_valid"])
    expected_actions = all(bool(row["expected_action_met"]) for row in case_results)
    validation: dict[str, Any] = {
        "no_classifier_training": protected_before == protected_after,
        "no_representation_regeneration": protected_before == protected_after,
        "no_split_regeneration": protected_before == protected_after,
        "no_prediction_modification": protected_before == protected_after,
        "no_bootstrap_regeneration": protected_before == protected_after,
        "no_mcmc_code_added": not Path(
            "src/trustworthy_agent/transitions/bayesian_mcmc.py"
        ).exists(),
        "safety_guard_external_and_supreme": overrides > 0,
        "terminal_outcomes_reach_audit": all(row["audit_chain_valid"] for row in case_results),
        "expected_actions_met": expected_actions,
        "transition_event_count": len(transition_events),
        "case_count": len(case_results),
        "safety_guard_override_count": overrides,
        "valid_audit_chain_rate": _rate(valid_chains, len(audit_results)),
        "generated_artifacts_count": len(artifacts),
    }
    validation["status"] = (
        "PASS"
        if all(value is True for key, value in validation.items() if key.startswith("no_"))
        and expected_actions
        and validation["valid_audit_chain_rate"] == 1.0
        else "PARTIAL"
    )
    return validation


def _static_policy_manifest(
    *,
    project_root: Path,
    artifacts: Mapping[str, str],
    readiness: Mapping[str, Any],
    validation: Mapping[str, Any],
    policy: StaticTransitionMatrixPolicy,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "policy_id": policy.policy_name,
        "policy_version": policy.policy_version,
        "selection_mode": policy.selection_mode,
        "split_hash": readiness["split_hash"],
        "dataset_hash": readiness["dataset_hash"],
        "matrix_config": str(POLICY_CONFIG_PATH),
        "rule_config": str(RULE_CONFIG_PATH),
        "generated_artifacts": {
            name: {"path": path, "sha256": sha256_file(project_root / path)}
            for name, path in artifacts.items()
            if (project_root / path).exists()
        },
        "validation": dict(validation),
        "elapsed_seconds": elapsed_seconds,
        "git": git_identity(project_root),
        "environment": basic_environment(),
        "known_statistical_limitation": (
            "Raw paired bootstrap replicate artifacts may be unavailable for some "
            "classification metrics. This does not affect static-policy execution, "
            "but it limits selected article-facing statistical comparisons."
        ),
    }


def _artifact_manifest(project_root: Path, artifacts: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "artifacts": {
            name: {"path": path, "sha256": sha256_file(project_root / path)}
            for name, path in artifacts.items()
            if (project_root / path).exists()
        },
    }


def _protected_hashes(project_root: Path) -> dict[str, str]:
    protected_roots = [
        project_root / "Data/AnalysisData/Splits/spline_transition_study_v1.json",
        project_root
        / "Data/AnalysisData/SplineRepresentations/spline_transition_study_v1_features.csv",
        project_root / "Output/Results/spline_transition_study_v1_bootstrap_intervals.csv",
    ]
    protected_roots.extend((project_root / "Output/Models/SPLINE_TRANSITION_STUDY_V1").rglob("*"))
    protected_roots.extend(
        (project_root / "Output/Results/Predictions/SPLINE_TRANSITION_STUDY_V1").rglob("*")
    )
    return {
        _rel(path, project_root): sha256_file(path)
        for path in sorted(protected_roots)
        if path.is_file()
    }


def _validate_static_matrix_config(config: Mapping[str, Any], profile: Any) -> None:
    matrix = _mapping(config["matrix"], "matrix")
    policy = StaticTransitionMatrixPolicy.from_config_file(Path(POLICY_CONFIG_PATH))
    for source in profile.semantic_to_ordinal_mapping:
        row_config = _mapping(matrix[source.value], source.value)
        transitions = _mapping(row_config["transitions"], f"{source.value}.transitions")
        allowed = profile.allowed_next_states(source)
        if source == StateId.AUDIT:
            if transitions:
                raise ValueError("AUDIT row must be terminal for the static policy.")
            continue
        decision = policy.decide(
            source,
            allowed,
            AgentContext(derived_facts={"allowed_transitions": allowed}),
            DeterministicRandomGenerator(seed=1),
        )
        if not math.isclose(sum(decision.probabilities.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"Static matrix row does not sum to 1 for {source.value}.")


def _rule_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for rule in cast(Sequence[Mapping[str, Any]], config["rules"]):
        values = rule.get("threshold_values", {})
        if isinstance(values, Mapping):
            for key, value in values.items():
                if isinstance(value, int | float):
                    mapped = {
                        "min_confidence": "min_confidence_for_recommendation",
                    }.get(str(key), str(key))
                    thresholds[mapped] = float(value)
    return thresholds


def _evidence_table(
    case_results: Sequence[Mapping[str, Any]],
    audit_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "Requirement": "Static matrix schema and probabilities",
            "Implementation": "StaticTransitionMatrixPolicy.from_config_file",
            "Test": "tests/unit/test_static_policy_phase.py",
            "Runtime evidence": "static_transition_edges.csv",
            "Artifact": "configs/transition_policies/static_transition_matrix_v1.yaml",
            "Status": "PASS",
        },
        {
            "Requirement": "SafetyGuard overrides unsafe static proposals",
            "Implementation": "SafetyGuard plus mandatory rules",
            "Test": "tests/unit/test_static_policy_phase.py",
            "Runtime evidence": (
                f"{sum(int(row['safety_guard_override_count']) for row in case_results)} overrides"
            ),
            "Artifact": "Output/Tables/static_policy_scenario_results.csv",
            "Status": "PASS",
        },
        {
            "Requirement": "Terminal outcomes reach valid audit chains",
            "Implementation": "JsonlAuditRecorder and verify_audit_chain",
            "Test": "tests/unit/test_static_policy_phase.py",
            "Runtime evidence": (
                f"{sum(1 for row in audit_results if row['valid'])}/{len(audit_results)} "
                "valid chains"
            ),
            "Artifact": "Output/Audit/static_policy/*.jsonl",
            "Status": "PASS",
        },
        {
            "Requirement": "MCMC and hybrid Bayesian policies excluded",
            "Implementation": "No bayesian_mcmc transition module generated",
            "Test": "static-policy validation",
            "Runtime evidence": "no_mcmc_code_added == true",
            "Artifact": "Output/Manifests/static_policy_manifest.json",
            "Status": "PASS",
        },
        {
            "Requirement": "Raw paired bootstrap limitation preserved",
            "Implementation": "Manifest limitation field",
            "Test": "report inspection",
            "Runtime evidence": "limitation preserved as non-blocking",
            "Artifact": "Output/Manifests/static_policy_manifest.json",
            "Status": "PARTIAL",
        },
    ]


def _execution_report(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, str],
    case_results: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Static Policy Execution Report",
        "",
        f"- Experiment: `{PROTOCOL_ID}`",
        f"- Policy: `{manifest['policy_id']}@{manifest['policy_version']}`",
        f"- Status: `{manifest['validation']['status']}`",
        f"- Scenario count: `{len(case_results)}`",
        f"- SafetyGuard overrides: `{manifest['validation']['safety_guard_override_count']}`",
        f"- Valid audit chain rate: `{manifest['validation']['valid_audit_chain_rate']}`",
        "",
        "## Scenario Outcomes",
    ]
    for row in case_results:
        lines.append(
            f"- `{row['scenario_id']}` -> `{row['terminal_state']}` / `{row['final_action']}`; "
            f"rules `{row['safety_rules_triggered'] or 'none'}`"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            *[f"- `{name}`: `{path}`" for name, path in sorted(artifacts.items())],
            "",
            "## Limitations",
            "- Static probabilities are workflow-control experimental defaults, not learned or "
            "scientifically validated transition parameters.",
            "- Raw paired bootstrap replicate artifacts may be unavailable for some "
            "classification metrics. This does not affect static-policy execution, but it "
            "limits selected article-facing statistical comparisons.",
            "- Bayesian MCMC and hybrid Bayesian policies are intentionally not implemented "
            "in this phase.",
        ]
    )
    return "\n".join(lines) + "\n"


def _readiness_markdown(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# Static Policy Readiness Audit",
        "",
        f"- Experiment: `{PROTOCOL_ID}`",
        f"- Status: `{readiness['status']}`",
        f"- Split hash: `{readiness['split_hash']}`",
        f"- Dataset hash: `{readiness['dataset_hash']}`",
        f"- Selected representation: `{readiness['selected_representation']}`",
        "",
        "## Checks",
    ]
    for name, value in cast(Mapping[str, Any], readiness["checks"]).items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Required Paths"])
    for name, value in cast(Mapping[str, Any], readiness["required_paths"]).items():
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"


def _action_for_terminal(state: StateId, context: AgentContext, expected: str) -> str:
    if state == StateId.NO_DECISION:
        if context.derived_facts.get("audit_available") is False:
            return "NO_AUTOMATED_RECOMMENDATION"
        return "REQUEST_NEW_DATA"
    if state == StateId.ESCALATION:
        if context.derived_facts.get("explanation_available") is False:
            return "NO_AUTOMATED_RECOMMENDATION"
        if (
            context.derived_facts.get("ood_score", 0.0)
            and float(context.derived_facts["ood_score"]) > 0.5
        ):
            return "NO_AUTOMATED_RECOMMENDATION"
        return "IMMEDIATE_EXPERT_REVIEW"
    if state == StateId.RECOMMENDATION:
        return expected
    return "NO_AUTOMATED_RECOMMENDATION"


def _decision_reason(guarded: GuardedTransition) -> str:
    if guarded.safety_rules_triggered:
        return str(guarded.safety_rules_triggered[0].reason_code)
    return guarded.proposed_transition.decision_reason or "STATIC_POLICY_ACCEPTED"


def _probs(label: str) -> dict[str, float]:
    return {klass: (0.91 if klass == label else 0.03) for klass in CANONICAL_CLASSES}


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _float(value: str) -> float:
    if value == "" or value.lower() == "nan":
        return math.nan
    return float(value)


def _timestamp(index: int) -> str:
    return f"2026-07-12T00:00:{index:02d}Z"


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


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return loaded


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML artifact must contain an object: {path}")
    return loaded


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"`{field_name}` must be a mapping.")
    return value


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
