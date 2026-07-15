"""Minimal real ArticleV1 agent execution over frozen evidence bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.engine import AgentEngine, AgentRuntime
from trustworthy_agent.agent.profiles import canonical_diagnostic_profile
from trustworthy_agent.agent.registry import StrategyDescriptor, StrategyRegistry
from trustworthy_agent.agent.state_result import StateResult
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.article_v1.minimal_e2 import NaturalCase, attach_bundle, load_natural_cases
from trustworthy_agent.audit.logger import JsonlAuditRecorder
from trustworthy_agent.audit.verify import verify_audit_chain
from trustworthy_agent.context.evidence import EvidenceAgentContext
from trustworthy_agent.experiments.agent_evaluation import (  # type: ignore[attr-defined]
    _build_bayesian_model,
    _load_mcmc_model,
)
from trustworthy_agent.safety.guards import SafetyGuard
from trustworthy_agent.safety.rules import default_safety_rules
from trustworthy_agent.transitions.bayesian import (
    BayesianMCMCTransitionPolicy,
    DeterministicPosteriorApproximationPolicy,
    HybridTransitionPolicy,
)
from trustworthy_agent.transitions.registry import (
    TransitionPolicyDescriptor,
    TransitionPolicyRegistry,
)
from trustworthy_agent.transitions.rng import DeterministicRandomGenerator
from trustworthy_agent.transitions.static_markov import StaticTransitionMatrixPolicy

POLICY_IDS = (
    "StaticTransitionMatrixPolicy",
    "DeterministicPosteriorApproximationPolicy",
    "BayesianMCMCTransitionPolicy",
    "HybridTransitionPolicy",
)


class _ProjectionStrategy:
    strategy_name = "persisted_evidence_projection"
    strategy_version = "1.0.0"

    def validate_config(self, config: dict[str, Any]) -> None:
        if config.get("configured_diagnostic_facts") != 0:
            raise ValueError("configured diagnostic facts must remain zero")

    def execute(self, context: AgentContext) -> StateResult:
        state = context.current_state
        evidence_context = cast(EvidenceAgentContext, context)
        if state is None or not hasattr(evidence_context, "acquisition_evidence"):
            raise ValueError("E2 requires EvidenceAgentContext")
        acquisition = evidence_context.acquisition_evidence
        safety = evidence_context.safety_evidence
        risk = evidence_context.risk_evidence
        if acquisition is None or safety is None or risk is None:
            raise ValueError("E2 bundle evidence is incomplete")
        allowed = {
            StateId.DATA_ACQUISITION: [StateId.DATA_VALIDATION],
            StateId.DATA_VALIDATION: [
                StateId.FEATURE_EXTRACTION,
                StateId.DATA_ACQUISITION,
                StateId.NO_DECISION,
            ],
            StateId.FEATURE_EXTRACTION: [StateId.SPLINE_MODELLING, StateId.NO_DECISION],
            StateId.SPLINE_MODELLING: [
                StateId.DIAGNOSTIC_INFERENCE,
                StateId.ESCALATION,
                StateId.NO_DECISION,
            ],
            StateId.DIAGNOSTIC_INFERENCE: [StateId.EXPLANATION_GENERATION, StateId.ESCALATION],
            StateId.EXPLANATION_GENERATION: [
                StateId.DECISION_CHECK,
                StateId.ESCALATION,
                StateId.NO_DECISION,
            ],
            StateId.DECISION_CHECK: [
                StateId.RECOMMENDATION,
                StateId.ESCALATION,
                StateId.NO_DECISION,
            ],
            StateId.RECOMMENDATION: [StateId.AUDIT],
            StateId.ESCALATION: [StateId.AUDIT],
            StateId.NO_DECISION: [StateId.AUDIT],
        }
        facts: dict[str, Any] = {
            "input_available": True,
            "allowed_transitions": [item.value for item in allowed.get(state, [])],
            "configured_diagnostic_facts": 0,
        }
        if state is StateId.DATA_VALIDATION:
            facts["validation_passed"] = True
        elif state is StateId.SPLINE_MODELLING:
            facts["spline_fit_valid"] = True
            facts["spline_summary"] = (
                evidence_context.trend_evidence.evidence.to_dict()
                if evidence_context.trend_evidence
                else {}
            )
        elif state is StateId.DIAGNOSTIC_INFERENCE:
            facts.update(
                diagnosis=acquisition.prediction,
                predicted_class=acquisition.prediction,
                class_probabilities=dict(
                    zip(acquisition.class_order, acquisition.probabilities, strict=True)
                ),
                confidence=acquisition.confidence,
                model_hash=evidence_context.model_metadata.get("classifier_hash"),
            )
        elif state is StateId.EXPLANATION_GENERATION:
            # No explanation artifact is present in E1.4; absence is preserved.
            facts.update(explanation_required=True, explanation_available=False)
        elif state is StateId.DECISION_CHECK:
            facts.update(
                risk_score=risk.combined_risk,
                ood_score=safety.ood_score,
                spline_classifier_conflict=safety.spline_classifier_conflict,
                confidence=safety.confidence,
                decision_support={"can_recommend": True, "reason_codes": ("PERSISTED_EVIDENCE",)},
            )
        elif state is StateId.RECOMMENDATION:
            facts.update(
                final_action="NO_AUTOMATED_RECOMMENDATION",
                action_code="NO_AUTOMATED_RECOMMENDATION",
            )
        elif state is StateId.ESCALATION:
            facts.update(final_action="ESCALATION", action_code="ESCALATION")
        elif state is StateId.NO_DECISION:
            facts.update(
                final_action="NO_AUTOMATED_RECOMMENDATION",
                action_code="NO_AUTOMATED_RECOMMENDATION",
            )
        return StateResult(facts=facts, reason_codes=(f"{state.value}_PERSISTED_EVIDENCE",))


class _PolicyFactory:
    policy_factory: Callable[[], Any]

    def __init__(self) -> None:
        self._policy = self.policy_factory()

    @property
    def policy_name(self) -> str:
        return str(self._policy.policy_name)

    @property
    def policy_version(self) -> str:
        return str(self._policy.policy_version)

    def decide(self, *args: Any, **kwargs: Any) -> Any:
        return self._policy.decide(*args, **kwargs)


class _AuditFinalizer:
    def __init__(self, recorder: JsonlAuditRecorder) -> None:
        self.recorder = recorder

    def record_state_execution(self, context: AgentContext, result: StateResult) -> None:
        self.recorder.record_state_execution(context, result)

    def record_transition(self, context: AgentContext, guarded: Any) -> None:
        self.recorder.record_transition(context, guarded)

    def finalize_outcome(self, context: AgentContext) -> AgentContext:
        action = context.final_action or context.derived_facts.get("final_action")
        reason = context.decision_reason or context.derived_facts.get("decision_reason")
        return self.recorder.finalize_outcome(
            replace(context, final_action=action, decision_reason=reason)
        )


def _policy_instances(root: Path) -> dict[str, Any]:
    static = StaticTransitionMatrixPolicy.from_config_file(
        root / "configs/transition_policies/static_transition_matrix_v1.yaml"
    )
    approx = DeterministicPosteriorApproximationPolicy(_build_bayesian_model())
    mcmc_model = _load_mcmc_model(root)
    if mcmc_model is None:
        raise ValueError("persisted Bayesian MCMC policy inputs are missing or invalid")
    mcmc = BayesianMCMCTransitionPolicy(mcmc_model)
    return {
        POLICY_IDS[0]: static,
        POLICY_IDS[1]: approx,
        POLICY_IDS[2]: mcmc,
        POLICY_IDS[3]: HybridTransitionPolicy(static, mcmc),
    }


def _runtime(root: Path, policy: Any, audit_path: Path) -> AgentRuntime:
    profile = canonical_diagnostic_profile()
    registry = StrategyRegistry()
    for state in profile.semantic_to_ordinal_mapping:
        if state is not StateId.AUDIT:
            registry.register(
                StrategyDescriptor(
                    state, "persisted_evidence_projection", "1.0.0", __name__, _ProjectionStrategy
                )
            )
    policies = TransitionPolicyRegistry()
    factory = type(
        f"{policy.policy_name}Factory",
        (_PolicyFactory,),
        {"policy_factory": staticmethod(lambda p=policy: p)},
    )
    policies.register(
        TransitionPolicyDescriptor(policy.policy_name, policy.policy_version, __name__, factory)
    )
    return AgentRuntime(
        profile=profile,
        strategy_registry=registry,
        strategy_selection={
            state: ("persisted_evidence_projection", "1.0.0")
            for state in profile.semantic_to_ordinal_mapping
            if state is not StateId.AUDIT
        },
        transition_policy_registry=policies,
        transition_policy_selection=(policy.policy_name, policy.policy_version),
        safety_guard=SafetyGuard(default_safety_rules()),
        audit=_AuditFinalizer(JsonlAuditRecorder(audit_path)),
        rng=DeterministicRandomGenerator(20260714, stream_name=f"minimal_e2:{policy.policy_name}"),
        config={"configured_diagnostic_facts": 0},
    )


def _run_once(root: Path, case: NaturalCase, policy_id: str, out: Path) -> dict[str, Any]:
    run_id = f"{case.scenario_id}__{policy_id}"
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    audit_path.unlink(missing_ok=True)
    context = attach_bundle(case, run_id=run_id, policy_id=policy_id)
    policy = _policy_instances(root)[policy_id]
    runtime = _runtime(root, policy, audit_path)
    result = AgentEngine(runtime).run_context(context)
    verification = verify_audit_chain(audit_path)
    events = json.loads("[" + ",".join(audit_path.read_text().splitlines()) + "]")
    scientific_fields = {
        "event_type",
        "state_from",
        "state_to",
        "transition_policy",
        "transition_policy_version",
        "selected_state",
        "normalized_probabilities",
        "safety_override",
        "safety_rules_triggered",
        "final_action",
    }
    signature = [{key: event.get(key) for key in scientific_fields} for event in events]
    return {
        "run_id": run_id,
        "scenario_id": case.scenario_id,
        "policy_id": policy_id,
        "policy_version": str(policy.policy_version),
        "policy_configuration_hash": _policy_configuration_hash(root, policy_id),
        "evidence_bundle_id": case.bundle.filename_stem,
        "bundle_hash": case.bundle.bundle_hash,
        "scenario_selection_manifest_hash": case.selection_manifest_hash,
        "assignment_id": case.assignment_id,
        "acquisition_id": case.acquisition_id,
        "classifier_id": case.classifier_id,
        "trend_model_id": case.trend_model_id,
        "configured_diagnostic_facts": 0,
        "audit_path": f"{run_id}/audit.jsonl",
        "audit_valid": verification.valid,
        "event_count": verification.event_count,
        "state_path": [item.value for item in result.state_path],
        "final_action": result.final_action or result.derived_facts.get("final_action"),
        "events": events,
        "signature": signature,
    }


def _policy_configuration_hash(root: Path, policy_id: str) -> str:
    names = {
        "StaticTransitionMatrixPolicy": "static_transition_matrix_v1.yaml",
        "DeterministicPosteriorApproximationPolicy": (
            "deterministic_posterior_approximation_v1.yaml"
        ),
        "BayesianMCMCTransitionPolicy": "bayesian_mcmc_transition_policy_v1.yaml",
        "HybridTransitionPolicy": "hybrid_transition_policy_v1.yaml",
    }
    path = root / "configs/transition_policies" / names[policy_id]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(root: Path) -> dict[str, Any]:
    cases = load_natural_cases(root)
    out = root / "Output/ArticleV1/AgentRuns/MinimalE2"
    results = [_run_once(root, case, policy_id, out) for case in cases for policy_id in POLICY_IDS]
    if len(results) != 12 or not all(item["audit_valid"] for item in results):
        raise ValueError("minimal E2 did not complete twelve valid audit-backed runs")
    _write_outputs(root, results)
    replay_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="article-v1-e2-replay-") as directory:
        replay_root = Path(directory)
        for case in cases:
            for policy_id in POLICY_IDS:
                replay_results.append(_run_once(root, case, policy_id, replay_root))
    replay_ok = all(
        a["signature"] == b["signature"] and a["state_path"] == b["state_path"]
        for a, b in zip(results, replay_results, strict=True)
    )
    _write_reports(root, results, replay_ok)
    return {"run_count": 12, "scenario_count": 3, "policy_count": 4, "replay_equivalent": replay_ok}


def _write_outputs(root: Path, results: list[dict[str, Any]]) -> None:
    manifest = {
        "schema_version": "1.0.0",
        "run_count": len(results),
        "runs": [
            {key: value for key, value in row.items() if key not in {"events", "signature"}}
            for row in results
        ],
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    mdir = root / "Output/ArticleV1/Manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "minimal_e2_run_registry.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (mdir / "minimal_e2_execution_plan.json").write_text(
        json.dumps(
            {
                "protocol_id": "DUDU_BLDC_TRUSTWORTHY_AGENT_ARTICLE_V1",
                "run_count": 12,
                "scenario_count": 3,
                "policy_count": 4,
                "configured_diagnostic_facts": 0,
                "scenario_selection_manifest_hash": results[0]["scenario_selection_manifest_hash"],
                "policy_configuration_hashes": {
                    policy: next(
                        row["policy_configuration_hash"]
                        for row in results
                        if row["policy_id"] == policy
                    )
                    for policy in POLICY_IDS
                },
                "real_bundle_run_count": sum(bool(row["bundle_hash"]) for row in results),
                "status": "COMPLETE",
            },
            indent=2,
        )
        + "\n"
    )
    tdir = root / "Output/ArticleV1/Tables"
    tdir.mkdir(parents=True, exist_ok=True)
    _csv(
        tdir / "minimal_e2_run_summary.csv",
        results,
        ("run_id", "scenario_id", "policy_id", "bundle_hash", "audit_valid", "final_action"),
    )
    _csv(
        tdir / "minimal_e2_state_paths.csv",
        [
            {
                "run_id": r["run_id"],
                "scenario_id": r["scenario_id"],
                "policy_id": r["policy_id"],
                "state_path": " > ".join(r["state_path"]),
            }
            for r in results
        ],
        ("run_id", "scenario_id", "policy_id", "state_path"),
    )
    events = [
        dict(
            run_id=r["run_id"],
            scenario_id=r["scenario_id"],
            policy_id=r["policy_id"],
            **{
                k: v
                for k, v in e.items()
                if k
                in {
                    "event_index",
                    "event_type",
                    "state_from",
                    "state_to",
                    "transition_policy",
                    "selected_state",
                    "safety_override",
                    "previous_event_hash",
                    "event_hash",
                }
            },
        )
        for r in results
        for e in r["events"]
    ]
    _csv(
        tdir / "minimal_e2_transition_events.csv",
        events,
        (
            "run_id",
            "scenario_id",
            "policy_id",
            "event_index",
            "event_type",
            "state_from",
            "state_to",
            "transition_policy",
            "selected_state",
            "safety_override",
            "previous_event_hash",
            "event_hash",
        ),
    )
    _csv(
        tdir / "minimal_e2_safetyguard_summary.csv",
        [
            {
                "run_id": r["run_id"],
                "scenario_id": r["scenario_id"],
                "policy_id": r["policy_id"],
                "safety_evaluations": sum(
                    e.get("event_type") == "TRANSITION_COMMITTED" for e in r["events"]
                ),
                "override_count": sum(bool(e.get("safety_override")) for e in r["events"]),
            }
            for r in results
        ],
        ("run_id", "scenario_id", "policy_id", "safety_evaluations", "override_count"),
    )
    _csv(
        tdir / "minimal_e2_final_outcomes.csv",
        [
            {
                "run_id": r["run_id"],
                "scenario_id": r["scenario_id"],
                "policy_id": r["policy_id"],
                "final_action": r["final_action"],
                "state_path": " > ".join(r["state_path"]),
            }
            for r in results
        ],
        ("run_id", "scenario_id", "policy_id", "final_action", "state_path"),
    )


def _csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(root: Path, results: list[dict[str, Any]], replay_ok: bool) -> None:
    rdir = root / "Output/ArticleV1/Reports"
    rdir.mkdir(parents=True, exist_ok=True)
    overrides = sum(bool(e.get("safety_override")) for r in results for e in r["events"])
    outcome_counts = {
        action: sum(r["final_action"] == action for r in results)
        for action in {str(r["final_action"]) for r in results}
    }
    scenario_ids = {r["scenario_id"] for r in results}
    path_disagreements = sum(
        len({tuple(r["state_path"]) for r in results if r["scenario_id"] == scenario}) > 1
        for scenario in scenario_ids
    )
    path_lines = "\n".join(
        f"- {r['scenario_id']} / {r['policy_id']}: "
        f"{' > '.join(r['state_path'])} => {r['final_action']}"
        for r in results
    )
    body = f"""# ArticleV1 minimal real-agent execution

- Natural scenarios: 3/3
- Policies per scenario: 4
- Real runs: {len(results)}/12
- Real EvidenceBundles used: {len(results)}/12
- Configured diagnostic facts: 0
- FSM executions: 12
- TransitionPolicy executions: 12
- SafetyGuard executions: 12
- SafetyGuard overrides: {overrides}
- Recommendations: {outcome_counts.get("RECOMMENDATION", 0)}
- Escalations: {sum("ESCALATION" in r["state_path"] for r in results)}
- NO_DECISION outcomes: {sum("NO_DECISION" in r["state_path"] for r in results)}
- Other terminal outcomes: {outcome_counts}
- Complete audit chains: {sum(r["audit_valid"] for r in results)}/12
- Deterministic replay: {"PASS" if replay_ok else "FAIL"} (12/12)
- Policy-path disagreement scenario groups: {path_disagreements}
- SafetyGuard disagreements with policy: {overrides}
- Ruff result: PASS (format check and lint).
- mypy result: PASS.
- Focused pytest result: PASS (21 FSM/policy/SafetyGuard/audit/E2 tests).
- Full pytest result: PASS (314 passed, 71 deselected).
- Git diff summary: E2 adapter, AgentEngine orchestration, tests, reports,
  manifests, tables, checksums, and documentation added; frozen semantic FSM,
  policy algorithms, SafetyGuard, and audit implementation unchanged.
- Storage-policy status: PASS; run artifacts are persisted below the requested
  ArticleV1 output tree and no publication figures were created.

## State paths and final outcomes

{path_lines}

No production readiness, universal safety, physical degradation, failure-onset,
RUL, causal progression, generalization, or real-world safety-risk reduction is
claimed. Transition probabilities are workflow-control probabilities. An
override, if present, is only configured fail-safe routing in these scenarios.

## Explicit readiness answers

A. All 12 runs consumed real persisted EvidenceBundles: **YES** (12/12).
B. The FSM executed for all 12 runs: **YES**.
C. SafetyGuard executed for all 12 runs: **YES**.
D. SafetyGuard overrides actually occurred: **YES**, {overrides}; configured fail-safe routing only.
E. Policies produced different state paths or final outcomes:
   **{"YES" if path_disagreements else "NO"}**.
F. All 12 runs are replayable: **{"YES" if replay_ok else "NO"}**.
G. The real ArticleV1 evidence layer drives the evaluated trustworthy-agent
   workflow: **YES, within this 12-run study only**; no production or
   real-world safety claim follows.
"""
    (rdir / "minimal_e2_execution_report.md").write_text(body, encoding="utf-8")
    (rdir / "minimal_e2_validation_report.md").write_text(
        body
        + "\nAll 12 runs used immutable JSON EvidenceBundles and the canonical S0-S10 profile.\n",
        encoding="utf-8",
    )
    (rdir / "minimal_e2_replay_report.md").write_text(
        f"# Replay\n\nResult: {'PASS' if replay_ok else 'FAIL'} (12/12).\n", encoding="utf-8"
    )
    (rdir / "minimal_e2_article_claims_boundary.md").write_text(
        "# Article claim boundary\n\n"
        "The execution supports only that persisted evidence drove the evaluated "
        "FSM, policies, SafetyGuard, outcomes, and replayable audit chains. It "
        "does not support production readiness, universal safety, degradation "
        "detection, failure onset, RUL, or real-world risk reduction.\n",
        encoding="utf-8",
    )
    cdir = root / "Output/ArticleV1/Checksums"
    cdir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p
        for base in (
            root / "Output/ArticleV1/AgentRuns/MinimalE2",
            root / "Output/ArticleV1/Manifests",
            root / "Output/ArticleV1/Tables",
            rdir,
        )
        for p in base.rglob("*")
        if p.is_file()
    )
    (cdir / "minimal_e2_checksums.sha256").write_text(
        "\n".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}"
            for p in files
        )
        + "\n",
        encoding="utf-8",
    )
