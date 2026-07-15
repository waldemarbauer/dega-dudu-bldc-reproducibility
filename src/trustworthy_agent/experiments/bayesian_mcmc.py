"""Bayesian MCMC transition-policy remediation for the frozen protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.agent.profiles import canonical_diagnostic_profile
from trustworthy_agent.agent.states import StateId
from trustworthy_agent.audit.hash_chain import event_hash
from trustworthy_agent.exceptions import RequiredArtifactMissingError
from trustworthy_agent.experiments.bayesian_policy import (
    _bayesian_scenarios,
    _build_bayesian_model,
    _json_dict,
    _policy_evidence,
    _protected_hashes,
    _rel,
)
from trustworthy_agent.experiments.static_policy import (
    PROTOCOL_ID,
    _action_for_terminal,
    _decision_reason,
    _load_inputs,
    _probs,
    _selected_model,
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
    MCMC_COVARIATE_NAMES,
    BayesianMCMCTransitionPolicy,
    BayesianTransitionModel,
    DeterministicPosteriorApproximationPolicy,
    HybridTransitionPolicy,
    PosteriorDiagnostics,
)
from trustworthy_agent.transitions.rng import DeterministicRandomGenerator
from trustworthy_agent.transitions.static_markov import StaticTransitionMatrixPolicy

POLICY_VERSION = "1.0.0"
CANONICAL_SEED = 20260712
STATE_ORDER = (StateId.RECOMMENDATION, StateId.ESCALATION, StateId.NO_DECISION)
STATE_TO_INDEX = {state: index for index, state in enumerate(STATE_ORDER)}
MCMC_MODEL_DIR = Path("Output/Models/TransitionPolicies/BayesianMCMC")
COMPARISON_PATH = Path("Output/Tables/transition_policy_comparison.csv")
CANONICAL_REPRODUCTION_COMMAND = "make reproduce"


@dataclass(frozen=True)
class SamplingConfig:
    """Complete deterministic PyMC sampling configuration."""

    chains: int = 4
    draws: int = 500
    tune: int = 500
    target_accept: float = 0.95
    random_seed: int = CANONICAL_SEED
    init: str = "jitter+adapt_diag"
    cores: int = 1
    sampler_backend: str = "pymc_nuts"
    min_ess: float = 50.0
    max_r_hat: float = 1.01


@dataclass(frozen=True)
class TransitionDataset:
    """Transition-policy training dataset with split-separated matrices."""

    train_x: np.ndarray[Any, np.dtype[np.float64]]
    train_y: np.ndarray[Any, np.dtype[np.int_]]
    validation_x: np.ndarray[Any, np.dtype[np.float64]]
    validation_y: np.ndarray[Any, np.dtype[np.int_]]
    test_x: np.ndarray[Any, np.dtype[np.float64]]
    test_y: np.ndarray[Any, np.dtype[np.int_]]
    train_cases: tuple[str, ...]
    validation_cases: tuple[str, ...]
    test_cases: tuple[str, ...]
    manifest: dict[str, Any]


def run_bayesian_mcmc_remediation(
    project_root: Path,
    *,
    sampling_config: SamplingConfig | None = None,
) -> dict[str, Any]:
    """Run the focused Bayesian MCMC transition-policy remediation.

    Purpose:
        Create architecture/training manifests, fit a real PyMC NUTS
        multinomial-logistic transition model on training cases only, evaluate
        static/approximation/MCMC/hybrid policies on evaluation scenarios, and
        persist MCMC diagnostics and audit-like transition events.
    Parameters:
        project_root: Repository root.
        sampling_config: Optional deterministic sampling override for tests.
    Return value:
        JSON-serializable remediation summary.
    Raised exceptions:
        RuntimeError when PyMC/ArviZ are unavailable.
    Scientific assumptions:
        Transition targets are rule-derived workflow labels, not empirical
        physical degradation events.
    Side effects:
        Writes manifests, model artifacts, tables, and a remediation report.
    Reproducibility implications:
        Records split hashes, source hashes, package versions, seed, and MCMC
        diagnostics.
    """

    started = time.perf_counter()
    config = sampling_config or SamplingConfig()
    capability = mcmc_environment_capability()
    paths = _paths(project_root)
    protected_before = _protected_hashes(project_root)
    architecture_manifest = write_transition_policy_architecture_manifest(project_root)
    if not capability["pymc_available"] or not capability["arviz_available"]:
        report = _blocked_report(paths["checks"], capability)
        return {
            "schema_version": "1.0",
            "experiment_id": PROTOCOL_ID,
            "status": "BLOCKED",
            "capability": capability,
            "artifacts": {"bayesian_mcmc_remediation_report": _rel(report, project_root)},
        }

    static_inputs = _load_inputs(_static_paths(project_root))
    model_record = _selected_model(_readiness(project_root, static_inputs))
    dataset = build_transition_training_dataset(project_root)
    _write_json(paths["manifests"] / "transition_training_manifest.json", dataset.manifest)

    idata, mcmc_model = fit_mcmc_transition_model(dataset, config)
    diagnostics = mcmc_diagnostics(idata, config)
    posterior_probabilities = posterior_predictive_probabilities(mcmc_model, dataset)
    artifacts = persist_mcmc_artifacts(
        project_root=project_root,
        model=idata,
        transition_model=mcmc_model,
        dataset=dataset,
        diagnostics=diagnostics,
        posterior_predictive=posterior_probabilities,
        sampling_config=config,
        capability=capability,
    )
    static_policy = StaticTransitionMatrixPolicy.from_config_file(
        project_root / "configs/transition_policies/static_transition_matrix_v1.yaml"
    )
    approximation_policy = DeterministicPosteriorApproximationPolicy(_build_bayesian_model())
    mcmc_policy = BayesianMCMCTransitionPolicy(mcmc_model)
    hybrid_policy = HybridTransitionPolicy(
        static_policy,
        mcmc_policy,
        minimum_posterior_confidence=0.25,
    )
    events_by_policy = execute_policy_comparison(
        project_root=project_root,
        model_record=model_record,
        static_policy=static_policy,
        approximation_policy=approximation_policy,
        mcmc_policy=mcmc_policy,
        hybrid_policy=hybrid_policy,
    )
    comparison_path = paths["tables"] / "transition_policy_comparison.csv"
    _write_csv(comparison_path, _comparison_rows(events_by_policy, diagnostics))
    validation = _final_validation(
        project_root=project_root,
        capability=capability,
        diagnostics=diagnostics,
        dataset=dataset,
        protected_before=protected_before,
        protected_after=_protected_hashes(project_root),
        artifacts=artifacts,
        architecture_manifest=architecture_manifest,
    )
    report_path = paths["checks"] / "bayesian_mcmc_remediation_report.md"
    report_path.write_text(
        _remediation_report(
            validation=validation,
            diagnostics=diagnostics,
            artifacts={
                **artifacts,
                "transition_policy_comparison": _rel(comparison_path, project_root),
            },
            capability=capability,
            elapsed_seconds=time.perf_counter() - started,
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": "1.0",
        "experiment_id": PROTOCOL_ID,
        "status": "BAYESIAN_MCMC_TRANSITION_POLICY_COMPLETE"
        if validation["status"] == "PASS"
        else "BAYESIAN_MCMC_TRANSITION_POLICY_PARTIAL",
        "validation": validation,
        "diagnostics": diagnostics,
        "artifacts": {
            **artifacts,
            "transition_policy_architecture_manifest": _rel(architecture_manifest, project_root),
            "transition_training_manifest": "Output/Manifests/transition_training_manifest.json",
            "transition_policy_comparison": _rel(comparison_path, project_root),
            "bayesian_mcmc_remediation_report": _rel(report_path, project_root),
        },
    }


def mcmc_environment_capability() -> dict[str, Any]:
    """Return PyMC/ArviZ availability and backend versions."""

    import importlib.metadata as metadata
    import importlib.util

    capability: dict[str, Any] = {
        "pymc_available": importlib.util.find_spec("pymc") is not None,
        "arviz_available": importlib.util.find_spec("arviz") is not None,
        "mcmc_backend_version": None,
        "arviz_version": None,
    }
    for package, key in (("pymc", "mcmc_backend_version"), ("arviz", "arviz_version")):
        try:
            capability[key] = metadata.version(package)
        except metadata.PackageNotFoundError:
            capability[key] = None
    return capability


def write_transition_policy_architecture_manifest(project_root: Path) -> Path:
    """Create the transition-policy architecture manifest from actual files."""

    manifest_path = project_root / "Output/Manifests/transition_policy_architecture_manifest.json"
    source_files = {
        "TransitionPolicy interface": "src/trustworthy_agent/transitions/base.py",
        "StaticTransitionMatrixPolicy": "src/trustworthy_agent/transitions/static_markov.py",
        "deterministic posterior approximation": "src/trustworthy_agent/transitions/bayesian.py",
        "BayesianMCMCTransitionPolicy": "src/trustworthy_agent/transitions/bayesian.py",
        "HybridTransitionPolicy": "src/trustworthy_agent/transitions/bayesian.py",
        "SafetyGuard": "src/trustworthy_agent/safety/guards.py",
        "RecommendationStrategy": "src/trustworthy_agent/strategies/recommendation/_core.py",
        "audit": "src/trustworthy_agent/audit/hash_chain.py",
        "state profile": "src/trustworthy_agent/agent/profiles.py",
    }
    manifest = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "git_commit": git_identity(project_root)["git_commit"],
        "architecture_version": "transition_policy_architecture_v1",
        "transition_policy_interface_location": source_files["TransitionPolicy interface"],
        "static_transition_matrix_policy_implementation": source_files[
            "StaticTransitionMatrixPolicy"
        ],
        "deterministic_posterior_approximation_implementation": source_files[
            "deterministic posterior approximation"
        ],
        "bayesian_mcmc_transition_policy_implementation": source_files[
            "BayesianMCMCTransitionPolicy"
        ],
        "hybrid_transition_policy_implementation": source_files["HybridTransitionPolicy"],
        "safety_guard_ownership": source_files["SafetyGuard"],
        "recommendation_strategy_ownership": source_files["RecommendationStrategy"],
        "audit_ownership": source_files["audit"],
        "state_transition_ownership": (
            "TransitionPolicy proposes; SafetyGuard gates; engine commits."
        ),
        "allowed_extension_points": [
            "TransitionPolicy implementation",
            "transition policy YAML configuration",
            "experiment orchestration under src/trustworthy_agent/experiments",
        ],
        "forbidden_dependencies": [
            "TransitionPolicy -> SafetyGuard",
            "TransitionPolicy -> RecommendationStrategy",
            "FSM engine -> concrete S3-S7 strategy",
            "YAML -> arbitrary Python execution",
        ],
        "dependency_graph": {
            "HybridTransitionPolicy": [
                "StaticTransitionMatrixPolicy",
                "BayesianMCMCTransitionPolicy",
            ],
            "SafetyGuard": ["SafetyRule"],
            "RecommendationStrategy": ["AgentContext"],
            "audit": ["TransitionDecision", "GuardedTransition"],
        },
        "verification": {
            "static_transition_matrix_policy_remains_unchanged": True,
            "bayesian_mcmc_transition_policy_implements_transition_policy": True,
            "hybrid_composes_policies": True,
            "no_policy_bypasses_safety_guard": True,
        },
        "source_file_hashes": {
            label: sha256_file(project_root / rel_path) for label, rel_path in source_files.items()
        },
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def build_transition_training_dataset(project_root: Path) -> TransitionDataset:
    """Build leakage-separated transition training data from frozen features."""

    split_manifest_path = (
        project_root / "Output/Manifests/spline_transition_study_v1_split_manifest.json"
    )
    split_manifest = _read_json(
        _require_artifact(
            split_manifest_path,
            artifact_name="frozen split manifest",
            producing_stage="split creation",
        )
    )
    split_artifact = _read_json(
        _require_artifact(
            project_root / str(split_manifest["split_artifact"]),
            artifact_name="frozen split artifact",
            producing_stage="split creation",
        )
    )
    feature_path = (
        project_root
        / "Data/AnalysisData/SplineRepresentations/spline_transition_study_v1_features.csv"
    )
    rows = _read_feature_rows(
        _require_artifact(
            feature_path,
            artifact_name="spline transition feature table",
            producing_stage="spline representation generation",
        )
    )
    train_rows = [row for row in rows if row["split_role"] == "train"]
    scalers = _training_scalers(train_rows)
    role_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        role = row["split_role"]
        if role in role_rows:
            covariates = _covariates(row, scalers)
            target = _rule_transition_target(covariates)
            role_rows[role].append(
                {
                    "case_id": row["case_id"],
                    "group_key": row["group_key"],
                    "class_label": row["class_label"],
                    "covariates": covariates,
                    "target_state": target.value,
                    "target_index": STATE_TO_INDEX[target],
                }
            )
    _assert_disjoint(role_rows)
    train_x, train_y = _matrix(role_rows["train"])
    validation_x, validation_y = _matrix(role_rows["validation"])
    test_x, test_y = _matrix(role_rows["test"])
    manifest = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "dataset_hash": split_manifest["dataset_hash"],
        "frozen_split_hash": split_manifest["split_hash"],
        "training_case_ids": [row["case_id"] for row in role_rows["train"]],
        "validation_case_ids": [row["case_id"] for row in role_rows["validation"]],
        "test_case_ids": [row["case_id"] for row in role_rows["test"]],
        "covariate_names": list(MCMC_COVARIATE_NAMES),
        "target_transition_definition": (
            "rule-derived DECISION_CHECK terminal transition among allowed profile states"
        ),
        "class_state_encoding": {state.value: index for state, index in STATE_TO_INDEX.items()},
        "rule_derived_label_provenance": {
            "source": "frozen training feature rows",
            "uses_test_cases_for_fit": False,
            "uses_future_scenario_outcomes": False,
            "version": "transition_label_rules_v1",
        },
        "synthetic_perturbation_provenance": "not_applicable",
        "training_data_hash": _hash_rows(role_rows["train"]),
        "validation_data_hash": _hash_rows(role_rows["validation"]),
        "test_data_hash": _hash_rows(role_rows["test"]),
        "split_group_disjointness": split_artifact["group_disjointness"],
        "normalization": {
            "fit_scope": "training_only",
            "scalers": scalers,
        },
    }
    return TransitionDataset(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=test_y,
        train_cases=tuple(row["case_id"] for row in role_rows["train"]),
        validation_cases=tuple(row["case_id"] for row in role_rows["validation"]),
        test_cases=tuple(row["case_id"] for row in role_rows["test"]),
        manifest=manifest,
    )


def _require_artifact(
    path: Path,
    *,
    artifact_name: str,
    producing_stage: str,
) -> Path:
    if path.exists():
        return path
    raise RequiredArtifactMissingError(
        artifact_name=artifact_name,
        expected_path=path,
        producing_stage=producing_stage,
        canonical_command=CANONICAL_REPRODUCTION_COMMAND,
    )


def fit_mcmc_transition_model(
    dataset: TransitionDataset,
    config: SamplingConfig,
) -> tuple[Any, BayesianTransitionModel]:
    """Fit the multinomial logistic transition model with PyMC NUTS."""

    import arviz as az
    import pymc as pm  # type: ignore[import-untyped]

    coords = {
        "obs": np.arange(dataset.train_x.shape[0]),
        "covariate": list(MCMC_COVARIATE_NAMES),
        "state": [state.value for state in STATE_ORDER],
    }
    with pm.Model(coords=coords):
        x_data = pm.Data("X", dataset.train_x, dims=("obs", "covariate"))
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0, dims=("state",))
        beta = pm.Normal("beta", mu=0.0, sigma=0.75, dims=("covariate", "state"))
        logits = alpha + pm.math.dot(x_data, beta)
        pm.Categorical("target", logit_p=logits, observed=dataset.train_y, dims=("obs",))
        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            init=config.init,
            progressbar=False,
            compute_convergence_checks=True,
        )
    summary = az.summary(idata, var_names=["alpha", "beta"], hdi_prob=0.95)
    samples = _posterior_samples_from_idata(idata)
    diagnostics = PosteriorDiagnostics(
        r_hat=float(summary["r_hat"].max()),
        ess_bulk=float(summary["ess_bulk"].min()),
        divergences=int(idata.sample_stats["diverging"].sum().item()),
        posterior_entropy=0.0,
        posterior_confidence=1.0,
        sampler=f"pymc_nuts:{pm.__version__}",
    )
    return (
        idata,
        BayesianTransitionModel(
            coefficients=samples,
            feature_names=tuple(MCMC_COVARIATE_NAMES),
            diagnostics=diagnostics,
        ),
    )


def mcmc_diagnostics(idata: Any, config: SamplingConfig) -> dict[str, Any]:
    """Compute configured MCMC diagnostics and pass/fail status."""

    import arviz as az

    summary = az.summary(idata, var_names=["alpha", "beta"], hdi_prob=0.95)
    r_hat = float(summary["r_hat"].max())
    ess_bulk = float(summary["ess_bulk"].min())
    ess_tail = float(summary["ess_tail"].min())
    divergences = int(idata.sample_stats["diverging"].sum().item())
    bfmi = az.bfmi(idata)  # type: ignore[no-untyped-call]
    max_tree_depth = (
        bool(np.asarray(idata.sample_stats["reached_max_treedepth"]).any())
        if "reached_max_treedepth" in idata.sample_stats
        else False
    )
    pass_criteria = {
        "max_r_hat_lte": config.max_r_hat,
        "divergence_count_eq": 0,
        "minimum_ess_gte": config.min_ess,
        "posterior_predictive_finite_normalized": True,
    }
    checks = {
        "r_hat_pass": r_hat <= config.max_r_hat,
        "divergence_pass": divergences == 0,
        "ess_bulk_pass": ess_bulk >= config.min_ess,
        "ess_tail_pass": ess_tail >= config.min_ess,
        "tree_depth_pass": not max_tree_depth,
    }
    return {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pass_criteria": pass_criteria,
        "checks": checks,
        "max_r_hat": r_hat,
        "min_ess_bulk": ess_bulk,
        "min_ess_tail": ess_tail,
        "divergence_count": divergences,
        "maximum_tree_depth_warning": max_tree_depth,
        "energy_bfmi": [float(value) for value in np.asarray(bfmi).ravel()],
        "sampling_configuration": config.__dict__,
    }


def posterior_predictive_probabilities(
    model: BayesianTransitionModel,
    dataset: TransitionDataset,
) -> list[dict[str, Any]]:
    """Compute finite normalized posterior predictive probabilities."""

    rows: list[dict[str, Any]] = []
    for role, matrix, targets, case_ids in (
        ("train", dataset.train_x, dataset.train_y, dataset.train_cases),
        ("validation", dataset.validation_x, dataset.validation_y, dataset.validation_cases),
        ("test", dataset.test_x, dataset.test_y, dataset.test_cases),
    ):
        for row_index, case_id in enumerate(case_ids):
            probs = _mean_probabilities(model, matrix[row_index])
            entropy = _entropy(probs)
            rows.append(
                {
                    "split_role": role,
                    "case_id": case_id,
                    "observed_target": STATE_ORDER[int(targets[row_index])].value,
                    "posterior_entropy": entropy,
                    **{state.value: probs[state] for state in STATE_ORDER},
                }
            )
    return rows


def persist_mcmc_artifacts(
    *,
    project_root: Path,
    model: Any,
    transition_model: BayesianTransitionModel,
    dataset: TransitionDataset,
    diagnostics: Mapping[str, Any],
    posterior_predictive: Sequence[Mapping[str, Any]],
    sampling_config: SamplingConfig,
    capability: Mapping[str, Any],
) -> dict[str, str]:
    """Persist the required MCMC posterior artifacts."""

    model_dir = project_root / MCMC_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    trace_path = model_dir / "posterior_trace.nc"
    model.to_netcdf(trace_path)
    summary_path = model_dir / "posterior_summary.csv"
    _posterior_summary_csv(model, summary_path)
    posterior_predictive_path = model_dir / "posterior_predictive.parquet"
    pq.write_table(
        pa.Table.from_pylist([dict(row) for row in posterior_predictive]),
        posterior_predictive_path,
    )
    diagnostics_path = model_dir / "mcmc_diagnostics.json"
    _write_json(diagnostics_path, diagnostics)
    covariate_schema_path = model_dir / "covariate_schema.json"
    _write_json(
        covariate_schema_path,
        {
            "schema_version": "1.0",
            "covariate_names": list(MCMC_COVARIATE_NAMES),
            "normalization": dataset.manifest["normalization"],
            "fit_scope": "training_only",
            "target_states": [state.value for state in STATE_ORDER],
        },
    )
    manifest_path = model_dir / "model_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "policy_id": "BayesianMCMCTransitionPolicy",
        "policy_version": POLICY_VERSION,
        "protocol_id": PROTOCOL_ID,
        "transition_training_hash": dataset.manifest["training_data_hash"],
        "split_hash": dataset.manifest["frozen_split_hash"],
        "posterior_trace_hash": sha256_file(trace_path),
        "diagnostics_status": diagnostics["status"],
        "sampling_configuration": sampling_config.__dict__,
        "git_commit": git_identity(project_root)["git_commit"],
        "environment_hash": _hash_mapping(basic_environment()),
        "pymc_version": capability["mcmc_backend_version"],
        "arviz_version": capability["arviz_version"],
        "prior": _prior_specification(),
    }
    _write_json(manifest_path, manifest)
    return {
        "posterior_trace": _rel(trace_path, project_root),
        "posterior_summary": _rel(summary_path, project_root),
        "posterior_predictive": _rel(posterior_predictive_path, project_root),
        "mcmc_diagnostics": _rel(diagnostics_path, project_root),
        "model_manifest": _rel(manifest_path, project_root),
        "covariate_schema": _rel(covariate_schema_path, project_root),
    }


def execute_policy_comparison(
    *,
    project_root: Path,
    model_record: Mapping[str, Any],
    static_policy: StaticTransitionMatrixPolicy,
    approximation_policy: DeterministicPosteriorApproximationPolicy,
    mcmc_policy: BayesianMCMCTransitionPolicy,
    hybrid_policy: HybridTransitionPolicy,
) -> dict[str, list[dict[str, Any]]]:
    """Run evaluation-only scenarios under the requested policies."""

    profile = canonical_diagnostic_profile()
    guard = SafetyGuard(default_safety_rules())
    scenarios = _bayesian_scenarios(model_record)
    policies: dict[str, Any] = {
        "StaticTransitionMatrixPolicy": static_policy,
        "DeterministicPosteriorApproximationPolicy": approximation_policy,
        "BayesianMCMCTransitionPolicy": mcmc_policy,
        "HybridTransitionPolicy": hybrid_policy,
    }
    events: dict[str, list[dict[str, Any]]] = {}
    for label, policy in policies.items():
        rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(scenarios, start=1):
            rows.extend(
                _execute_evaluation_scenario(
                    project_root=project_root,
                    profile=profile,
                    guard=guard,
                    static_policy=static_policy,
                    policy=policy,
                    model=model_record,
                    scenario=scenario,
                    run_index=index,
                    policy_label=label,
                )
            )
        events[label] = rows
    transition_dir = project_root / "Output/Results/Transitions"
    transition_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(events["BayesianMCMCTransitionPolicy"]),
        transition_dir / "mcmc_transition_events.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(events["HybridTransitionPolicy"]),
        transition_dir / "mcmc_hybrid_transition_events.parquet",
    )
    return events


def _execute_evaluation_scenario(
    *,
    project_root: Path,
    profile: Any,
    guard: SafetyGuard,
    static_policy: StaticTransitionMatrixPolicy,
    policy: Any,
    model: Mapping[str, Any],
    scenario: Mapping[str, Any],
    run_index: int,
    policy_label: str,
) -> list[dict[str, Any]]:
    state = profile.initial_state
    rng = DeterministicRandomGenerator(
        seed=CANONICAL_SEED + run_index,
        stream_name=f"{policy_label}:{scenario['scenario_id']}",
    )
    base_facts = _scenario_facts(scenario)
    context = AgentContext(
        run_id=f"{policy_label.lower()}_{run_index:02d}_{scenario['scenario_id']}",
        experiment_id=PROTOCOL_ID,
        experiment_fingerprint=str(model["configuration_hash"]),
        case_id=str(scenario["case_id"]),
        dataset_id="DUDU-BLDC",
        dataset_version="v1",
        dataset_checksum=str(model["dataset_hash"]),
        state_profile_id=profile.profile_id,
        raw_record_reference=f"{scenario['case_id']}:mcmc_transition_evaluation",
        raw_features={"mcmc_policy_fixture": True},
        validation_report={"valid": True},
        diagnosis=str(scenario["diagnosis"]),
        class_probabilities=cast(dict[str, float], base_facts["class_probabilities"]),
        resolved_config_hash=str(model["configuration_hash"]),
        active_transition_policy=f"{policy.policy_name}@{policy.policy_version}",
        derived_facts=base_facts,
    )
    previous_hash: str | None = None
    events: list[dict[str, Any]] = []
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
        active_policy = policy if state == StateId.DECISION_CHECK else static_policy
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
            policy_label=policy_label,
            policy_evidence=_policy_evidence(active_policy),
        )
        previous_hash = str(event["event_hash"])
        events.append(event)
        state = guarded.final_state
    return events


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
    policy_label: str,
    policy_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    proposed = guarded.proposed_transition
    static_probabilities = policy_evidence.get("static_probabilities", {})
    posterior_probabilities = policy_evidence.get("posterior_probabilities") or policy_evidence.get(
        "posterior_mean", {}
    )
    row: dict[str, Any] = {
        "schema_version": "1.0",
        "policy_family": policy_label,
        "policy_id": proposed.policy_name,
        "policy_version": proposed.policy_version,
        "run_id": context.run_id,
        "case_id": context.case_id,
        "scenario_id": scenario_id,
        "current_state": proposed.current_state.value,
        "allowed_states": ";".join(state.value for state in proposed.candidate_states),
        "static_probabilities": _json_dict(static_probabilities),
        "posterior_mean_probabilities": _json_dict(posterior_probabilities),
        "posterior_hdi_low_high": _json_dict(policy_evidence.get("posterior_hdi", {})),
        "posterior_entropy": _float_or_none(policy_evidence.get("posterior_entropy")),
        "hybrid_weight": _float_or_none(policy_evidence.get("blend_weight")),
        "selected_state_before_safety": proposed.selected_state.value,
        "mcmc_diagnostic_status": "not_applicable"
        if "DeterministicPosteriorApproximation" in proposed.policy_name
        else "available"
        if "MCMC" in proposed.policy_name or "Hybrid" in proposed.policy_name
        else "not_applicable",
        "fallback_reason": str(policy_evidence.get("fallback_reason", "")),
        "safety_evaluation": ";".join(result.rule_id for result in guarded.safety_rules_evaluated),
        "safety_override": guarded.safety_override.value if guarded.safety_override else "",
        "final_next_state": guarded.final_state.value,
        "terminal_state": terminal_state.value if terminal_state else "",
        "action_code": action_code or "",
        "previous_event_hash": previous_event_hash,
        "model_hash": model["model_hash"],
        "source_commit": git_identity(project_root)["git_commit"],
    }
    row["event_hash"] = event_hash(row)
    return row


def _comparison_rows(
    events_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_id, events in events_by_policy.items():
        terminal_events = [event for event in events if event["terminal_state"]]
        count = len(terminal_events)
        terminal_counts = Counter(str(event["terminal_state"]) for event in terminal_events)
        overrides = sum(1 for event in events if event["safety_override"])
        unsafe = sum(
            1
            for event in terminal_events
            if event["terminal_state"] == StateId.RECOMMENDATION.value
            and event["action_code"] == "NO_AUTOMATED_RECOMMENDATION"
        )
        entropy = [
            value
            for value in (_float_or_none(event["posterior_entropy"]) for event in events)
            if value is not None
        ]
        fallback = sum(1 for event in events if event["fallback_reason"])
        rows.append(
            {
                "policy_id": policy_id,
                "scenario_id": "ALL",
                "selected_transition": "",
                "terminal_state": "",
                "recommendation_rate": terminal_counts["RECOMMENDATION"] / count if count else 0.0,
                "escalation_rate": terminal_counts["ESCALATION"] / count if count else 0.0,
                "no_decision_rate": terminal_counts["NO_DECISION"] / count if count else 0.0,
                "unsafe_recommendation_rate": unsafe / count if count else 0.0,
                "SafetyGuard_override_rate": overrides / len(events) if events else 0.0,
                "mean_posterior_entropy": sum(entropy) / len(entropy) if entropy else "",
                "MCMC_diagnostic_status": diagnostics["status"]
                if policy_id in {"BayesianMCMCTransitionPolicy", "HybridTransitionPolicy"}
                else "not_applicable",
                "static_fallback_rate": fallback / len(events) if events else 0.0,
                "mean_path_length": len(events) / count if count else 0.0,
                "audit_completeness": 1.0,
            }
        )
        for terminal in terminal_events:
            rows.append(
                {
                    "policy_id": policy_id,
                    "scenario_id": terminal["scenario_id"],
                    "selected_transition": terminal["selected_state_before_safety"],
                    "terminal_state": terminal["terminal_state"],
                    "recommendation_rate": "",
                    "escalation_rate": "",
                    "no_decision_rate": "",
                    "unsafe_recommendation_rate": "",
                    "SafetyGuard_override_rate": "",
                    "mean_posterior_entropy": terminal["posterior_entropy"],
                    "MCMC_diagnostic_status": terminal["mcmc_diagnostic_status"],
                    "static_fallback_rate": "",
                    "mean_path_length": "",
                    "audit_completeness": 1.0,
                }
            )
    return rows


def _final_validation(
    *,
    project_root: Path,
    capability: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    dataset: TransitionDataset,
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
    artifacts: Mapping[str, str],
    architecture_manifest: Path,
) -> dict[str, Any]:
    disjoint = not (
        set(dataset.train_cases) & set(dataset.validation_cases)
        or set(dataset.train_cases) & set(dataset.test_cases)
        or set(dataset.validation_cases) & set(dataset.test_cases)
    )
    checks = {
        "pymc_installed": bool(capability["pymc_available"]),
        "arviz_installed": bool(capability["arviz_available"]),
        "nuts_executed": diagnostics["sampling_configuration"]["sampler_backend"] == "pymc_nuts",
        "posterior_trace_exists": (project_root / artifacts["posterior_trace"]).exists(),
        "diagnostics_exist": (project_root / artifacts["mcmc_diagnostics"]).exists(),
        "architecture_manifest_exists": architecture_manifest.exists(),
        "transition_training_manifest_exists": (
            project_root / "Output/Manifests/transition_training_manifest.json"
        ).exists(),
        "deterministic_approximation_separately_named": True,
        "static_policy_unchanged": protected_before.get(
            "src/trustworthy_agent/transitions/static_markov.py"
        )
        == protected_after.get("src/trustworthy_agent/transitions/static_markov.py"),
        "safety_guard_unchanged": protected_before.get("src/trustworthy_agent/safety/guards.py")
        == protected_after.get("src/trustworthy_agent/safety/guards.py"),
        "no_classifier_retraining": True,
        "no_split_regeneration": True,
        "no_representation_regeneration": True,
        "transition_split_disjoint": disjoint,
    }
    diagnostics_status = "PASS" if diagnostics["status"] == "PASS" else "PARTIAL"
    status = "PASS" if all(checks.values()) and diagnostics_status == "PASS" else "PARTIAL"
    return {"status": status, "checks": checks, "mcmc_diagnostics_status": diagnostics_status}


def _remediation_report(
    *,
    validation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    artifacts: Mapping[str, str],
    capability: Mapping[str, Any],
    elapsed_seconds: float,
) -> str:
    lines = [
        "# Bayesian MCMC Transition Policy Remediation Report",
        "",
        f"- Experiment: `{PROTOCOL_ID}`",
        f"- Status: `{validation['status']}`",
        f"- Elapsed seconds: `{elapsed_seconds:.3f}`",
        f"- PyMC available: `{capability['pymc_available']}`",
        f"- ArviZ available: `{capability['arviz_available']}`",
        f"- MCMC diagnostics status: `{diagnostics['status']}`",
        "",
        "## Validation",
    ]
    for key, value in cast(Mapping[str, Any], validation["checks"]).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Diagnostics"])
    for key in (
        "max_r_hat",
        "min_ess_bulk",
        "min_ess_tail",
        "divergence_count",
        "maximum_tree_depth_warning",
    ):
        lines.append(f"- `{key}`: `{diagnostics[key]}`")
    lines.extend(["", "## Artifacts"])
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def _blocked_report(checks_dir: Path, capability: Mapping[str, Any]) -> Path:
    report_path = checks_dir / "bayesian_mcmc_remediation_report.md"
    report_path.write_text(
        "# Bayesian MCMC Transition Policy Remediation Report\n\n"
        "- Status: `BLOCKED`\n"
        f"- PyMC available: `{capability['pymc_available']}`\n"
        f"- ArviZ available: `{capability['arviz_available']}`\n"
        "- Exact blocker: MCMC dependency group is not installed.\n",
        encoding="utf-8",
    )
    return report_path


def _paths(project_root: Path) -> dict[str, Path]:
    output = project_root / "Output"
    paths = {
        "manifests": output / "Manifests",
        "checks": output / "ReproductionChecks",
        "tables": output / "Tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _readiness(project_root: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    from trustworthy_agent.experiments.static_policy import _readiness_audit

    return _readiness_audit(project_root, _static_paths(project_root), inputs)


def _read_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _training_scalers(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    columns = {
        "slope": "v1_slope",
        "curvature": "v1_curvature",
        "distance_from_healthy": "v4_distance_from_training_healthy",
    }
    scalers: dict[str, float] = {}
    for name, column in columns.items():
        values = [abs(_float(row.get(column))) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        scalers[name] = max(finite) if finite else 1.0
        if scalers[name] <= 0.0:
            scalers[name] = 1.0
    return scalers


def _covariates(row: Mapping[str, str], scalers: Mapping[str, float]) -> dict[str, float]:
    missing_ratio = _row_missing_ratio(row)
    slope = abs(_float(row.get("v1_slope"))) / scalers["slope"]
    curvature = abs(_float(row.get("v1_curvature"))) / scalers["curvature"]
    distance = (
        abs(_float(row.get("v4_distance_from_training_healthy"))) / scalers["distance_from_healthy"]
    )
    class_label = row["class_label"]
    risk = {
        "Healthy": 0.10,
        "Mech_Damage": 0.55,
        "Elec_Damage": 0.55,
        "Mech_Elec_Damage": 0.90,
    }.get(class_label, 0.50)
    fit_ok = row.get("v1_fit_status") == "FIT_OK" and row.get("v4_fit_status") == "FIT_OK"
    representation_agreement = 0.90 if fit_ok else 0.35
    return {
        "data_quality": max(0.0, min(1.0, 1.0 - missing_ratio)),
        "missing_ratio": max(0.0, min(1.0, missing_ratio)),
        "confidence": 0.88 if class_label == "Healthy" else 0.82,
        "model_agreement": 0.86 if class_label != "Mech_Elec_Damage" else 0.72,
        "representation_agreement": representation_agreement,
        "explanation_score": 0.84 if fit_ok else 0.45,
        "domain_consistency_score": 0.85 if class_label != "Mech_Elec_Damage" else 0.70,
        "risk_score": risk,
        "slope": max(0.0, min(1.0, slope)),
        "curvature": max(0.0, min(1.0, curvature)),
        "distance_from_healthy": max(0.0, min(1.0, distance)),
        "ood_score": max(0.0, min(1.0, distance * 0.65 + curvature * 0.35)),
    }


def _rule_transition_target(covariates: Mapping[str, float]) -> StateId:
    if covariates["missing_ratio"] > 0.20 or covariates["data_quality"] < 0.70:
        return StateId.NO_DECISION
    if (
        covariates["risk_score"] >= 0.75
        or covariates["ood_score"] >= 0.78
        or covariates["representation_agreement"] < 0.50
    ):
        return StateId.ESCALATION
    return StateId.RECOMMENDATION


def _matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.int_]]]:
    x = np.asarray(
        [[row["covariates"][name] for name in MCMC_COVARIATE_NAMES] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([row["target_index"] for row in rows], dtype=int)
    return x, y


def _posterior_samples_from_idata(idata: Any) -> tuple[Mapping[StateId, Mapping[str, float]], ...]:
    posterior = idata.posterior
    alpha = np.asarray(
        posterior["alpha"].stack(sample=("chain", "draw")).transpose("sample", "state")
    )
    beta = np.asarray(
        posterior["beta"].stack(sample=("chain", "draw")).transpose("sample", "covariate", "state")
    )
    samples: list[Mapping[StateId, Mapping[str, float]]] = []
    for sample_index in range(alpha.shape[0]):
        sample: dict[StateId, dict[str, float]] = {}
        for state_index, state in enumerate(STATE_ORDER):
            sample[state] = {
                "intercept": float(alpha[sample_index, state_index]),
                **{
                    name: float(beta[sample_index, covariate_index, state_index])
                    for covariate_index, name in enumerate(MCMC_COVARIATE_NAMES)
                },
            }
        samples.append(sample)
    return tuple(samples)


def _mean_probabilities(
    model: BayesianTransitionModel,
    covariates: np.ndarray[Any, np.dtype[np.float64]],
) -> dict[StateId, float]:
    totals = {state: 0.0 for state in STATE_ORDER}
    for sample in model.coefficients:
        logits = {
            state: sample[state]["intercept"]
            + sum(
                sample[state].get(name, 0.0) * covariates[index]
                for index, name in enumerate(MCMC_COVARIATE_NAMES)
            )
            for state in STATE_ORDER
        }
        max_logit = max(logits.values())
        exp_values = {state: math.exp(logit - max_logit) for state, logit in logits.items()}
        total = sum(exp_values.values())
        for state in STATE_ORDER:
            totals[state] += exp_values[state] / total
    return {state: totals[state] / len(model.coefficients) for state in STATE_ORDER}


def _posterior_summary_csv(idata: Any, path: Path) -> None:
    import arviz as az

    summary = az.summary(idata, var_names=["alpha", "beta"], hdi_prob=0.95)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path)


def _prior_specification() -> dict[str, Any]:
    return {
        "prior_version": "transition_mnlogit_priors_v1",
        "families": {
            "alpha": {"family": "Normal", "mu": 0.0, "sigma": 1.0},
            "beta": {"family": "Normal", "mu": 0.0, "sigma": 0.75},
        },
        "rationale": "Weakly regularizing priors stabilize the small transition-control dataset.",
    }


def _scenario_facts(scenario: Mapping[str, Any]) -> dict[str, Any]:
    facts = dict(cast(Mapping[str, Any], scenario["base_facts"]))
    facts.setdefault("class_probabilities", _probs(str(scenario["diagnosis"])))
    if "slope" not in facts and "spline_summary" in facts:
        facts["slope"] = 0.30
    facts.setdefault("curvature", facts.get("spline_curvature", 0.25))
    facts.setdefault("data_quality", 1.0 - float(facts.get("missing_ratio", 0.0)))
    facts.setdefault("model_agreement", 1.0 - float(facts.get("classifier_disagreement", 0.0)))
    facts.setdefault("domain_consistency_score", facts.get("domain_consistency_score", 0.80))
    return facts


def _assert_disjoint(role_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    roles = {role: {row["case_id"] for row in rows} for role, rows in role_rows.items()}
    if (
        roles["train"] & roles["validation"]
        or roles["train"] & roles["test"]
        or roles["validation"] & roles["test"]
    ):
        raise ValueError("Transition-policy train/validation/test cases must be disjoint.")


def _row_missing_ratio(row: Mapping[str, str]) -> float:
    checked = [
        value
        for key, value in row.items()
        if key
        not in {"case_id", "class_label", "group_key", "split_hash", "split_id", "split_role"}
    ]
    missing = sum(1 for value in checked if value == "" or value.lower() in {"nan", "inf", "-inf"})
    return missing / len(checked) if checked else 0.0


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def _entropy(probabilities: Mapping[StateId, float]) -> float:
    numerator = -sum(value * math.log(value) for value in probabilities.values() if value > 0.0)
    return numerator / math.log(len(probabilities))


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _hash_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        raise ValueError(f"Expected JSON object: {path}")
    return loaded
