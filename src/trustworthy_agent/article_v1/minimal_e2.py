# ruff: noqa: E501
"""Input boundary helpers for the ArticleV1 minimal real-agent execution.

This module deliberately contains no policy, FSM, or SafetyGuard logic.  It
only resolves the frozen natural-case manifest, loads one persisted
``PersistedEvidenceBundle`` per case, and attaches that bundle to the stable
``EvidenceAgentContext`` subtype.  Keeping this boundary small makes it hard
for an E2 runner to accidentally synthesize diagnostic facts or regenerate
evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from trustworthy_agent.agent.context import AgentContext
from trustworthy_agent.context.evidence import EvidenceAgentContext
from trustworthy_agent.evidence.bundles import PersistedEvidenceBundle

NATURAL_SCENARIOS = (
    "ARTICLE_SCENARIO_01_NATURAL_HEALTHY",
    "ARTICLE_SCENARIO_03_NATURAL_HIGH_RISK",
    "ARTICLE_SCENARIO_05_NATURAL_CONFLICT_CANDIDATE",
)
POLICY_IDS = ("Static", "Deterministic Approximation", "Bayesian MCMC", "Hybrid")
_MANIFEST_SCENARIO = {
    NATURAL_SCENARIOS[0]: "NATURAL_HEALTHY",
    NATURAL_SCENARIOS[1]: "NATURAL_HIGH_RISK",
    NATURAL_SCENARIOS[2]: "NATURAL_CONFLICT_CANDIDATE",
}


def sha256_file(path: Path) -> str:
    """Hash a frozen input without changing it."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class NaturalCase:
    """Manifest-selected natural case and its immutable source bundle."""

    scenario_id: str
    assignment_id: str
    acquisition_id: str
    classifier_id: str
    trend_model_id: str
    configured_diagnostic_facts: int
    selection_manifest_hash: str
    bundle_path: Path
    bundle: PersistedEvidenceBundle


def load_natural_cases(project_root: Path) -> tuple[NaturalCase, ...]:
    """Resolve exactly the three frozen cases and verify their bundle identity.

    No fallback case selection is permitted.  Missing or mismatched bundles
    raise ``ValueError`` so callers can fail closed before executing the agent.
    """

    manifest_path = (
        project_root / "Output/ArticleV1/Manifests/article_v1_natural_case_selection_manifest.json"
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = raw.get("records", [])
    if len(records) != len(NATURAL_SCENARIOS):
        raise ValueError("E2 natural-case manifest must contain exactly three records")
    manifest_hash = sha256_file(manifest_path)
    # JSON is the canonical case-level bundle representation.  The matching
    # Parquet file is a storage mirror and must not be treated as a second
    # scientific bundle identity.
    bundles = sorted((project_root / "Output/ArticleV1/EvidenceBundles").glob("*.bundle.json"))
    resolved: list[NaturalCase] = []
    for scenario in NATURAL_SCENARIOS:
        matching = [
            item for item in records if item.get("scenario_id") == _MANIFEST_SCENARIO[scenario]
        ]
        # Persisted E1.4 uses NATURAL_* identifiers; support an explicit direct
        # scenario match as well while never selecting a different acquisition.
        if not matching:
            suffix = scenario.split("_", 4)[-1]
            matching = [
                item for item in records if str(item.get("scenario_id", "")).endswith(suffix)
            ]
        if len(matching) != 1:
            raise ValueError(f"No unique frozen selection for {scenario}")
        record = matching[0]
        if int(record.get("configured_diagnostic_facts", -1)) != 0:
            raise ValueError(f"Configured diagnostic facts are nonzero for {scenario}")
        candidates: list[tuple[Path, PersistedEvidenceBundle]] = []
        for path in bundles:
            bundle = (
                PersistedEvidenceBundle.load_json(path)
                if path.suffix == ".json"
                else PersistedEvidenceBundle.load_parquet(path)
            )
            metadata = bundle.metadata
            if (
                metadata.assignment_id == record.get("assignment_id")
                and metadata.acquisition_id == record.get("acquisition_id")
                and metadata.classifier_id == record.get("classifier_id")
                and metadata.trend_model_id == record.get("trend_model_id")
            ):
                candidates.append((path, bundle))
        unique = {bundle.bundle_hash: (path, bundle) for path, bundle in candidates}
        if len(unique) != 1:
            raise ValueError(f"Expected one immutable bundle for {scenario}, found {len(unique)}")
        path, bundle = next(iter(unique.values()))
        resolved.append(
            NaturalCase(
                scenario,
                str(record["assignment_id"]),
                str(record["acquisition_id"]),
                str(record["classifier_id"]),
                str(record["trend_model_id"]),
                0,
                manifest_hash,
                path,
                bundle,
            )
        )
    return tuple(resolved)


def attach_bundle(case: NaturalCase, *, run_id: str, policy_id: str) -> EvidenceAgentContext:
    """Attach persisted evidence while leaving diagnostic fields unknown.

    The FSM's existing strategies may derive workflow facts from this context;
    this adapter itself never injects diagnosis, risk, confidence, OOD, or
    probabilities.
    """

    bundle = case.bundle
    base = AgentContext(
        run_id=run_id,
        experiment_id="DUDU_BLDC_TRUSTWORTHY_AGENT_ARTICLE_V1",
        experiment_fingerprint=f"article_v1_minimal_e2:{case.scenario_id}:{policy_id}",
        case_id=case.acquisition_id,
        dataset_id="DUDU-BLDC",
        dataset_version="v1",
        state_profile_id="diagnostic_full_v1",
        active_transition_policy=policy_id,
        derived_facts={
            "configured_diagnostic_facts": 0,
            "input_hash": bundle.bundle_hash,
            "bundle_hash": bundle.bundle_hash,
            "evidence_hashes": {
                "acquisition": bundle.evidence.acquisition_evidence.evidence_hash,
                "trend": bundle.evidence.trend_evidence.evidence_hash,
                "risk": bundle.evidence.risk_evidence.evidence_hash,
                "safety": list(bundle.evidence.safety_evidence.evidence_hashes),
            },
        },
        provenance={
            "selection_manifest_hash": case.selection_manifest_hash,
            "bundle_hash": bundle.bundle_hash,
        },
    )
    return EvidenceAgentContext.from_context(
        base,
        window_evidence=bundle.evidence.window_evidence,
        trend_evidence=bundle.evidence.trend_evidence,
        acquisition_evidence=bundle.evidence.acquisition_evidence,
        risk_evidence=bundle.evidence.risk_evidence,
        safety_evidence=bundle.evidence.safety_evidence,
        explanation_references=bundle.evidence.explanation_references,
        representation_metadata=bundle.metadata.to_dict(),
        model_metadata={
            "classifier_id": bundle.metadata.classifier_id,
            "classifier_hash": bundle.metadata.classifier_hash,
        },
        healthy_reference_metadata={
            "healthy_reference_hash": bundle.metadata.healthy_reference_hash
        },
    )
