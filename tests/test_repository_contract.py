from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_article_protocol_contract() -> None:
    cfg = yaml.safe_load(
        (ROOT / "configs/experiments/article_v1/article_window_analysis_v1.yaml").read_text()
    )
    assert len(cfg["raw_acquisitions"]) == 8
    assert cfg["dataset"]["canonical_window_count"] == 200
    assert cfg["dataset"]["windows_per_acquisition"] == 25
    assert cfg["assignments"]["assignment_ids"] == [f"A{i:02d}" for i in range(16)]


def test_feature_contract_is_28_and_harmonics_are_not_fabricated() -> None:
    cfg = yaml.safe_load(
        (ROOT / "configs/features/article_v1/canonical_window_features_v1.yaml").read_text()
    )
    assert len(cfg["feature_order"]) == 28
    assert cfg["harmonic_semantics"]["status"] == "NOT_IMPLEMENTED_FOR_ARTICLE_V1"


def test_reproduction_stage_scripts_exist() -> None:
    for i in range(7):
        assert list((ROOT / "scripts").glob(f"{i:02d}_*.py"))


def test_publication_routing_snapshot_is_normalized() -> None:
    payload = json.loads(
        (
            ROOT
            / "reference_inputs/transition_policies/article_v1_routing_snapshot.json"
        ).read_text()
    )
    assert set(payload["policies"]) == {
        "BayesianMCMCTransitionPolicy",
        "HybridTransitionPolicy",
    }
    for record in payload["policies"].values():
        for scores in record["routing"].values():
            assert abs(sum(scores.values()) - 1.0) < 1e-9


def test_expected_article_contract() -> None:
    expected = json.loads((ROOT / "reference/expected_article_results.json").read_text())
    assert expected["classifier_contract"]["models"] == 48
    assert expected["evidence_contract"]["complete_evidence_bundles"] == 768
    assert expected["dega_contract"]["representative_runs"] == 12
    assert expected["dega_contract"]["distinct_underlying_evidence_bundles_for_scenarios"] == 2
