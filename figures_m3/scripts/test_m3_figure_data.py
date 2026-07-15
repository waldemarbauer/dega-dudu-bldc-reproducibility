"""Lightweight data-integrity tests for the M3 figures.

Each test re-derives a plotted quantity directly from the frozen ArticleV1 source
and asserts it matches the exported figure_data, guarding against silent drift or
an accidental transformation. Run:

    python figures_m3/scripts/test_m3_figure_data.py     # standalone
    pytest figures_m3/scripts/test_m3_figure_data.py     # under pytest
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _m3_common import ROOT, read_csv  # noqa: E402

FD = ROOT / "figures_m3/output/figure_data"


def _fig(name: str) -> list[dict[str, str]]:
    with (FD / name).open() as fh:
        return list(csv.DictReader(fh))


def test_fig_a_values_match_source() -> None:
    src = read_csv("Output/ArticleV1/Tables/article_acquisition_metrics_by_assignment.csv")
    want = {(r["assignment_id"], r["classifier_id"], "macro_f1"): float(r["macro_f1"]) for r in src}
    want |= {(r["assignment_id"], r["classifier_id"], "balanced_accuracy"): float(r["balanced_accuracy"]) for r in src}
    got = _fig("FIG_A_data.csv")
    assert len(got) == 96, len(got)  # 48 rows x 2 metrics
    for r in got:
        key = (r["assignment_id"], r["classifier_id"], r["metric"])
        assert abs(float(r["value"]) - want[key]) < 1e-12, key


def test_fig_b_pairing_is_exact() -> None:
    win = {(r["assignment_id"], r["classifier_id"]): float(r["macro_f1"])
           for r in read_csv("Output/ArticleV1/Tables/article_window_metrics_by_assignment.csv")}
    acq = {(r["assignment_id"], r["classifier_id"]): float(r["macro_f1"])
           for r in read_csv("Output/ArticleV1/Tables/article_acquisition_metrics_by_assignment.csv")}
    got = _fig("FIG_B_data.csv")
    assert len(got) == 96
    for r in got:
        key = (r["assignment_id"], r["classifier_id"])
        ref = win[key] if r["level"] == "window" else acq[key]
        assert abs(float(r["macro_f1"]) - ref) < 1e-12, key


def test_fig_c_is_scalar_only() -> None:
    got = _fig("FIG_C_data.csv")
    a = [r for r in got if r["panel"] == "a"]
    b = [r for r in got if r["panel"] == "b"]
    assert len(a) == 256, len(a)
    assert {r["quantity"] for r in b} == {
        "trend_value", "first_derivative", "second_derivative", "curvature", "maximum_curvature"
    }


def test_fig_d_saturation_counts() -> None:
    got = [r for r in _fig("FIG_D_data.csv") if r["panel"] == "a"]
    assert len(got) == 768 * 5
    ood = [float(r["value"]) for r in got if r["component"] == "ood_contribution"]
    frac = sum(v == 1.0 for v in ood) / len(ood)
    assert 0.74 < frac < 0.76, frac  # 75% saturated, as audited


def test_fig_e_no_harmonics_and_28_features() -> None:
    got = _fig("FIG_E_data.csv")
    assert len(got) == 28
    assert not any("harmonic" in r["feature"] for r in got)


def test_fig_f_pairs_and_symmetry() -> None:
    got = [r for r in _fig("FIG_F_data.csv") if r["panel"] == "a"]
    assert len(got) == 256
    m = {(r["assignment_a"], r["assignment_b"]): float(r["jaccard"]) for r in got}
    for (a, b), v in m.items():
        assert abs(v - m[(b, a)]) < 1e-12
        if a == b:
            assert abs(v - 1.0) < 1e-12


def test_fig_g_outcome_tally() -> None:
    got = _fig("FIG_G_data.csv")
    assert len(got) == 12
    tally: dict[str, int] = {}
    for r in got:
        tally[r["final_action"]] = tally.get(r["final_action"], 0) + 1
    assert tally == {"ESCALATION": 3, "NO_AUTOMATED_RECOMMENDATION": 9}, tally
    assert len({r["bundle_hash"] for r in got}) == 2


def test_fig_h_two_paths() -> None:
    got = _fig("FIG_H_data.csv")
    assert len({r["path_index"] for r in got}) == 2


def test_fig_i_counts() -> None:
    got = {r["check"]: (int(r["verified"]), int(r["total"])) for r in _fig("FIG_I_data.csv")}
    assert ("EvidenceBundle reconstructions") in got
    assert got["EvidenceBundle reconstructions"] == (768, 768)
    assert all(v == t for v, t in got.values()), got


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[pass] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
