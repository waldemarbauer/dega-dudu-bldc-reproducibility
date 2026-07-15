"""Figures G, H and I: DEGA execution outcomes, state paths, and audit/replay.

These figures describe 12 representative workflow executions. They are path
demonstrations, not a statistical benchmark, so no proportion is used as the
primary encoding. Two of the three scenario labels share one EvidenceBundle
hash; Figure G discloses this rather than concealing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _m3_common import (
    BLUE,
    NEUTRAL,
    ROOT,
    YELLOW,
    FigureRecord,
    attach_sources,
    export,
    read_csv,
    style_for,
    write_figure_data,
)

OUTCOMES = "Output/ArticleV1/Tables/minimal_e2_final_outcomes.csv"
SAFETY = "Output/ArticleV1/Tables/minimal_e2_safetyguard_summary.csv"
RUNS = "Output/ArticleV1/Tables/minimal_e2_run_summary.csv"
PATHS = "Output/ArticleV1/Tables/minimal_e2_state_paths.csv"
BUNDLES = "Output/ArticleV1/Tables/article_v1_bundle_coverage.csv"
AGENT_RUNS = ROOT / "Output/ArticleV1/AgentRuns/MinimalE2"

OUTCOME_SHORT = {"ESCALATION": "ESCALATION", "NO_AUTOMATED_RECOMMENDATION": "NO DECISION"}
STATE_SHORT = {
    "DATA_ACQUISITION": "Data\nacquisition",
    "DATA_VALIDATION": "Data\nvalidation",
    "FEATURE_EXTRACTION": "Feature\nextraction",
    "SPLINE_MODELLING": "Spline\nmodelling",
    "DIAGNOSTIC_INFERENCE": "Diagnostic\ninference",
    "EXPLANATION_GENERATION": "Explanation\ngeneration",
    "DECISION_CHECK": "Decision\ncheck",
    "ESCALATION": "Escalation",
    "NO_DECISION": "No\ndecision",
    "AUDIT": "Audit",
}


def build_fig_g(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_G"]
    scen = cfg["orderings"]["scenario"]
    pols = cfg["orderings"]["policy"]
    slab = cfg["orderings"]["scenario_label"]
    plab = cfg["orderings"]["policy_label"]

    outcomes = read_csv(OUTCOMES)
    safety = read_csv(SAFETY)
    runs = read_csv(RUNS)
    assert len(outcomes) == 12 and len(safety) == 12 and len(runs) == 12

    O = {(r["scenario_id"], r["policy_id"]): r for r in outcomes}
    S = {(r["scenario_id"], r["policy_id"]): r for r in safety}
    B = {r["scenario_id"]: r["bundle_hash"] for r in runs}

    # Verify the outcome tallies stated in the article.
    tally = {a: sum(1 for r in outcomes if r["final_action"] == a)
             for a in {r["final_action"] for r in outcomes}}
    assert tally.get("ESCALATION") == 3, tally
    assert tally.get("NO_AUTOMATED_RECOMMENDATION") == 9, tally
    assert "RECOMMENDATION" not in tally, "unexpected recommendation outcome"

    # Distinct EvidenceBundles behind the 3 scenario labels.
    uniq = sorted(set(B.values()))
    bundle_tag = {h: f"EB{i + 1}" for i, h in enumerate(uniq)}
    assert len(uniq) == 2, f"expected 2 distinct bundles, got {len(uniq)}"

    record = FigureRecord("FIG_G", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [OUTCOMES, SAFETY, RUNS])
    record.display_transforms = [
        "pivot of 12 runs to a scenario x policy matrix (reshape only)",
        "state-path length = number of states in the persisted state_path string",
    ]

    plotted: list[dict] = []
    with style_for(spec["export_profile"]):
        fig, ax = plt.subplots()
        for i, s in enumerate(scen):
            for j, p in enumerate(pols):
                r, sg = O[(s, p)], S[(s, p)]
                action = r["final_action"]
                ov = int(sg["override_count"])
                plen = len(r["state_path"].split(">"))
                esc = action == "ESCALATION"
                # Fill is a redundant cue only; the outcome is written in the cell.
                ax.add_patch(plt.Rectangle((j - 0.46, i - 0.34), 0.92, 0.78,
                                           facecolor=BLUE["light"] if esc else NEUTRAL["near_white"],
                                           edgecolor=NEUTRAL["light"], linewidth=0.6, zorder=1))
                ax.annotate(OUTCOME_SHORT[action], xy=(j, i - 0.13), ha="center", va="center",
                            fontsize=7.0, color=NEUTRAL["near_black"],
                            fontweight="bold" if esc else "normal", zorder=3)
                sg_txt = f"{ov} SafetyGuard override" if ov else "no override"
                ax.annotate(f"{plen} states", xy=(j, i + 0.14), ha="center", va="center",
                            fontsize=6.0, color=NEUTRAL["dark"], zorder=3)
                ax.annotate(sg_txt, xy=(j, i + 0.29), ha="center", va="center",
                            fontsize=5.6, color=NEUTRAL["dark"], zorder=3)
                plotted.append({"figure": "FIG_G", "scenario_id": s, "policy_id": p,
                                "final_action": action, "override_count": ov,
                                "state_path_length": plen, "bundle_hash": B[s]})

        ax.set_xlim(-0.6, len(pols) - 0.4)
        ax.set_ylim(len(scen) - 0.52, -0.55)
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([plab[p].replace(" ", "\n", 1) for p in pols], fontsize=6.6)
        ax.set_yticks(range(len(scen)))
        # Scenario label carries its EvidenceBundle identity: S01 and S05 share one.
        ax.set_yticklabels(
            [f"{slab[s]}\n[{bundle_tag[B[s]]}]" for s in scen], fontsize=6.8
        )
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        # The n = 12 caveat and the EB1/EB2 shared-bundle explanation are in the
        # caption; the EB tags stay on the axis as compact data labels.
        fig.subplots_adjust(bottom=0.08, left=0.22, right=0.985, top=0.82)

    write_figure_data("FIG_G", plotted, record)
    record.checks = {
        "runs": 12, "scenarios": 3, "policies": 4,
        "escalations": 3, "no_decision": 9, "recommendations": 0,
        "distinct_bundle_hashes": len(uniq),
        "shared_bundle_scenarios": [s for s in scen if B[s] == uniq[0]],
    }
    return fig, record


def build_fig_h(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_H"]
    pols = cfg["orderings"]["policy"]
    plab = cfg["orderings"]["policy_label"]
    paths = read_csv(PATHS)
    assert len(paths) == 12

    # Enumerate the distinct realized paths; there are only two.
    distinct: dict[str, list[str]] = {}
    for r in paths:
        distinct.setdefault(r["state_path"], []).append(r["policy_id"])
    assert len(distinct) == 2, f"expected 2 distinct paths, got {len(distinct)}"

    ordered = sorted(distinct, key=lambda p: -len(p.split(">")))
    maxlen = max(len(p.split(">")) for p in ordered)

    record = FigureRecord("FIG_H", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [PATHS])
    record.display_transforms = ["deduplication of the 12 persisted state paths to their 2 distinct sequences"]

    plotted: list[dict] = []
    with style_for(spec["export_profile"]):
        fig, ax = plt.subplots()
        for i, path in enumerate(ordered):
            states = [s.strip() for s in path.split(">")]
            policies = sorted(set(distinct[path]))
            runs_n = len(distinct[path])
            terminal = "ESCALATION" if "ESCALATION" in states else "NO_DECISION"
            color = BLUE["dark"] if terminal == "ESCALATION" else NEUTRAL["mid"]
            ls = "-" if terminal == "ESCALATION" else "--"
            ax.plot(range(len(states)), [i] * len(states), color=color, ls=ls, lw=1.0, zorder=2)
            for k, s in enumerate(states):
                is_term = s in ("ESCALATION", "NO_DECISION")
                ax.scatter([k], [i], s=54 if is_term else 30,
                           marker="D" if is_term else "o",
                           facecolor=color if is_term else NEUTRAL["white"],
                           edgecolor=color, linewidth=0.9, zorder=4)
                ax.annotate(STATE_SHORT[s], xy=(k, i - 0.19), ha="center", va="bottom",
                            fontsize=5.6, color=NEUTRAL["near_black"], zorder=5)
                plotted.append({"figure": "FIG_H", "path_index": i, "step": k, "state_id": s,
                                "terminal": terminal, "runs_realizing": runs_n})
            # Compact data label only; which policies realize each path is in the caption.
            ax.annotate(f"{runs_n} of 12 runs", xy=(-0.42, i + 0.30), ha="left", va="center",
                        fontsize=6.3, color=NEUTRAL["dark"], zorder=5)

        ax.set_xlim(-0.5, maxlen - 0.5)
        ax.set_ylim(len(ordered) - 0.45, -0.55)
        ax.set_xlabel("Step in the deterministic state sequence")
        ax.set_xticks(range(maxlen))
        ax.set_yticks([])
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)

        # Marker/line-style meanings and the "2 distinct paths" statement are in the caption.
        fig.subplots_adjust(left=0.04, right=0.985, bottom=0.20, top=0.95)

    write_figure_data("FIG_H", plotted, record)
    record.checks = {"runs": 12, "distinct_paths": 2,
                     "path_lengths": [len(p.split(">")) for p in ordered]}
    return fig, record


def build_fig_i(cfg: dict) -> tuple[plt.Figure, FigureRecord]:
    spec = cfg["figures"]["FIG_I"]

    # Re-verify each integrity claim directly from the frozen artifacts rather
    # than restating a number from a report.
    bundles = read_csv(BUNDLES)
    runs = read_csv(RUNS)
    bundles_ok = sum(1 for r in bundles if r["status"] == "COMPLETE_RELOAD_VERIFIED")
    audit_ok = sum(1 for r in runs if r["audit_valid"] == "True")

    chains_ok, events = 0, 0
    for d in sorted(AGENT_RUNS.iterdir()):
        ev = [json.loads(line) for line in (d / "audit.jsonl").open()]
        events += len(ev)
        prev = None
        linked = True
        for e in ev:
            if e["previous_event_hash"] != prev:
                linked = False
            prev = e["event_hash"]
        chains_ok += linked
    n_runs = len(list(AGENT_RUNS.iterdir()))

    # Deterministic replay is an executed ArticleV1 result; read its verdict.
    replay_txt = (ROOT / "Output/ArticleV1/Reports/minimal_e2_replay_report.md").read_text()
    assert "PASS (12/12)" in replay_txt, "replay report verdict changed"

    items = [
        ("EvidenceBundle\nreconstructions", bundles_ok, len(bundles)),
        ("Audit-chain hash\nlinkage (re-verified)", chains_ok, n_runs),
        ("Audit-chain validity\n(persisted flag)", audit_ok, len(runs)),
        ("Deterministic\nreplay", 12, 12),
    ]
    assert (bundles_ok, len(bundles)) == (768, 768)
    assert (chains_ok, n_runs) == (12, 12)

    record = FigureRecord("FIG_I", spec["title"], spec["export_profile"], 0, 0)
    attach_sources(record, [BUNDLES, RUNS, "Output/ArticleV1/Reports/minimal_e2_replay_report.md",
                            "Output/ArticleV1/Reports/article_v1_evidence_bundle_validation.md"])
    record.display_transforms = [
        "verified/total counts tallied from status columns",
        "audit hash-chain linkage independently recomputed from audit.jsonl",
    ]

    plotted: list[dict] = []
    with style_for(spec["export_profile"]):
        fig, ax = plt.subplots()
        for i, (label, ok, tot) in enumerate(items):
            frac = ok / tot
            ax.barh(i, 1.0, height=0.52, color=NEUTRAL["near_white"],
                    edgecolor=NEUTRAL["light"], linewidth=0.6, zorder=2)
            ax.barh(i, frac, height=0.52, color=BLUE["dark"],
                    edgecolor=NEUTRAL["near_black"], linewidth=0.5, zorder=3)
            ax.annotate(f"{ok}/{tot}", xy=(frac, i), xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7.2, color=NEUTRAL["near_black"],
                        fontweight="bold", zorder=4)
            plotted.append({"figure": "FIG_I", "check": label.replace("\n", " "),
                            "verified": ok, "total": tot, "fraction": frac})

        ax.set_xlim(0, 1.30)
        ax.set_ylim(len(items) - 0.4, -0.6)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([i[0] for i in items], fontsize=6.4)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50%", "100%"])
        ax.set_xlabel("Verified fraction")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # Audit-event count and the integrity-vs-accuracy caveat are in the caption.
        fig.subplots_adjust(left=0.36, right=0.96, top=0.96, bottom=0.17)

    write_figure_data("FIG_I", plotted, record)
    record.checks = {"bundles": [bundles_ok, len(bundles)], "chains": [chains_ok, n_runs],
                     "audit_valid": [audit_ok, len(runs)], "replay": [12, 12],
                     "audit_events": events}
    return fig, record
