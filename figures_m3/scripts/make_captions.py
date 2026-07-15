"""Emit the Markdown and LaTeX caption files for the M3 computational figures.

Both formats are generated from one source of truth so they cannot drift apart.
Every quantity stated in a caption was verified against the frozen ArticleV1
artifact during the M3 audit; nothing here is inferred.

Run:  python figures_m3/scripts/make_captions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _m3_common import ROOT  # noqa: E402

CAPTIONS: dict[str, dict[str, str]] = {
    "A": {
        "short": "Acquisition-level classifier performance across 16 dependent assignment views.",
        "body": """Acquisition-level performance of the three ArticleV1 classifiers on the
DUDU-BLDC case study, from the frozen table
Output/ArticleV1/Tables/article_acquisition_metrics_by_assignment.csv.
The experimental unit is the acquisition. Left panel: macro-F1. Right panel:
balanced accuracy; both are dimensionless and bounded in [0, 1] and share a
common horizontal scale. Each mark is one of the 16 acquisition-level assignment
views, and marks are stacked vertically where views take exactly the same value,
so all 16 are visible. Marker shape and fill encode the classifier redundantly
with position (circle: logistic regression; square: random forest; triangle:
histogram gradient boosting), so the figure remains readable in grayscale. The
black vertical rule is the median across the 16 views, repeated numerically at
the right of each row. Logistic regression attains the highest median
acquisition-level macro-F1 (1.00) against 0.67 for both random forest and
histogram gradient boosting.
Interpretation limits: the 16 assignment views are dependent exhaustive views of
the same 8 source acquisitions, not 16 independent experiments, and each view
scores only 4 held-out acquisitions. The spread across views is therefore a
descriptive range over dependent views, not a sampling error bar, and no
confidence statement may be derived from it.""",
    },
    "B": {
        "short": "Window-level versus acquisition-level evaluation of the same fitted models.",
        "body": """Effect of the evaluation unit on reported performance, from the frozen tables
Output/ArticleV1/Tables/article_window_metrics_by_assignment.csv and
article_acquisition_metrics_by_assignment.csv. One panel per classifier
(left to right: logistic regression, random forest, histogram gradient boosting).
The vertical axis is macro-F1 (dimensionless, [0, 1]), shared across panels. Each
thin line joins the window-level and acquisition-level macro-F1 of one of the 16
assignment views; the pairing is exact, because both values come from the same
fitted model in the same view. Markers repeat the classifier encoding used
throughout the figure set. The thick black line joins the medians over the 16
views, printed at each end. Window-level medians cluster near 0.72-0.75 for all
three classifiers, whereas acquisition-level medians separate them (1.00 for
logistic regression, 0.67 for the other two), so the choice of evaluation unit
changes both the level and the ranking.
Interpretation limits: each view supplies 100 held-out windows drawn from only 4
held-out acquisitions, out of 8 source acquisitions and 200 windows in total.
Windows within an acquisition are correlated and are not independent test
objects; window-level scores must not be read as independent test performance.
The 16 views are dependent exhaustive views, not replications.""",
    },
    "C": {
        "short": "The spline (trend) evidence persisted by ArticleV1 is scalar-only.",
        "body": """Complete inventory of the spline evidence that the frozen ArticleV1 pipeline
persists. Left panel: every one of the 256 persisted trend records
(16 assignment views x 4 held-out acquisitions x 4 trend representations),
showing the scalar trend value for each of the four rolling-spline
representations, 64 records each, from
Output/ArticleV1/Figures/v1/Figure_05_v6_temporal_spline_comparison_data.csv.
The horizontal axis is symmetric-log because the values span several orders of
magnitude and take both signs; the vertical position within a row is jitter for
visibility only and carries no meaning. Right panel: the five scalar descriptors
persisted for a single record (assignment view A05, acquisition healthy_1,
rolling smoothing spline), from
Output/ArticleV1/Figures/v1/Figure_06_v6_derivatives_curvature_data.csv, plotted
on a symmetric-log axis with values printed directly; the yellow diamond marks
the one negative value. These five numbers are the entire spline evidence for
that record.
Interpretation limits: the ArticleV1 TrendEvidence record
(src/trustworthy_agent/evidence/trend.py) is a scalar-only structure. No ordered
diagnostic feature sequence, spline evaluation grid, first- or second-derivative
array, curvature array, Healthy reference profile or uncertainty band was
persisted, and Output/ArticleV1/Results/TrendEvidence is absent from the
repository. A fitted-trajectory figure therefore cannot be produced without
refitting the splines, which would constitute a new experiment; this figure is
deliberately reduced to what exists. Derivatives are taken with respect to
ordered window position, not physical time: they are not degradation rates or
accelerations. Maximum curvature indicates the strongest local geometric change
in the ordered sequence, not physical fault onset, and no remaining-useful-life
interpretation is supported.""",
    },
    "D": {
        "short": "Resolution and saturation of the four DEGA risk components and the combined risk.",
        "body": """Distribution of the evidence components that DEGA combines into a risk score,
over all 768 frozen risk-evidence records (16 dependent assignment views x 4
held-out acquisitions x 3 classifiers x 4 trend representations), from
Output/ArticleV1/Figures/v1/Figure_10_combined_risk_acquisitions_data.csv.
Left panel: empirical cumulative distribution function of each component. All
components are dimensionless and bounded in [0, 1]. Line style and colour encode
the component redundantly (see legend), and each legend entry reports the
percentage of records sitting exactly at the upper bound 1.0, marked by the
yellow vertical line. Right panel: the number of distinct values each component
takes across the 768 records, on a logarithmic axis; yellow marks components with
20 or fewer distinct values.
Main result: only classifier uncertainty (192 distinct values, 0% at the bound)
and the combined risk (408 distinct values) resolve cases. The OOD contribution
is saturated - 75% of records sit exactly at 1.0 and the component takes only 6
distinct values in total, all within [0.94, 1.00] - and trend uncertainty is
almost as degenerate (72% at 1.0, 16 distinct values). Healthy deviation is
intermediate (25% at 1.0, 11 distinct values).
Interpretation limits: the saturation is displayed exactly as persisted. No
weight, scale or threshold was retuned and no component was omitted. The combined
risk is a descriptive aggregate, not a probability of failure. The 768 records
are a full crossing of dependent views and are not independent observations, so
the CDFs describe the frozen record set rather than a sampling distribution.""",
    },
    "E": {
        "short": "Global permutation importance for the executed random-forest model in assignment view A00.",
        "body": """Global permutation importance, the executed ArticleV1 global explanation, from
Output/ArticleV1/XAI/Minimal/GlobalPermutationImportance/distributions.csv and
repeat_distributions.csv. All 28 diagnostic features of the ArticleV1 feature
vector are shown, ordered by persisted mean importance. The horizontal axis is
the drop in balanced accuracy when the feature is permuted (dimensionless; larger
means more important, and a value at or below zero means permuting the feature did
not degrade the score). Dark circles are the persisted mean importance over 30
permutation repeats; pale dots behind them are the 30 individual repeats, showing
the spread actually produced by the executed analysis. Speed-channel amplitude
features (speed RMS, speed mean, speed variance) lead the ranking, but the largest
mean drop is only about 0.06 balanced-accuracy points, and most features are
indistinguishable from zero.
Interpretation limits: this ranking is specific to the random-forest model in
assignment view A00, scored by balanced accuracy on that view's 100 correlated
held-out windows. It is a case-specific ranking, not a universal or physical
feature ranking, and Figure F shows that the top-10 set is unstable across
assignment views. The 28 features are the complete implemented set; the 1x, 2x
and 3x harmonic-amplitude features appear in the feature dictionary only as
NOT_IMPLEMENTED_FOR_ARTICLE_V1 (rotational frequency was unavailable for all
windows) and are correctly absent here.""",
    },
    "F": {
        "short": "Instability of the top-10 explanation across the 16 dependent assignment views.",
        "body": """Stability of the global explanation across assignment views, from
Output/ArticleV1/XAI/Minimal/ExplanationStability/jaccard_matrix.csv. For each
pair of assignment views, the top-10 permutation-importance feature sets are
compared by Jaccard similarity (k = 10; the size of the intersection divided by
the size of the union of the two feature sets; 1.0 means the two top-10 sets are
identical, 0 means they are disjoint). Left panel: the full 16 x 16 symmetric
matrix of pairwise Jaccard values; darker blue is higher similarity, and the dark
diagonal is trivial self-similarity (1.0). Right panel: the distribution of the
120 unique off-diagonal pairs as a dot histogram, one dot per pair; the black rule
marks the median and the yellow line marks the value 1.0 that identical top-10
sets would attain.
Main result: agreement between views is weak. The median pairwise Jaccard
similarity is 0.33, the range is 0.05 to 0.82, and no pair of views produces an
identical top-10 set. Two views typically share only about 3 of their 10
most-important features.
Interpretation limits: the 16 views are dependent exhaustive views of the same 8
source acquisitions, each trained on only 4 acquisitions. Low similarity therefore
exposes instability of the explanation under very small acquisition-level training
sets; it must not be read as variation across independent datasets.""",
    },
    "G": {
        "short": "Terminal outcome of all 12 representative DEGA executions, by scenario and transition policy.",
        "body": """Terminal outcome of every representative DEGA workflow execution in the frozen
minimal end-to-end study, from Output/ArticleV1/Tables/minimal_e2_final_outcomes.csv,
minimal_e2_safetyguard_summary.csv and minimal_e2_run_summary.csv. The matrix
crosses the 3 article scenarios (rows) with the 4 transition policies (columns);
each of the 12 cells is one complete execution. The terminal outcome is written in
the cell rather than encoded by colour alone; the blue fill is a redundant cue for
escalation. Below each outcome, the cell reports the number of states in the
persisted state path and whether SafetyGuard overrode a proposed transition.
Main result: all 12 executions terminate without an automated recommendation. The
static transition-matrix policy escalates in all 3 scenarios, each time after a
9-state path and exactly 1 SafetyGuard override; the other three policies return
NO_AUTOMATED_RECOMMENDATION after a 4-state path with no override. The terminal
outcome is therefore determined by the transition policy alone and does not vary
with the scenario. There are 3 escalations, 9 no-decision outcomes and 0
recommendations.
Interpretation limits: n = 12 is a set of representative workflow executions, or
path demonstrations, not a statistical benchmark, and no proportion derived from it
may be read as a performance estimate. The bracketed tags EB1 and EB2 are the
distinct EvidenceBundle hashes underlying the rows: scenarios S01 (natural healthy)
and S05 (natural conflict candidate) are evaluated on the *same* EvidenceBundle
(EB1), so the 3 scenario labels rest on only 2 distinct evidence bundles, and the
S01 and S05 rows are not independent cases.""",
    },
    "H": {
        "short": "The two distinct state paths realized by the 12 DEGA executions.",
        "body": """The complete set of deterministic state paths realized in the frozen minimal
end-to-end study, from Output/ArticleV1/Tables/minimal_e2_state_paths.csv and the
per-run hash-chained logs in Output/ArticleV1/AgentRuns/MinimalE2/*/audit.jsonl.
The horizontal axis is the step index in the deterministic state sequence. Each
row is one distinct path, drawn in full; open circles are intermediate states and
filled diamonds are terminal states. The solid dark line is the escalation path and
the dashed grey line is the no-decision path, so the two are distinguishable
without colour. The annotation under each path states how many of the 12 runs
realize it and which policies produce it.
Main result: the 12 executions collapse onto only 2 distinct paths. The static
transition-matrix policy traverses the full 9-state path (data acquisition, data
validation, feature extraction, spline modelling, diagnostic inference, explanation
generation, decision check, escalation, audit) in 3 of 12 runs. The remaining 9
runs - the deterministic posterior-approximation, Bayesian MCMC and hybrid policies
across all 3 scenarios - short-circuit after data validation onto a 4-state path
(data acquisition, data validation, no decision, audit), never reaching feature
extraction, spline modelling, inference or explanation.
Interpretation limits: because only 2 distinct paths exist, this figure enumerates
them exhaustively rather than summarising a distribution over paths. n = 12 is not
a benchmark.""",
    },
    "I": {
        "short": "Evidence hashing, audit-chain integrity and deterministic replay in ArticleV1.",
        "body": """Verification status of the ArticleV1 reproducibility mechanisms. Bars show the
verified fraction of each check; the exact verified/total counts are printed at the
right of each bar. EvidenceBundle reconstructions: 768 of 768 bundles carry status
COMPLETE_RELOAD_VERIFIED in
Output/ArticleV1/Tables/article_v1_bundle_coverage.csv, i.e. every bundle was
rebuilt, persisted, reloaded and checksum-verified. Audit-chain hash linkage: for
this figure the hash chain of each run was recomputed independently from
Output/ArticleV1/AgentRuns/MinimalE2/*/audit.jsonl by checking that every event's
previous_event_hash equals the preceding event's event_hash; all 12 chains link
without a break across 114 audit events (51 state executions, 51 committed
transitions, 12 audit finalisations). Audit-chain validity: the persisted
audit_valid flag is true for all 12 runs in
Output/ArticleV1/Tables/minimal_e2_run_summary.csv. Deterministic replay: the
frozen replay report Output/ArticleV1/Reports/minimal_e2_replay_report.md records
PASS for 12 of 12 runs.
Interpretation limits: these are integrity checks over frozen artifacts. They
establish that the recorded computation is complete, hash-linked and exactly
reproducible; they say nothing about diagnostic accuracy, and they are not evidence
that DEGA's decisions are correct.""",
    },
}

TEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}


def to_tex(text: str) -> str:
    for a, b in TEX_ESCAPES.items():
        text = text.replace(a, b)
    return text.replace("\n", " ").replace("  ", " ").strip()


def main() -> int:
    out = ROOT / "captions"
    out.mkdir(exist_ok=True)
    for fid, cap in CAPTIONS.items():
        body = cap["body"].strip()
        md = out / f"Figure_{fid}.md"
        md.write_text(
            f"# Figure {fid}\n\n**{cap['short']}**\n\n{body}\n\n"
            f"---\n\n"
            f"- Figure files: `figures_m3/output/pdf/Figure_{fid}.pdf`, "
            f"`figures_m3/output/png/Figure_{fid}.png`, `figures_m3/output/svg/Figure_{fid}.svg`\n"
            f"- Plotted values: `figures_m3/output/figure_data/FIG_{fid}_data.csv`\n"
            f"- Provenance: `figures_m3/data_manifest/figure_manifest.yaml`\n"
            f"- Scientific transform applied: none (frozen ArticleV1 outputs)\n"
        )
        tex = out / f"Figure_{fid}.tex"
        tex.write_text(
            f"% Caption for Figure {fid} (DEGA M3 computational figure set).\n"
            f"% Generated by figures_m3/scripts/make_captions.py -- do not edit by hand.\n"
            f"\\caption{{\\textbf{{{to_tex(cap['short'])}}} {to_tex(body)}}}\n"
            f"\\label{{fig:m3_{fid.lower()}}}\n"
        )
        print(f"[ok] captions/Figure_{fid}.md, captions/Figure_{fid}.tex")
    print(f"\n{len(CAPTIONS)} captions written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
