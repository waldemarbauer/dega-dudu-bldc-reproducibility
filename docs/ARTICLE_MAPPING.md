# Code-to-manuscript mapping

This mapping targets the **July 15, 2026** manuscript version titled
*DEGA: A Deterministic Diagnostic Evidence Governance Agent for Industrial
IoT—A DUDU-BLDC Case Study*.

## Sections

| Manuscript section | Reproduction source |
|---|---|
| 3.1 Data Units and Feature Representation | `scripts/01_prepare_windows.py`, `trustworthy_agent.article_v1.features`, frozen ArticleV1 protocol/config |
| 3.2 Temporal Spline Evidence | Stage 3, `trustworthy_agent.trends`, `TrendEvidence` |
| 3.3 Training-Only References and Deviation Evidence | Stage 3, deviation scorer and Healthy-reference artifacts |
| 3.4 Immutable Evidence Hierarchy | `trustworthy_agent.evidence`, Stage 3 |
| 4 DEGA Architecture and Deterministic Workflow | `trustworthy_agent.agent`, `transitions`, `safety`, `audit` |
| 5.1 Acquisition-Disjoint Assignments | Stage 1 |
| 5.2 Classifier and Aggregation Protocol | Stage 2 |
| 5.3 Temporal, Reference, Risk, and Explanation Evidence | Stages 3–4 |
| 5.4 Representative DEGA Executions | Stage 5 |
| 6 Results | Stages 2–6 |

## Computational figures in the supplied manuscript

The final plotting pipeline under `figures_m3/` maps to the paper as follows:

| Paper figure | M3 ID | Source |
|---|---|---|
| Figure 6 — Acquisition-level classifier performance | FIG_A | Stage 2 tables |
| Figure 7 — Window vs acquisition evaluation | FIG_B | Stage 2 tables |
| Figure 8 — Scalar spline evidence | FIG_C | Stage 3 / intermediate figure-data |
| Figure 9 — Risk component resolution and saturation | FIG_D | Stage 3 risk evidence |
| Figure 10 — Global permutation importance | FIG_E | Stage 4 XAI |
| Figure 11 — Explanation stability | FIG_F | Stage 4 XAI |
| Figure 12 — Terminal outcomes of 12 DEGA runs | FIG_G | Stage 5 |
| Figure 13 — Two realized state paths | FIG_H | Stage 5 |
| Figure 14 — Integrity and deterministic replay | FIG_I | Stages 3 and 5 |

Figures 1–5 in the manuscript are conceptual/architectural diagrams rather than
statistical plots from the experiment pipeline.

## Article-level expected results

See `reference/expected_article_results.json` for the machine-readable counts,
key metrics and interpretation limits used by this repository.
