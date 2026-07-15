# Machine-readable experiment summary

The authoritative expected article-level contract is
`reference/expected_article_results.json`. The stages below are executable and
map directly to the manuscript case-study pipeline.

| ID | Experiment | Input unit | Output |
|---|---|---|---|
| E1.2 | Canonical window prerequisite | 8 raw acquisitions | 200 windows, 28 features, 16 assignment views |
| E1.3 | Window classifiers | 100 train windows per assignment | 48 models, 4,800 held-out predictions, 192 acquisition aggregates |
| E1.4 | Evidence generation | frozen predictions + ordered windows | deviation, Healthy references, temporal evidence, 768 bundles |
| X1 | Minimal explainability | persisted models and held-out data | permutation importance, exact LR contributions, stability |
| E2 | Representative DEGA execution | persisted EvidenceBundles | 12 FSM/policy/SafetyGuard/audit/replay runs |

Scientific unit hierarchy:

- independent source unit: raw acquisition;
- diagnostic observation: canonical 0.8-s window;
- temporal evidence unit: ordered windows within one acquisition;
- primary evaluation unit: held-out acquisition.

The 16 views and the windows within an acquisition are dependent.
