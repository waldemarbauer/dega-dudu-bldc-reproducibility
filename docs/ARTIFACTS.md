# Artifact guide

## Generated data

- `Data/AnalysisData/V2/Windows/window_provenance.parquet`: 200 canonical window identities and raw hashes.
- `Data/AnalysisData/ArticleV1/WindowFeatures/article_v1_canonical_window_features.parquet`: 28-feature corpus.
- `Data/AnalysisData/ArticleV1/Assignments/A00.json` … `A15.json`: acquisition-disjoint views.

## Classifier artifacts

- `Output/ArticleV1/Models/WindowClassifiers/`: persisted preprocessing + classifier pipelines.
- `Output/ArticleV1/Results/WindowPredictions/`: evidence-safe prediction records; no true label.
- `Output/ArticleV1/Results/AcquisitionPredictions/`: mean-probability acquisition aggregation.
- Label-bearing evaluation tables are separate and must not be consumed by the agent.

## Evidence artifacts

- `Output/ArticleV1/OOD/`: training-distribution deviation scorers and held-out scores.
- `Output/ArticleV1/HealthyReferences/`: assignment-specific training-only Healthy references.
- `Output/ArticleV1/Results/TrendEvidence/`: scalar ordered-window temporal evidence.
- `Output/ArticleV1/Results/WindowEvidence/`: per-window evidence projections.
- `Output/ArticleV1/Results/AcquisitionEvidence/`: acquisition-level evidence.
- `Output/ArticleV1/Results/RiskEvidence/`: decision-free risk decomposition.
- `Output/ArticleV1/Results/SafetyEvidence/`: SafetyGuard input projections; not decisions.
- `Output/ArticleV1/EvidenceBundles/`: immutable, hash-linked case bundles.

## DEGA artifacts

- `Output/ArticleV1/AgentRuns/MinimalE2/`: one audit JSONL per scenario/policy run.
- `Output/ArticleV1/Tables/minimal_e2_*`: state paths, outcomes and SafetyGuard summaries.

## Figures

- `Output/ArticleV1/Figures/`: intermediate publication figure assets.
- `figures_m3/output/`: final manuscript-aligned computational Figures 6–14.
- `figures_m3/output/figure_data/`: exact plotted values.

## Interpretation boundaries

- `combined_risk` is not failure probability.
- distribution deviation is an OOD surrogate, not a calibrated unknown-fault probability.
- temporal derivatives are with respect to ordered-window position, not physical degradation time.
- SafetyEvidence is an input projection, not an executed SafetyGuard decision.
