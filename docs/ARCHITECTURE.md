# DEGA architecture summary

The publication workflow separates evidence generation from deterministic
workflow governance.

```text
DUDU-BLDC acquisitions
  → canonical 0.8-s windows
  → 28 deterministic features
  → classifier probabilities
  → WindowEvidence
  → AcquisitionEvidence
  → temporal TrendEvidence
  → training-distribution deviation + training-only Healthy reference
  → RiskEvidence
  → SafetyEvidence projection
  → immutable EvidenceBundle
  → routing policy proposal
  → external higher-priority SafetyGuard
  → canonical 11-state DEGA FSM
  → recommendation / escalation / no decision
  → mandatory audit
  → deterministic replay
```

Canonical states:

`S0 DATA_ACQUISITION`, `S1 DATA_VALIDATION`, `S2 FEATURE_EXTRACTION`,
`S3 SPLINE_MODELLING`, `S4 DIAGNOSTIC_INFERENCE`,
`S5 EXPLANATION_GENERATION`, `S6 DECISION_CHECK`, `S7 RECOMMENDATION`,
`S8 ESCALATION`, `S9 NO_DECISION`, `S10 AUDIT`.

Workflow-routing probabilities are not physical motor-degradation probabilities.
