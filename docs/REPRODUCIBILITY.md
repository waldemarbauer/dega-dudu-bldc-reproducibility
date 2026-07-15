# Reproducibility guide

## Scope

This repository reproduces the bounded computational case study described in
the July 15, 2026 manuscript version. It is designed to regenerate scientific
artifacts and manuscript result figures from the pinned DUDU-BLDC source data.

## Environment

Use Python 3.11 or 3.12.

```bash
uv sync --all-groups
```

The `mcmc` dependency group is optional for the ArticleV1 publication-replay
path because the source package did not include the historical posterior trace.
It is retained for users who possess the original transition-policy training
artifacts and wish to rerun that separate MCMC study.

## Full ordered workflow

### Stage 0 — Dataset acquisition

```bash
uv run python scripts/00_fetch_dataset.py
```

Downloads Zenodo record 15522163, verifies the pinned MD5, computes SHA-256,
and extracts the archive to `Data/InputData/External/DUDU-BLDC-v1/`.

### Stage 1 — Canonical windows, features and assignment views

```bash
uv run python scripts/01_prepare_windows.py
```

Creates 25 non-overlapping 40,000-sample windows for each of eight raw
acquisitions, computes raw-window hashes, extracts 28 deterministic case-local
features, and creates all 16 exhaustive acquisition-disjoint assignment views.

Expected counts: 8 acquisitions, 200 windows, 28 features, 16 views, 100 train
and 100 held-out windows per view.

### Stage 2 — Classifier execution

```bash
uv run python scripts/02_train_classifiers.py
```

Runs Logistic Regression, Random Forest and Histogram Gradient Boosting for
A00–A15. No held-out acquisition may influence fitting, preprocessing, tuning,
calibration or model selection.

Expected: 48 models, 48 reload-equivalence checks, 4,800 held-out window
predictions, 192 acquisition-level mean-probability aggregations.

### Stage 3 — Real evidence layer

```bash
uv run python scripts/03_generate_evidence.py
```

Expected: 16 training-distribution deviation scorers, 1,600 held-out deviation
records, 16 training-only Healthy references, 256 scalar temporal trend records,
4,800 WindowEvidence, 192 AcquisitionEvidence, 768 RiskEvidence, 768
SafetyEvidence projections and 768 EvidenceBundles.

### Stage 4 — Bounded XAI

```bash
uv run python scripts/04_run_xai.py
```

Runs only the paper methods: global permutation importance for the frozen A00
Random Forest identity, exact local Logistic Regression logit contributions,
and descriptive top-10 feature-set stability across the 16 dependent assignment
views.

### Stage 5 — Representative DEGA executions

```bash
uv run python scripts/05_run_dega.py
```

Runs three persisted natural scenario labels under four routing policies,
producing 12 workflow executions with FSM paths, SafetyGuard evaluations,
hash-linked audit chains and deterministic replay.

The three scenario labels are based on only two distinct EvidenceBundle hashes
in the supplied manuscript study; this is disclosed in the paper figure config
and should not be interpreted as three statistically independent cases.

### Stage 6 — Manuscript figures

```bash
uv run python scripts/06_generate_figures.py
```

Generates intermediate ArticleV1 publication assets and the nine computational
figures corresponding to manuscript Figures 6–14.

## Clean rerun

Generated files are ignored. To remove them:

```bash
make clean-generated
```

Then rerun the stages from 0 or from a later stage when prerequisites exist.

## Verification

```bash
uv run pytest -q
uv run ruff check .
```

After a complete run, compare counts and high-level results with
`reference/expected_article_results.json`.
