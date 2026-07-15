# Validation status of this extracted publication repository

This repository was synthesized from the supplied project ZIP and aligned to the
July 15, 2026 manuscript version. The validation performed during extraction was
intentionally limited to checks that can be run without the external DUDU-BLDC
dataset and without a compatible Python 3.11/3.12 scientific environment.

## Checks completed during extraction

| Check | Result |
|---|---|
| Python byte-code compilation of `src/`, `scripts/`, `Scripts/`, `figures_m3/` | PASS |
| Public repository contract tests | PASS — 5 tests |
| Curated internal import closure | PASS — 0 unresolved `trustworthy_agent` imports |
| Referenced configuration literals | PASS — 0 missing files |
| Machine-local absolute path scan | PASS — no machine-local home, temporary-runtime, or extraction-work paths embedded |
| Generated output exclusion | PASS — no trained models, Parquet experiment results, EvidenceBundles, or generated figures included |
| JSON/TOML/YAML parse checks | PASS |
| Repository checksum manifest | PASS after package finalization |

## Checks not completed in this environment

### Full scientific reproduction

Not run. A complete execution requires the external DUDU-BLDC dataset and a
compatible Python 3.11 or 3.12 environment with the project dependencies.
Therefore this extraction does **not** claim that a clean clone was executed
end-to-end from raw data through all manuscript figures inside this runtime.

### Ruff and mypy

The extraction runtime did not provide the project development environment, so
`ruff` and full `mypy` validation were not executed here. The public repository
includes both tools in the `dev` dependency group and CI configuration.

### `uv.lock` resolution

The source project supplied an existing `uv.lock`. It was retained for
publication packaging and minimally aligned with the explicit direct dependency
list. The extraction runtime had Python 3.13 only, while the project requires
Python 3.11 or 3.12, so the lock file was not freshly re-resolved here.

## Publication-specific MCMC routing fallback

The supplied ZIP contained completed ArticleV1 E2 audit records but not the
historical PyMC/NUTS posterior file
`Output/Models/TransitionPolicies/BayesianMCMC/posterior_trace.nc`.

For reproducibility of the 12 representative manuscript DEGA runs, the public
repository includes:

`reference_inputs/transition_policies/article_v1_routing_snapshot.json`

The Stage 5 runner uses the original posterior when present; otherwise it uses
the frozen routing snapshot to reproduce the representative workflow routing.
This fallback is explicitly a **routing replay input**, not a new Bayesian fit.

## Release blockers outside code validation

Before public release, the repository owner should:

1. choose and insert the intended public software license;
2. run the full workflow from a clean clone under Python 3.11 or 3.12 with the
   pinned DUDU-BLDC dataset;
3. confirm the final generated figure hashes or numerical figure-data against
   the manuscript production artifacts;
4. regenerate `uv.lock` in the release environment if dependency metadata is
   changed.
