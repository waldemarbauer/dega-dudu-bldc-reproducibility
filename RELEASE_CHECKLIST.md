# Public release checklist

- [ ] Select and insert the intended software license. The current `LICENSE`
      file intentionally preserves the placeholder from the supplied project.
- [ ] Run `uv sync --all-groups` with Python 3.11 or 3.12.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run mypy`.
- [ ] Run `uv run pytest -q`.
- [ ] Fetch and verify DUDU-BLDC with `scripts/00_fetch_dataset.py`.
- [ ] Execute `scripts/reproduce_all.py` from a clean clone.
- [ ] Confirm expected counts in `reference/expected_article_results.json`.
- [ ] Validate manuscript-aligned Figures 6–14 using
      `figures_m3/scripts/validate_m3_figures.py`.
- [ ] Review the MCMC routing fallback note in `docs/VALIDATION.md` and replace
      the snapshot with the original posterior artifact only when redistribution
      and storage policy allow it.
- [ ] Confirm final article bibliographic metadata and update `CITATION.cff`.
- [ ] Rebuild `REPOSITORY_MANIFEST.json`, `FILE_INVENTORY.csv`, and
      `CHECKSUMS.sha256` after any release edits.
