.PHONY: sync test quality fetch prepare classify evidence xai dega figures reproduce clean-generated

sync:
	uv sync --all-groups

test:
	uv run pytest -q

quality:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src tests scripts Scripts
	uv run pytest -q

fetch:
	uv run python scripts/00_fetch_dataset.py

prepare:
	uv run python scripts/01_prepare_windows.py

classify:
	uv run python scripts/02_train_classifiers.py

evidence:
	uv run python scripts/03_generate_evidence.py

xai:
	uv run python scripts/04_run_xai.py

dega:
	uv run python scripts/05_run_dega.py

figures:
	uv run python scripts/06_generate_figures.py

reproduce:
	uv run python scripts/reproduce_all.py

clean-generated:
	rm -rf Output/ArticleV1 Data/AnalysisData/ArticleV1 Data/AnalysisData/V2 figures_m3/output figures_m3/data_manifest
	mkdir -p Output Data/AnalysisData Data/IntermediateData figures_m3/output figures_m3/data_manifest
	touch Output/.gitkeep Data/AnalysisData/.gitkeep Data/IntermediateData/.gitkeep figures_m3/output/.gitkeep figures_m3/data_manifest/.gitkeep
