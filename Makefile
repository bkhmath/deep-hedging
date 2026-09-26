.PHONY: help install test lint format data study study-offline study-quick clean

help:
	@echo "install       install the package and development dependencies"
	@echo "test          run the test suite (no network, no cached quotes needed)"
	@echo "lint          ruff check and format check"
	@echo "format        apply ruff formatting"
	@echo "data          download and cache one Deribit BTC option surface"
	@echo "study         run the full study on the cached surface (~25 min)"
	@echo "study-offline run the full study on a synthetic surface, no download"
	@echo "study-quick   run the study at small sizes, for a smoke test"
	@echo "clean         remove caches and build artefacts"

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

data:
	python scripts/download_surface.py

study:
	python scripts/run_study.py

study-offline:
	python scripts/run_study.py --synthetic

study-quick:
	python scripts/run_study.py --quick --synthetic

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
