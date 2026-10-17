# DIF-FNO -- developer entry points.  Run `make help` for the list of targets.
PY      ?= python
SRC     := src
TESTS   := tests
DATA    := data
RESULTS := results/repro
DOMAINS ?= star lshape annulus naca0012

.PHONY: help install install-dev test lint format data benchmark figures clean clean-data clean-results

help:
	@echo "DIF-FNO targets"
	@echo "  install        install the package (editable)"
	@echo "  install-dev    install with the [dev] extra"
	@echo "  test           run the test suite"
	@echo "  lint           ruff check"
	@echo "  format         ruff --fix"
	@echo "  data           generate the Darcy datasets (DOMAINS='$(DOMAINS)')"
	@echo "  benchmark      full reproduction sweep -> $(RESULTS)"
	@echo "  figures        rebuild the figures from $(RESULTS)"
	@echo "  clean          remove caches and build artefacts"

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest $(TESTS) -v

test-cov:
	$(PY) -m pytest $(TESTS) --cov=dagnese_fno --cov-report=term-missing

lint:
	$(PY) -m ruff check $(SRC) $(TESTS) scripts

format:
	$(PY) -m ruff check --fix $(SRC) $(TESTS) scripts

data:
	$(PY) scripts/generate_data.py --domains $(DOMAINS) --n 64 --samples 100 --out $(DATA)

benchmark:
	$(PY) scripts/run_benchmark.py --domains $(DOMAINS) \
		--variants fno dif-barrier dif-certified --seeds 0 1 2 --epochs 60 --out $(RESULTS)

figures:
	$(PY) scripts/make_figures.py --results $(RESULTS) --out figures/repro

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

clean-data:
	rm -rf $(DATA)

clean-results:
	rm -rf $(RESULTS)
