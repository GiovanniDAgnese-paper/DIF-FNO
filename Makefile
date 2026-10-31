# DIF-FNO -- developer entry points.  Run `make help` for the list of targets.
#
# `PY` picks up a local virtual environment automatically (./.venv, ../.venv),
# so `make stats` works in a sandbox where the interpreter is not on PATH;
# override it explicitly with `make stats PY=/path/to/python`.
VENV    := $(firstword $(wildcard .venv/bin/python ../.venv/bin/python))
PY      ?= $(if $(VENV),$(VENV),python)
SRC     := src
TESTS   := tests
DATA    := data
RESULTS := results/repro
DOMAINS ?= star lshape annulus naca0012 slit double_hole

.PHONY: help install install-dev test lint format data benchmark figures study paper-tables stats timing clean clean-data clean-results

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
	@echo "  study          deterministic verification of P5-P8 -> results/study"
	@echo "  paper-tables   rebuild paper/tables + results/v2/TABLE.md"
	@echo "  stats          every statistic quoted in the paper -> results/v2/STATS.*"
	@echo "  timing         wall-clock cost of the chart (the number quoted in the docs)"
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
		--variants fno dif-barrier chart-fixed chart-eq --seeds 0 1 2 \
		--epochs 40 --eval-every 5 --out results/v2

figures:
	$(PY) scripts/make_figures.py --results $(RESULTS) --out figures/repro

study:
	$(PY) scripts/run_chart_study.py --out results/study --figures figures/study

paper-tables:
	$(PY) scripts/make_paper_table.py --results results/v2

stats:
	$(PY) scripts/report_stats.py --results results/v2

timing:
	$(PY) scripts/bench_timing.py --resolution 64 --batch 10 --threads 2

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

clean-data:
	rm -rf $(DATA)

clean-results:
	rm -rf $(RESULTS)
